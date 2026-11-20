"""Live agent: WebSocket protocol, voice turn-taking, corrections, replies.

Voice design (honest): this is *half-duplex with barge-in*, not full-duplex.
  * Browser: mic -> client-side VAD -> utterance PCM -> server Whisper.
  * While the agent speaks, the browser keeps listening. Speech onset triggers
    `barge_in`: TTS playback is cut client-side, the server cancels the pending
    reply task and drops queued guidance, then handles the new utterance.
  * Agent replies are templated from *structured results* (counts etc. come from
    the inventory). The local Qwen is used only to map unrecognised phrasing to a
    closed action set (corrections.py).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from ..claim.calculations import book_excluded
from ..pipeline import postprocess
from ..storage import storage
from ..voice import stt, tts
from . import corrections
from .session import SweepSession

log = logging.getLogger(__name__)
SESSIONS: dict[str, SweepSession] = {}
PACKET_TASKS: dict[str, asyncio.Task] = {}
SPOKEN = {"intro", "reply", "guidance", "narration"}   # the agent talks while you sweep; after Stop only the final summary is spoken
WHISPER_NOISE = {"thank you", "thanks for watching", "you", "bye", "okay", "ok", "uh", "um", "hmm"}

INTRO = ("Walk slowly along your shelves with the camera pointed at them. Don't pull anything out. "
         "Say a correction any time, like 'that's a first edition', and say 'done' when you finish. I'll process everything after that.")


class LiveAgent:
    def __init__(self, ws: WebSocket, session: SweepSession):
        self.ws, self.s = ws, session
        self.reply_task: Optional[asyncio.Task] = None
        self.say_q: asyncio.Queue = asyncio.Queue()
        self.speaking = False
        self.frame_slot: Optional[tuple] = None
        self.frame_evt = asyncio.Event()
        self.closed = False
        session.emit_cb = self.send
        session.narrate_cb = lambda text: self.say(text, "narration")
        # Live PREVIEW (when recording): a second, light session analyses ~1 frame/s from memory so the screen fills in and the
        # agent can direct the capture. It never writes frames or feeds the packet: the packet comes from the batch pass after Stop.
        self.pv: Optional[SweepSession] = None
        self.pv_busy, self.pv_last = False, 0.0
        if session.batch_mode:
            from ..config import get_settings
            if get_settings().lca_live_preview:
                pv = SweepSession(session.id, session.country, session.currency, session.device, session.camera_height_cm)
                pv.batch_mode, pv.persist_frames, pv.frame_prefix, pv.t0 = False, False, f"{session.id}_pv", session.t0
                pv.emit_cb, pv.narrate_cb = self.send, (lambda text: self.say(text, "narration"))
                self.pv = pv

    async def send(self, msg: dict):
        if self.closed:
            return
        try:
            await self.ws.send_text(json.dumps(msg, default=str))
        except Exception:
            self.closed = True

    # ------------- speech -------------
    async def say(self, text: str, kind: str = "agent", interrupt: bool = False):
        """Queue speech. Guidance never talks over the user; replies to the user do."""
        if self.s.user_speaking and not interrupt:
            return
        speak = kind in SPOKEN and not self.s.ended_at   # after "stop" everything is text-only; only the conversation is voiced; guidance/narration stay as on-screen text
        audio = await asyncio.to_thread(tts.synthesize_b64, text) if speak and tts.engine() == "kokoro" else None
        await self.send({"type": "agent_say", "text": text, "kind": kind, "speak": speak, "audio_wav_b64": audio})

    async def on_guidance_event(self):  # guidance emitted by session -> also spoken
        pass

    # ------------- inbound -------------
    async def run(self):
        s = self.s
        s.start_worker()
        if self.pv:
            self.pv.start_worker()
        await self.send({"type": "ready", "sweep_id": s.id, "tts": tts.engine()})
        await self.say(f"I have you in {s.country}, pricing in {s.currency}. " + INTRO, "intro")
        proc = asyncio.create_task(self._frame_loop())
        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    m = json.loads(raw)
                    t = m.get("type")
                    if not isinstance(m, dict) or not t:
                        raise ValueError("no type")
                except Exception:
                    log.warning("ignoring malformed WS message (%d bytes)", len(raw))
                    continue
                try:
                    await self._dispatch(t, m)
                except WebSocketDisconnect:
                    raise
                except Exception:
                    log.exception("bad %s message ignored", t)
                    continue
        except WebSocketDisconnect:
            log.info("ws disconnected for %s", s.id)
        finally:
            self.closed = True
            proc.cancel()

    async def _dispatch(self, t: str, m: dict):
        s = self.s
        if t == "frame":
            if s.ended_at:
                return   # recording is over: late frames from the client are ignored
            if self.frame_slot is not None:
                s.dropped_frames += 1  # previous frame never got processed: counted, reported in the packet
            self.frame_slot = (base64.b64decode(m["data"]), m.get("ts"), m.get("orientation"))
            self.frame_evt.set()
        elif t == "utterance":
            pcm = base64.b64decode(m["pcm16_b64"])
            if len(pcm) > 16000 * 2 * 30:   # >30 s: not a command
                await self.send({"type": "transcript", "text": "", "final": True}); return
            asyncio.create_task(self._utterance(pcm, int(m.get("sample_rate", 16000))))
        elif t == "text":
            await self._handle_text(m["text"])
        elif t == "speech_start":
            s.user_speaking = True
            await self.barge_in()
        elif t == "speech_end":
            s.user_speaking = False
        elif t == "barge_in":
            await self.barge_in()
        elif t == "mark_corner":
            await self._mark(ceiling=bool(m.get("ceiling")))
        elif t == "set_camera_height":
            cm = float(m["cm"])
            if not (30 <= cm <= 250):
                await self.say("That camera height doesn't look right. Give it in centimetres, between 30 and 250.", "reply", True); return
            s.camera_height_cm = cm; await s.emit("event", text=f"Camera height set to {s.camera_height_cm:.0f} cm")
        elif t == "stop":
            await self.finish()
            # keep the socket open so packet_ready / processing events reach the client

    async def _frame_loop(self):
        """Single-slot latest-frame queue: if processing is slower than capture we drop old frames (counted, reported)."""
        while True:
            await self.frame_evt.wait()
            slot, self.frame_slot = self.frame_slot, None
            self.frame_evt.clear()
            if slot is None:
                continue
            try:
                g = await (self.s.record_frame(*slot) if self.s.batch_mode else self.s.on_frame(*slot))
                if g:
                    await self.say(g["text"], "guidance")
                if self.pv and not self.pv_busy and not self.s.ended_at and time.time() - self.pv_last >= 1.0:
                    self.pv_busy, self.pv_last = True, time.time()
                    asyncio.create_task(self._preview(slot))
            except Exception:
                log.exception("frame processing failed")
            if self.frame_slot is not None:
                self.frame_evt.set()

    async def _preview(self, slot: tuple):
        try:
            g = await self.pv.on_frame(*slot)
            if g and not self.s.ended_at:
                await self.say(g["text"], "guidance")
        except Exception:
            log.exception("preview frame failed")
        finally:
            self.pv_busy = False

    async def barge_in(self):
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        await self.send({"type": "stop_speaking"})

    async def _utterance(self, pcm: bytes, sr: int):
        text = await asyncio.to_thread(stt.transcribe_pcm16, pcm, sr)
        if text and text.strip().lower().strip(".! ") in WHISPER_NOISE:
            text = None   # Whisper hallucinates these on silence/noise
        await self.send({"type": "transcript", "text": text or "", "final": True})
        if text:
            await self._handle_text(text)

    async def _handle_text(self, text: str):
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        self.reply_task = asyncio.create_task(self._respond(text))

    async def _mark(self, ceiling: bool):
        if self.s.batch_mode:   # the phone's orientation at this instant is kept; the room is computed after Stop
            import copy
            self.s.deferred.append({"ts": self.s.last_frame_ts or time.time(), "action": "mark_ceiling" if ceiling else "mark_corner",
                                    "orientation": copy.deepcopy(self.s.last_orientation)})
            await self.say(f"{'Ceiling' if ceiling else 'Floor'} point noted.", "reply", True); return
        err = self.s.add_floor_mark(ceiling)
        if err:
            await self.say("I can't read your phone's orientation sensor, so I can't measure the room from this device. I'll mark the room dimensions as unavailable.", "reply", True)
        else:
            n = len(self.s.ceil_obs if ceiling else self.s.floor_obs)
            await self.s.emit("event", text=f"{'Ceiling' if ceiling else 'Floor'} corner {n} marked")
            await self.say(f"{'Ceiling' if ceiling else 'Floor'} point {n} marked.", "reply", True)

    # ------------- understanding + acting -------------
    async def _respond(self, text: str):
        try:
            acts = await corrections.parse(text)
            s = self.s
            if not acts:
                if getattr(self.pv or s, "pending_artwork", None) and any(w in text.lower() for w in ("print", "original")):
                    acts = corrections.parse_regex("it's a print" if "print" in text.lower() else "it's an original painting")
                if not acts:
                    await self.say("I didn't catch an inventory instruction there. You can say things like 'that's a first edition' or 'skip this shelf'.", "reply", True)
                    return
            replies = []
            for a in acts:
                replies.append(await self.apply(a))
            await self.say(" ".join(r for r in replies if r), "reply", True)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("respond failed")

    DEFERRABLE = {"set_first_edition", "set_not_mine", "skip_shelf", "set_artwork_type", "known_shelf_dimension"}

    async def apply(self, a: corrections.Action) -> str:
        s = self.s
        await s.emit("correction", action=a.action, target=a.target, value=a.value, raw=a.raw, source=a.source)
        if s.batch_mode and a.action in self.DEFERRABLE:   # applied to the book on screen at THIS moment when the sweep is analysed
            s.deferred.append({"ts": s.last_frame_ts or time.time(), "action": a.action, "target": a.target, "value": a.value, "number": a.number})
            return "Noted. I'll apply that to what's on screen right now when I process the sweep."
        if s.batch_mode and a.action in ("query_count", "query_status"):
            if self.pv:
                c = self.pv.counts(); return f"So far I see about {c['books']} books and {c['items']} other items. The exact count comes after you press Stop."
            return "I'll count everything after you press Stop."
        if a.action == "set_first_edition":
            trs = s.focus_tracks(a.target == "group")
            for t in trs: t.meta["first_edition"] = True
            await s._state(True)
            return (f"Noted: {len(trs)} book{'s' if len(trs) != 1 else ''} marked first edition. I won't price {'them' if len(trs) != 1 else 'it'} automatically; it goes to appraisal." if trs
                    else "I can't tell which book you mean. Point the camera at it and say it again.")
        if a.action == "set_not_mine":
            trs = s.focus_tracks(a.target == "group") or s.focus_tracks(False, "item")
            for t in trs: t.meta["excluded"] = True; t.meta["owner_note"] = "owner states it belongs to someone else"
            await s._state(True)
            return f"Excluded {len(trs)} item{'s' if len(trs) != 1 else ''} from your claim." if trs else "I can't tell what you're referring to. Point at it and say it again."
        if a.action == "set_artwork_type":
            if s.batch_mode and self.pv is not None:     # answer the art question now; applied to the artwork on screen at this moment
                if getattr(self.pv, "pending_artwork", None): self.pv.pending_artwork = []
                s.deferred.append({"ts": s.last_frame_ts or time.time(), "action": "set_artwork_type", "target": a.target, "value": a.value, "number": a.number})
                return "Got it, that's a print. I'll price it from listings." if a.value == "print" else "Got it, an original. It goes to human appraisal and won't be auto-priced."
            ids = getattr(s, "pending_artwork", [])
            trs = [s.objs.tracks[i] for i in ids if i in s.objs.tracks] or s.focus_tracks(False, "item")
            for t in trs: t.meta["user_note"] = a.value or ""
            if getattr(s, "pending_artwork", None): s.pending_artwork = []
            return ("Got it, that's a print. I'll price it from listings." if a.value == "print"
                    else "Got it, an original. It will go to human appraisal and won't be auto-priced.") if trs else "I don't see an artwork to update."
        if a.action == "skip_shelf":
            trs = s.focus_tracks(True)
            for t in trs: t.meta["excluded"] = True; t.meta["owner_note"] = "shelf skipped by owner"
            await s._state(True)
            return f"Skipping this shelf: {len(trs)} books removed from the inventory."
        if a.action == "set_camera_height" and a.number:
            if not (30 <= a.number <= 250):
                return "That camera height doesn't look right. Give it in centimetres, between 30 and 250."
            s.camera_height_cm = a.number
            return f"Camera height set to {a.number:.0f} centimetres."
        if a.action == "known_shelf_dimension" and a.number:
            ok, msg = s.calibrate_known_dimension(a.number, a.value or "height")
            return msg
        if a.action in ("mark_corner", "mark_ceiling"):
            await self._mark(a.action == "mark_ceiling"); return ""
        if a.action == "query_count":
            c = s.counts(); return f"So far I count {c['books']} books on {c['shelves']} shelves and {c['items']} other items."
        if a.action == "query_status":
            c = s.counts(); return f"{c['books']} books, {c['identified']} identified so far, {c['items']} other items, {c['calibrated_segments']} calibrated walls."
        if a.action == "stop":
            asyncio.create_task(self.finish()); return "Finishing the sweep and building your claim packet."
        return ""

    # ------------- end of sweep -------------
    async def finish(self):
        s = self.s
        if s.ended_at:
            return
        s.ended_at = time.time()
        if self.pv and self.pv._worker:
            self.pv._worker.cancel()
        s.store.update_sweep(s.id, state="processing", meta={"duration_s": s.elapsed})
        await self.send({"type": "sweep_ended", "duration_s": round(s.elapsed, 1)})
        await self.say("Sweep finished. I'm now identifying and pricing everything; it takes a few minutes.", "reply", True)
        PACKET_TASKS[s.id] = asyncio.create_task(postprocess.run(s))
