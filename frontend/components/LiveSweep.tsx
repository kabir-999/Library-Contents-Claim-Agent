"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { WS } from "@/lib/api";
import { fromEvent, Orient } from "@/lib/orientation";
import { pcmToB64, startVad } from "@/lib/vad";
import { deviceProfile, EdgeFilter } from "@/lib/edge";
import { say as speakText, unlockSpeech } from "@/lib/voice";
import type { FeedEntry, LiveState } from "@/lib/types";
import type { StartResult } from "./StartClaim";

const FPS_INTERVAL_MS = 350;   // ≈3 fps; server keeps only the newest frame if it is busy

export default function LiveSweep({ start, onEnded }: { start: StartResult; onEnded: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const ws = useRef<WebSocket | null>(null);
  const orient = useRef<Orient | null>(null);
  const audioEl = useRef<HTMLAudioElement | null>(null);
  const agentSpeaking = useRef(false);
  const muted = useRef(false);
  const [state, setState] = useState<LiveState | null>(null);
  const [feed, setFeed] = useState<FeedEntry[]>([]);
  const [agent, setAgent] = useState<"listening" | "speaking" | "you" | "processing">("listening");
  const [guidance, setGuidance] = useState("");
  const [recorded, setRecorded] = useState(0);
  const [found, setFound] = useState<{ books: any[]; items: any[] }>({ books: [], items: [] });
  const guideTimer = useRef<any>(null);
  const [stage, setStage] = useState<Record<string, string>>({});
  const [ended, setEnded] = useState(false);
  const [text, setText] = useState("");
  const [camH, setCamH] = useState(start.cameraHeightCm ? String(start.cameraHeightCm) : "");
  const [elapsed, setElapsed] = useState(0);
  const [mute, setMute] = useState(false);
  const n = useRef(0);
  const capturing = useRef(true);
  const afterStop = useRef(false);
  const sentN = useRef(0);
  const t0Rec = useRef(performance.now());
  const lastSay = useRef("");
  const [camLost, setCamLost] = useState(false);
  const [voiceBlocked, setVoiceBlocked] = useState(false);
  const clockRef = useRef<any>(null);
  const stopVadRef = useRef<() => void>(() => {});
  const [glassesOn, setGlassesOn] = useState(false);   // set after mount (device type is only known in the browser)
  const edge = useRef(new EdgeFilter());
  const intervalMs = useRef(150);   // ~6.7 frames per second
  const lastDropped = useRef(0);
  const [edgeStat, setEdgeStat] = useState("");

  const push = useCallback((kind: FeedEntry["kind"], text: string) => setFeed(f => [{ id: n.current++, kind, text, t: Date.now() }, ...f].slice(0, 60)), []);
  const send = (o: object) => ws.current?.readyState === WebSocket.OPEN && ws.current.send(JSON.stringify(o));

  const stopCapture = useCallback(() => {
    if (!capturing.current) return;
    capturing.current = false;                                   // no more frames are sent
    clearInterval(clockRef.current); stopVadRef.current();       // timer and microphone listener stop
    start.stream.getTracks().forEach(t => t.stop());             // camera + mic are released (indicator light goes off)
  }, [start.stream]);

  const stopSpeech = useCallback(() => {
    window.speechSynthesis?.cancel(); audioEl.current?.pause(); agentSpeaking.current = false; setAgent(a => (a === "speaking" ? "listening" : a));
  }, []);

  const speak = useCallback((m: any) => {
    if (muted.current || m.speak === false) return;
    if (afterStop.current && !m.final) return;          // after Stop only the final summary is spoken
    stopSpeech(); agentSpeaking.current = true; setAgent("speaking");
    const done = () => { agentSpeaking.current = false; setAgent(a => (a === "speaking" ? "listening" : a)); };
    if (m.audio_wav_b64) { const a = new Audio("data:audio/wav;base64," + m.audio_wav_b64); audioEl.current = a; a.onended = done; a.onerror = done; a.play().catch(done); }
    else if (window.speechSynthesis) { lastSay.current = m.text; if (!speakText(m.text, done, () => { done(); setVoiceBlocked(true); })) done(); }
    else done();
  }, [stopSpeech]);

  useEffect(() => {
    const v = video.current!; v.srcObject = start.stream; v.play().catch(() => {});
    const sock = new WebSocket(`${WS}/ws/sweeps/${start.sweepId}`); ws.current = sock;
    sock.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "state") { setState(m); const d = Number(m.dropped_frames || 0);
        intervalMs.current = d > lastDropped.current ? Math.min(250, intervalMs.current * 1.15) : Math.max(deviceProfile().intervalMs, intervalMs.current * 0.97); lastDropped.current = d; }
      else if (m.type === "event") push("event", m.text);
      else if (m.type === "guidance") { setGuidance(m.text); clearTimeout(guideTimer.current); guideTimer.current = setTimeout(() => setGuidance(""), 6000); }
      else if (m.type === "recording") setRecorded(m.frames);
      else if (m.type === "found") setFound({ books: m.books, items: m.items });
      else if (m.type === "agent_say") { if (m.kind !== "guidance") push("agent", m.text); speak(m); }
      else if (m.type === "transcript") push("user", m.text || "(not understood)");
      else if (m.type === "correction") push("correction", `${m.action}${m.value ? " → " + m.value : ""} (${m.target})`);
      else if (m.type === "stop_speaking") stopSpeech();
      else if (m.type === "processing") { setStage(s => ({ ...s, [m.stage]: m.status })); }
      else if (m.type === "sweep_ended") { stopCapture(); setEnded(true); setAgent("processing"); stopSpeech(); afterStop.current = true; }
      else if (m.type === "packet_ready") { push("processing", `Packet ready in ${Number(m.seconds).toFixed(0)} s`); setTimeout(onEnded, 1500); }
    };
    sock.onclose = () => push("event", "Connection closed");

    const onOri = (e: DeviceOrientationEvent) => { orient.current = fromEvent(e); };
    window.addEventListener("deviceorientation", onOri);

    const prof = deviceProfile(); let stopped = false;
    const tick = () => {
      if (stopped || !capturing.current) return;
      setTimeout(tick, intervalMs.current);
      const v = video.current;
      if (!v || sock.readyState !== WebSocket.OPEN || v.videoWidth === 0) return;
      sentN.current++; if (sentN.current % 20 === 0) setEdgeStat(`recording ~${(sentN.current / Math.max(1, (performance.now() - t0Rec.current) / 1000)).toFixed(1)} frames/s (${sentN.current} sent)`);
      const c = canvas.current!; const w = Math.min(prof.width, v.videoWidth), h = Math.round(w * v.videoHeight / v.videoWidth);
      c.width = w; c.height = h; c.getContext("2d")!.drawImage(v, 0, 0, w, h);
      c.toBlob(bl => { if (!bl) return; const r = new FileReader(); r.onload = () => {
        const data = (r.result as string).split(",")[1];
        send({ type: "frame", ts: Date.now() / 1000, orientation: orient.current, data });
      }; r.readAsDataURL(bl); }, "image/jpeg", prof.quality);
    };
    setTimeout(tick, 500);
    const timer = setInterval(() => {}, 1 << 30);
    const clock = setInterval(() => setElapsed(e => e + 1), 1000); clockRef.current = clock;

    let stopVad = () => {};
    startVad(start.stream, {
      isAgentSpeaking: () => agentSpeaking.current,
      onStart: () => { if (agentSpeaking.current) { stopSpeech(); send({ type: "barge_in" }); } send({ type: "speech_start" }); setAgent("you"); },
      onEnd: () => { send({ type: "speech_end" }); setAgent("processing"); setTimeout(() => setAgent(a => (a === "processing" ? "listening" : a)), 3000); },
      onUtterance: (pcm) => send({ type: "utterance", sample_rate: 16000, pcm16_b64: pcmToB64(pcm) }),
    }).then(f => { stopVad = f; stopVadRef.current = f; }).catch(e => push("event", "Microphone VAD unavailable: " + e));

    return () => { stopped = true; clearInterval(timer); clearInterval(clock); window.removeEventListener("deviceorientation", onOri); stopVad(); sock.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { const v = video.current; if (v && v.srcObject !== start.stream) { v.srcObject = start.stream; v.play().catch(() => {}); } }, [glassesOn, start.stream]);
  useEffect(() => { const p = deviceProfile(); intervalMs.current = p.intervalMs; if (p.kind === "glasses") setGlassesOn(true); }, []);
  useEffect(() => { const f = () => { unlockSpeech(); setVoiceBlocked(false); }; window.addEventListener("pointerdown", f, { once: true }); return () => window.removeEventListener("pointerdown", f); }, []);
  const restartCamera = useCallback(async () => {
    try {
      const ns = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } } });
      const nv = ns.getVideoTracks()[0];
      start.stream.getVideoTracks().forEach(t => { start.stream.removeTrack(t); t.stop(); });
      start.stream.addTrack(nv);
      const v = video.current; if (v) { v.srcObject = start.stream; await v.play().catch(() => {}); }
      setCamLost(false);
    } catch { setCamLost(true); }
  }, [start.stream]);
  useEffect(() => {   // phones can pause the preview (audio session changes, app switch): bring it back automatically
    const id = setInterval(() => {
      if (!capturing.current) return;
      const v = video.current, tr = start.stream.getVideoTracks()[0];
      if (!tr || tr.readyState === "ended") { setCamLost(true); return; }
      if (v && v.paused) v.play().catch(() => {});
      setCamLost(false);
    }, 1500);
    return () => clearInterval(id);
  }, [start.stream]);
  useEffect(() => { muted.current = mute; if (mute) stopSpeech(); }, [mute, stopSpeech]);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0"), ss = String(elapsed % 60).padStart(2, "0");
  const room = state?.room;
  const roomStatus = !room ? "…" : room.camera_height_cm == null ? "need camera height" : room.floor_corners < 3 ? `${room.floor_corners}/3+ floor corners` : room.ceiling_marks ? "floor + ceiling marked" : "floor marked; need ceiling corner";
  const color: Record<string, string> = { listening: "bg-emerald-500", speaking: "bg-sky-500", you: "bg-amber-500", processing: "bg-slate-400" };

  if (glassesOn) {
    const top = found.books.slice(-3).reverse();
    return (
      <div className="fixed inset-0 z-50 bg-black text-white">
        <video ref={video} muted playsInline className="h-full w-full object-cover opacity-90" />
        <canvas ref={canvas} className="hidden" />
        <div className="absolute inset-x-0 top-0 flex items-center justify-between bg-gradient-to-b from-black/80 to-transparent p-3 text-lg">
          <span>● {mm}:{ss}</span><span>{found.books.length} books · {found.items.length} items</span></div>
        <div className="absolute inset-x-0 bottom-0 space-y-2 bg-gradient-to-t from-black/90 to-transparent p-4">
          {guidance && <div className="rounded-lg bg-amber-500 p-3 text-2xl font-semibold text-black">{guidance}</div>}
          {top.length === 0 && !guidance && <div className="text-2xl text-slate-300">Recording… sweep slowly</div>}
          {top.map((b, i) => <div key={i} className="flex items-baseline justify-between gap-4 text-3xl font-bold leading-tight">
            <span className="truncate">{b.name}</span><span className="shrink-0 text-emerald-300">{b.price || "…"}</span></div>)}
          <div className="flex gap-3 pt-1">
            <button className="min-h-14 flex-1 rounded-xl bg-red-600 text-2xl font-bold" disabled={ended} onClick={() => { stopCapture(); send({ type: "stop" }); }}>{ended ? "Building packet…" : "Done"}</button>
            <button className="min-h-14 rounded-xl border border-white/40 px-5 text-xl" onClick={() => setGlassesOn(false)}>Normal view</button></div>
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
      <div className="space-y-3">
        {camLost && <div className="flex items-center justify-between gap-2 rounded-lg border border-red-300 bg-red-50 p-2 text-sm text-red-800"><span>The camera stopped.</span><button className="btn-ghost" onClick={restartCamera}>Restart camera</button></div>}
        {!mute && <div className="flex items-center justify-between gap-2 rounded-lg border border-sky-200 bg-sky-50 p-2 text-sm text-sky-900">
          <span>{voiceBlocked ? "Your phone blocked the agent's voice." : "Can't hear the agent?"} Check the volume and silent switch, then tap:</span>
          <button className="btn-ghost shrink-0" onClick={() => { unlockSpeech(); setVoiceBlocked(false); speakText(lastSay.current || "Voice check. Can you hear me?"); }}>🔊 Enable / test voice</button></div>}
        <div className="relative overflow-hidden rounded-xl bg-black">
          <video ref={video} muted playsInline autoPlay onPause={() => { if (capturing.current) video.current?.play().catch(() => {}); }} className="aspect-[4/3] w-full object-cover sm:aspect-video" />
          <canvas ref={canvas} className="hidden" />
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-sm text-white">
            <span className={`h-2.5 w-2.5 rounded-full ${ended ? "bg-slate-400" : "animate-pulse bg-red-500"}`} /> {ended ? "STOPPED" : "REC"} {mm}:{ss}</div>
          <div className="absolute right-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-sm text-white">
            <span className={`h-2.5 w-2.5 rounded-full ${color[agent]}`} /> {agent === "you" ? "hearing you…" : `agent ${agent}`}</div>
          {guidance && <div className="absolute inset-x-3 bottom-3 rounded-lg bg-amber-500/95 p-3 text-sm font-medium text-black shadow-lg transition-opacity">{guidance}</div>}
        </div>
        {edgeStat && <div className="text-xs text-slate-400">{edgeStat}</div>}
        <div className="grid grid-cols-2 gap-2">
          {found.books.length + found.items.length === 0 ? <Stat label="Frames recorded" value={recorded} sub="analysed after you stop" /> : <Stat label="Books" value={found.books.length} sub={`${found.books.filter(b => b.ok).length} confirmed`} />}
          {found.books.length + found.items.length === 0 ? <Stat label="Recording time" value={`${mm}:${ss}`} sub="keep sweeping slowly" /> : <Stat label="Other items" value={found.items.length} />}
        </div>
        <div className="card text-sm">
          <div className="mb-1 font-medium">Found so far</div>
          {found.books.length + found.items.length === 0 && <div className="text-slate-500">Recording. Sweep slowly across the shelves, then press <b>Stop sweep</b>. Every frame is analysed then and each book and item gets one entry.</div>}
          {found.books.map((b, i) => <div key={"b" + i} className="flex justify-between gap-3 border-b border-slate-100 py-1 last:border-0 dark:border-slate-800">
            <span><b>{b.name}</b>{b.author && <span className="text-slate-500"> · {b.author}</span>}{!b.ok && <span className="ml-1 text-xs text-amber-600">checking…</span>}</span>
            <span className="text-slate-600">{b.price || "…"}</span></div>)}
          {found.items.map((it, i) => <div key={"i" + i} className="flex justify-between gap-3 border-b border-slate-100 py-1 text-slate-600 last:border-0 dark:border-slate-800">
            <span>{it.name}</span><span>{it.price || "…"}</span></div>)}
        </div>
        <details className="card text-sm"><summary className="cursor-pointer font-medium">Room measurement (optional)</summary><div className="mt-2 flex flex-wrap items-center gap-2">
          <button className="btn-ghost" onClick={() => send({ type: "mark_corner" })} disabled={!start.orientationOk}>Mark floor corner</button>
          <button className="btn-ghost" onClick={() => send({ type: "mark_corner", ceiling: true })} disabled={!start.orientationOk}>Mark ceiling corner</button>
          <span className="flex items-center gap-1">Camera height cm
            <input className="w-20 rounded border p-1 dark:bg-slate-800" value={camH} onChange={e => setCamH(e.target.value)} onBlur={() => Number(camH) > 30 && send({ type: "set_camera_height", cm: Number(camH) })} /></span>
          <button className="btn-ghost" onClick={() => setGlassesOn(true)}>Glasses view</button>
          <label className="ml-auto flex items-center gap-1"><input type="checkbox" checked={mute} onChange={e => setMute(e.target.checked)} /> mute voice</label>
          {!start.orientationOk && <span className="w-full text-xs text-amber-600">No orientation sensor on this device: room dimensions will be reported unavailable (use a phone for room measurement).</span>}
        </div></details>
        <form className="flex flex-wrap gap-2" onSubmit={e => { e.preventDefault(); if (text.trim()) { send({ type: "text", text }); push("user", text); setText(""); } }}>
          <input className="min-w-0 flex-1 basis-full rounded border p-3 text-base sm:basis-0 sm:p-2 sm:text-sm dark:bg-slate-800" placeholder="Or type a correction (e.g. “those books belong to someone else”)" value={text} onChange={e => setText(e.target.value)} />
          <button className="btn-ghost min-h-11 flex-1 justify-center sm:flex-none">Send</button>
          <button type="button" className="btn min-h-11 flex-1 bg-red-600 hover:bg-red-700 sm:flex-none dark:bg-red-600 dark:text-white" disabled={ended} onClick={() => { stopCapture(); send({ type: "stop" }); }}>Stop sweep</button>
        </form>
      </div>
      <div className="space-y-3">
        {ended && <div className="card text-sm"><div className="mb-2 font-medium">Building claim packet…</div>
          {["frame_analysis", "book_identification", "book_pricing", "nonbook_classification", "nonbook_pricing", "measurement_validation", "room_measurement", "packet_generation"].map(s =>
            <div key={s} className="flex justify-between"><span>{s.replace(/_/g, " ")}</span><span className="text-slate-500">{stage[s] || "queued"}</span></div>)}</div>}
        <div className="card h-64 overflow-y-auto text-sm">
          <div className="mb-2 font-medium">Activity</div>
          {feed.length === 0 && <div className="text-slate-500">Waiting for the agent…</div>}
          {feed.filter(f => f.kind !== "guidance").map(f => <div key={f.id} className={`border-b py-1.5 last:border-0 ${f.kind === "guidance" ? "text-amber-600" : f.kind === "agent" ? "text-sky-600" : f.kind === "user" ? "font-medium" : f.kind === "correction" ? "text-emerald-600" : ""}`}>
            <span className="mr-1 text-xs uppercase text-slate-400">{f.kind}</span>{f.text}</div>)}
        </div>
      </div>
    </div>
  );
}
function Stat({ label, value, sub, small }: { label: string; value: any; sub?: string; small?: boolean }) {
  return <div className="card !p-3"><div className="text-xs text-slate-500">{label}</div><div className={small ? "text-sm font-medium" : "text-2xl font-semibold"}>{value}</div>{sub && <div className="text-xs text-slate-500">{sub}</div>}</div>;
}
