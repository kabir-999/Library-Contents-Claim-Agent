"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { WS } from "@/lib/api";
import { fromEvent, Orient } from "@/lib/orientation";
import { pcmToB64, startVad } from "@/lib/vad";
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
  const [stage, setStage] = useState<Record<string, string>>({});
  const [ended, setEnded] = useState(false);
  const [text, setText] = useState("");
  const [camH, setCamH] = useState(start.cameraHeightCm ? String(start.cameraHeightCm) : "");
  const [elapsed, setElapsed] = useState(0);
  const [mute, setMute] = useState(false);
  const n = useRef(0);

  const push = useCallback((kind: FeedEntry["kind"], text: string) => setFeed(f => [{ id: n.current++, kind, text, t: Date.now() }, ...f].slice(0, 60)), []);
  const send = (o: object) => ws.current?.readyState === WebSocket.OPEN && ws.current.send(JSON.stringify(o));

  const stopSpeech = useCallback(() => {
    window.speechSynthesis?.cancel(); audioEl.current?.pause(); agentSpeaking.current = false; setAgent(a => (a === "speaking" ? "listening" : a));
  }, []);

  const speak = useCallback((m: any) => {
    if (muted.current) return;
    stopSpeech(); agentSpeaking.current = true; setAgent("speaking");
    const done = () => { agentSpeaking.current = false; setAgent(a => (a === "speaking" ? "listening" : a)); };
    if (m.audio_wav_b64) { const a = new Audio("data:audio/wav;base64," + m.audio_wav_b64); audioEl.current = a; a.onended = done; a.onerror = done; a.play().catch(done); }
    else if (window.speechSynthesis) { const u = new SpeechSynthesisUtterance(m.text); u.onend = done; u.onerror = done; window.speechSynthesis.speak(u); }
    else done();
  }, [stopSpeech]);

  useEffect(() => {
    const v = video.current!; v.srcObject = start.stream; v.play().catch(() => {});
    const sock = new WebSocket(`${WS}/ws/sweeps/${start.sweepId}`); ws.current = sock;
    sock.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "state") setState(m);
      else if (m.type === "event") push("event", m.text);
      else if (m.type === "guidance") { setGuidance(m.text); push("guidance", m.text); }
      else if (m.type === "agent_say") { if (m.kind !== "guidance") push("agent", m.text); speak(m); }
      else if (m.type === "transcript") push("user", m.text || "(not understood)");
      else if (m.type === "correction") push("correction", `${m.action}${m.value ? " → " + m.value : ""} (${m.target})`);
      else if (m.type === "stop_speaking") stopSpeech();
      else if (m.type === "processing") { setStage(s => ({ ...s, [m.stage]: m.status })); }
      else if (m.type === "sweep_ended") { setEnded(true); setAgent("processing"); }
      else if (m.type === "packet_ready") { push("processing", `Packet ready in ${Number(m.seconds).toFixed(0)} s`); setTimeout(onEnded, 1500); }
    };
    sock.onclose = () => push("event", "Connection closed");

    const onOri = (e: DeviceOrientationEvent) => { orient.current = fromEvent(e); };
    window.addEventListener("deviceorientation", onOri);

    const timer = setInterval(() => {
      if (sock.readyState !== WebSocket.OPEN || v.videoWidth === 0) return;
      const c = canvas.current!; const w = Math.min(1280, v.videoWidth), h = Math.round(w * v.videoHeight / v.videoWidth);
      c.width = w; c.height = h; c.getContext("2d")!.drawImage(v, 0, 0, w, h);
      c.toBlob(b => { if (!b) return; const r = new FileReader(); r.onload = () => {
        const data = (r.result as string).split(",")[1];
        send({ type: "frame", ts: Date.now() / 1000, orientation: orient.current, data });
      }; r.readAsDataURL(b); }, "image/jpeg", 0.82);
    }, FPS_INTERVAL_MS);
    const clock = setInterval(() => setElapsed(e => e + 1), 1000);

    let stopVad = () => {};
    startVad(start.stream, {
      isAgentSpeaking: () => agentSpeaking.current,
      onStart: () => { if (agentSpeaking.current) { stopSpeech(); send({ type: "barge_in" }); } send({ type: "speech_start" }); setAgent("you"); },
      onEnd: () => { send({ type: "speech_end" }); setAgent("processing"); setTimeout(() => setAgent(a => (a === "processing" ? "listening" : a)), 3000); },
      onUtterance: (pcm) => send({ type: "utterance", sample_rate: 16000, pcm16_b64: pcmToB64(pcm) }),
    }).then(f => { stopVad = f; }).catch(e => push("event", "Microphone VAD unavailable: " + e));

    return () => { clearInterval(timer); clearInterval(clock); window.removeEventListener("deviceorientation", onOri); stopVad(); sock.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { muted.current = mute; if (mute) stopSpeech(); }, [mute, stopSpeech]);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0"), ss = String(elapsed % 60).padStart(2, "0");
  const room = state?.room;
  const roomStatus = !room ? "…" : room.camera_height_cm == null ? "need camera height" : room.floor_corners < 3 ? `${room.floor_corners}/3+ floor corners` : room.ceiling_marks ? "floor + ceiling marked" : "floor marked; need ceiling corner";
  const color: Record<string, string> = { listening: "bg-emerald-500", speaking: "bg-sky-500", you: "bg-amber-500", processing: "bg-slate-400" };

  return (
    <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
      <div className="space-y-3">
        <div className="relative overflow-hidden rounded-xl bg-black">
          <video ref={video} muted playsInline className="aspect-video w-full object-cover" />
          <canvas ref={canvas} className="hidden" />
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-sm text-white">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-red-500" /> REC {mm}:{ss}</div>
          <div className="absolute right-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-sm text-white">
            <span className={`h-2.5 w-2.5 rounded-full ${color[agent]}`} /> {agent === "you" ? "hearing you…" : `agent ${agent}`}</div>
          {guidance && <div className="absolute inset-x-3 bottom-3 rounded-lg bg-amber-500/95 p-3 text-sm font-medium text-black">{guidance}</div>}
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Books" value={state?.books ?? 0} sub={`${state?.identified ?? 0} identified`} />
          <Stat label="Other items" value={state?.items ?? 0} />
          <Stat label="Shelves" value={state?.shelves ?? 0} sub={`${state?.calibrated_segments ?? 0} walls scaled`} />
          <Stat label="Room" value={roomStatus} small />
        </div>
        <div className="card flex flex-wrap items-center gap-2 text-sm">
          <button className="btn-ghost" onClick={() => send({ type: "mark_corner" })} disabled={!start.orientationOk}>Mark floor corner</button>
          <button className="btn-ghost" onClick={() => send({ type: "mark_corner", ceiling: true })} disabled={!start.orientationOk}>Mark ceiling corner</button>
          <span className="flex items-center gap-1">Camera height cm
            <input className="w-20 rounded border p-1 dark:bg-slate-800" value={camH} onChange={e => setCamH(e.target.value)} onBlur={() => Number(camH) > 30 && send({ type: "set_camera_height", cm: Number(camH) })} /></span>
          <label className="ml-auto flex items-center gap-1"><input type="checkbox" checked={mute} onChange={e => setMute(e.target.checked)} /> mute agent voice</label>
          {!start.orientationOk && <span className="w-full text-xs text-amber-600">No orientation sensor on this device: room dimensions will be reported unavailable (use a phone for room measurement).</span>}
        </div>
        <form className="flex gap-2" onSubmit={e => { e.preventDefault(); if (text.trim()) { send({ type: "text", text }); push("user", text); setText(""); } }}>
          <input className="flex-1 rounded border p-2 text-sm dark:bg-slate-800" placeholder="Or type a correction (e.g. “those books belong to someone else”)" value={text} onChange={e => setText(e.target.value)} />
          <button className="btn-ghost">Send</button>
          <button type="button" className="btn bg-red-600 hover:bg-red-700 dark:bg-red-600 dark:text-white" disabled={ended} onClick={() => send({ type: "stop" })}>Stop sweep</button>
        </form>
      </div>
      <div className="space-y-3">
        {ended && <div className="card text-sm"><div className="mb-2 font-medium">Building claim packet…</div>
          {["book_identification", "book_pricing", "nonbook_classification", "nonbook_pricing", "packet_generation"].map(s =>
            <div key={s} className="flex justify-between"><span>{s.replace(/_/g, " ")}</span><span className="text-slate-500">{stage[s] || "queued"}</span></div>)}</div>}
        <div className="card h-[28rem] overflow-y-auto text-sm">
          <div className="mb-2 font-medium">Activity</div>
          {feed.length === 0 && <div className="text-slate-500">Waiting for the agent…</div>}
          {feed.map(f => <div key={f.id} className={`border-b py-1.5 last:border-0 ${f.kind === "guidance" ? "text-amber-600" : f.kind === "agent" ? "text-sky-600" : f.kind === "user" ? "font-medium" : f.kind === "correction" ? "text-emerald-600" : ""}`}>
            <span className="mr-1 text-xs uppercase text-slate-400">{f.kind}</span>{f.text}</div>)}
        </div>
      </div>
    </div>
  );
}
function Stat({ label, value, sub, small }: { label: string; value: any; sub?: string; small?: boolean }) {
  return <div className="card !p-3"><div className="text-xs text-slate-500">{label}</div><div className={small ? "text-sm font-medium" : "text-2xl font-semibold"}>{value}</div>{sub && <div className="text-xs text-slate-500">{sub}</div>}</div>;
}
