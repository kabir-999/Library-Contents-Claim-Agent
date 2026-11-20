"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { unlockSpeech } from "@/lib/voice";

export interface StartResult { sweepId: string; country: string; currency: string; stream: MediaStream; orientationOk: boolean; cameraHeightCm: number | null }

export default function StartClaim({ onStart }: { onStart: (r: StartResult) => void }) {
  const [markets, setMarkets] = useState<{ code: string; name: string; currency: string }[]>([]);
  const [country, setCountry] = useState("US");
  const [currency, setCurrency] = useState("USD");
  const [camH, setCamH] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<any>(null);
  const [cams, setCams] = useState<{ id: string; label: string }[]>([]);
  const [camId, setCamId] = useState("");

  const PHONE_CAM = /iphone|continuity|iriun|droidcam|epoccam|camo|phone|android|obs/i;
  async function loadCams(askPermission = false) {
    try {
      if (askPermission) { const t = await navigator.mediaDevices.getUserMedia({ video: true }); t.getTracks().forEach(x => x.stop()); }
      const list = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput")
        .map((d, i) => ({ id: d.deviceId, label: d.label || `Camera ${i + 1}` }));
      setCams(list);
      const saved = localStorage.getItem("lca_cam");
      const pick = list.find(c => c.id === saved) || list.find(c => PHONE_CAM.test(c.label));   // prefer a phone camera if one is present
      if (pick) setCamId(pick.id);
    } catch (e: any) { setErr("Could not list cameras: " + (e.message || e)); }
  }
  useEffect(() => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices) return;
    loadCams(false);
    const md = navigator.mediaDevices, onChange = () => loadCams(false);   // a phone camera appearing/disappearing refreshes the list
    md.addEventListener("devicechange", onChange);
    const poll = setInterval(onChange, 4000);                               // some browsers miss the event for Continuity Camera
    return () => { md.removeEventListener("devicechange", onChange); clearInterval(poll); };
  }, []);

  useEffect(() => { api.markets().then(setMarkets).catch(() => setErr("Backend not reachable. Start it first (see README).")); api.health().then(setHealth).catch(() => {}); }, []);
  useEffect(() => { const m = markets.find(m => m.code === country); if (m) setCurrency(m.currency); }, [country, markets]);

  async function start() {
    setErr(""); setBusy(true);
    unlockSpeech();   // must happen inside this tap, or phones will silently block the agent's voice
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera/mic need HTTPS or localhost.");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { ...(camId ? { deviceId: { exact: camId } } : { facingMode: "environment" }), width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      try { camId && localStorage.setItem("lca_cam", camId); } catch {}
      let orientationOk = false;
      const DO: any = (window as any).DeviceOrientationEvent;
      if (DO && typeof DO.requestPermission === "function") {   // iOS requires a user-gesture permission
        try { orientationOk = (await DO.requestPermission()) === "granted"; } catch { orientationOk = false; }
      } else orientationOk = !!DO;
      const h = camH ? Number(camH) : null;
      const r = await api.start({ country, currency, device: navigator.userAgent.slice(0, 180), camera_height_cm: h });
      onStart({ sweepId: r.sweep_id, country: r.country, currency: r.currency, stream, orientationOk, cameraHeightCm: h });
    } catch (e: any) { setErr(e.message || String(e)); } finally { setBusy(false); }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <h1 className="text-2xl font-semibold">Library Contents Claim Agent</h1>
      <div className="card space-y-3">
        <h2 className="font-medium">1. Where is the policyholder?</h2>
        <div className="grid grid-cols-2 gap-3">
          <label className="text-sm">Country
            <select className="mt-1 w-full rounded border p-2 dark:bg-slate-800" value={country} onChange={e => setCountry(e.target.value)}>
              {markets.map(m => <option key={m.code} value={m.code}>{m.name}</option>)}
            </select></label>
          <label className="text-sm">Currency
            <input className="mt-1 w-full rounded border p-2 uppercase dark:bg-slate-800" maxLength={3} value={currency} onChange={e => setCurrency(e.target.value.toUpperCase())} /></label>
        </div>
        <label className="block text-sm">Height of the phone/camera above the floor, cm <span className="text-slate-500">(optional now; needed for room measurements; you can say it later)</span>
          <input className="mt-1 w-full rounded border p-2 dark:bg-slate-800" inputMode="numeric" placeholder="e.g. 150" value={camH} onChange={e => setCamH(e.target.value)} /></label>
      </div>
      <div className="card space-y-2 text-sm">
        <h2 className="font-medium">Camera <span className="font-normal text-slate-500">(use your phone as the webcam; everything still runs on this laptop)</span></h2>
        <div className="flex flex-wrap items-center gap-2">
          <select className="min-w-0 flex-1 rounded border p-2 dark:bg-slate-800" value={camId} onChange={e => setCamId(e.target.value)}>
            <option value="">Default camera</option>
            {cams.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
          </select>
          <button type="button" className="btn-ghost" onClick={() => loadCams(true)}>Detect cameras</button>
        </div>
        {cams.length <= 1 && <div className="rounded border border-amber-300 bg-amber-50 p-2 text-xs text-amber-800">
          This browser sees only {cams.length} camera. If your iPhone works in FaceTime but is missing here: <b>fully quit this browser (Cmd+Q) and reopen it with the iPhone connected</b>,
          or open <b>http://localhost:3000</b> in <b>Safari</b>, which lists Continuity Camera reliably. This list refreshes by itself when a camera appears.</div>}
        <div className="text-xs text-slate-500">iPhone + Mac: Continuity Camera shows up as “iPhone Camera” (same Apple ID, Wi-Fi and Bluetooth on, phone near the Mac).
          Android or any phone: install <b>Iriun Webcam</b> or <b>DroidCam</b> on the phone and the Mac, connect by USB or Wi-Fi, then choose it here.</div>
      </div>
      <div className="card space-y-2 text-sm">
        <h2 className="font-medium">2. How the sweep works</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>Walk the room <b>once</b> in under three minutes, slowly, pointing the camera at shelves, walls and floor.</li>
          <li>Don’t remove books, scan barcodes, or photograph individual books. I’ll tell you if I need you to slow down, move closer or revisit a shelf.</li>
          <li>Speak corrections any time: “that’s a first edition”, “those books belong to someone else”, “that’s a print, not an original”, “skip this shelf”.</li>
          <li>For the room: point at the floor where each wall meets the floor and say “mark corner”; point at the ceiling above and say “ceiling corner”.</li>
          <li>For metric scale, place an <b>A4/Letter sheet</b> flat on a shelf edge within view, or say “this bookcase is 200 centimetres tall”.</li>
          <li>Use headphones if you want to interrupt the agent while it’s speaking.</li>
        </ul>
      </div>
      {health && <div className="text-xs text-slate-500">
        Backend: groq {health.llm.groq_configured ? "✓" : "✗"} · local LLM {health.llm.local_model} · eBay {health.pricing_sources.ebay ? "✓" : "✗"} · SerpAPI {health.pricing_sources.serpapi ? "✓" : "✗"} · TTS {health.tts}
        {Object.keys(health.models.errors).length > 0 && <div className="text-amber-600">Unavailable models: {Object.keys(health.models.errors).join(", ")}</div>}
      </div>}
      {err && <div className="rounded border border-red-300 bg-red-50 p-2 text-sm text-red-700">{err}</div>}
      <button className="btn w-full" disabled={busy} onClick={start}>{busy ? "Requesting permissions…" : "Allow camera & microphone, then start sweep"}</button>
    </div>
  );
}
