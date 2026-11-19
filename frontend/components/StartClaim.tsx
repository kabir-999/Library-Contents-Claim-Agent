"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export interface StartResult { sweepId: string; country: string; currency: string; stream: MediaStream; orientationOk: boolean; cameraHeightCm: number | null }

export default function StartClaim({ onStart }: { onStart: (r: StartResult) => void }) {
  const [markets, setMarkets] = useState<{ code: string; name: string; currency: string }[]>([]);
  const [country, setCountry] = useState("US");
  const [currency, setCurrency] = useState("USD");
  const [camH, setCamH] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => { api.markets().then(setMarkets).catch(() => setErr("Backend not reachable. Start it first (see README).")); api.health().then(setHealth).catch(() => {}); }, []);
  useEffect(() => { const m = markets.find(m => m.code === country); if (m) setCurrency(m.currency); }, [country, markets]);

  async function start() {
    setErr(""); setBusy(true);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera/mic need HTTPS or localhost.");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
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
