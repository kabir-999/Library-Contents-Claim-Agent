"use client";
import { useRef, useState } from "react";
import { api, money } from "@/lib/api";
import type { Packet } from "@/lib/types";

export default function PacketView({ p, refresh, frame, setFrame }: { p: Packet; refresh: () => void; frame: string | null; setFrame: (f: string | null) => void }) {
  const [country, setCountry] = useState("GB");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const sid = p.sweep.id;
  const sc = p.second_country_pricing;
  const frames = Array.from(new Set([...p.books.map(b => b.frame_ref), ...p.items.map(i => i.frame_ref), ...p.calibrations.map(c => c.frame_ref)])).filter(Boolean);
  const urls = [...p.books.flatMap(b => [["replacement", b.title, b.replacement_cost], ["used", b.title, b.used_value]] as const).filter(([, , x]) => x.url),
    ...p.items.flatMap(i => (i.replacement_cost.sources || []).slice(0, 3).map(s => ["item", i.description, s] as const))];
  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-center gap-3">
        <a className="btn" href={api.jsonUrl(sid)}>Download claim_packet.json</a>
        <a className="btn" href={api.pdfUrl(sid)} target="_blank" rel="noreferrer">Download PDF report</a>
        <span className="text-xs text-slate-500">Sweep {sid} · {p.sweep.duration_s?.toFixed(0)}s · generated {p.processing.find(x => x.stage === "packet_generation_total")?.seconds}s after the sweep</span>
      </div>

      <div className="card space-y-2 text-sm">
        <h3 className="font-semibold">Second-country pricing comparison</h3>
        <div className="flex items-center gap-2">
          <input className="w-20 rounded border p-1 uppercase dark:bg-slate-800" value={country} maxLength={2} onChange={e => setCountry(e.target.value.toUpperCase())} />
          <button className="btn-ghost" disabled={busy} onClick={async () => { setBusy(true); setErr(""); try { await api.secondCountry(sid, country, 10); await refresh(); } catch (e: any) { setErr(e.message); } setBusy(false); }}>{busy ? "Pricing…" : "Price the first 10 identified books in this country"}</button>
        </div>
        {err && <div className="text-red-600">{err}</div>}
        {sc && <table className="w-full"><thead><tr><th>Book</th><th>{p.sweep.currency} (home)</th><th>{sc.currency} replacement</th><th>{sc.currency} used</th><th>Source</th><th>Note</th></tr></thead>
          <tbody>{sc.rows.map(r => { const home = p.books.find(b => b.id === r.book_id); return (
            <tr key={r.book_id} className="border-t border-slate-100 dark:border-slate-800"><td>{r.title}</td><td>{money(home?.replacement_cost.amount ?? null, p.sweep.currency)}</td><td>{money(r.replacement_cost, sc.currency)}</td><td>{money(r.used_value, sc.currency)}</td>
              <td>{r.url ? <a className="text-sky-600 underline" href={r.url} target="_blank" rel="noreferrer">{r.source.slice(0, 24) || "link"}</a> : "—"}</td><td className="text-xs text-slate-500">{r.note}</td></tr>); })}</tbody></table>}
      </div>

      <div className="card text-sm"><h3 className="mb-2 font-semibold">Evidence frames ({frames.length})</h3>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-6 lg:grid-cols-8">{frames.map(f => <img key={f} src={api.frameUrl(f)} alt={f} title={f} className="aspect-video cursor-pointer rounded object-cover" onClick={() => setFrame(f)} />)}</div></div>

      <div className="card text-sm"><h3 className="mb-2 font-semibold">Price sources ({urls.length})</h3>
        <div className="max-h-64 overflow-y-auto"><table className="w-full"><thead><tr><th>Type</th><th>Item</th><th>Source</th><th>Retrieved</th></tr></thead><tbody>
          {urls.map(([k, n, x], i) => <tr key={i} className="border-t border-slate-100 dark:border-slate-800"><td>{k}</td><td>{String(n).slice(0, 40)}</td>
            <td><a className="text-sky-600 underline" href={(x as any).url} target="_blank" rel="noreferrer">{(x as any).source?.slice(0, 40) || "link"}</a></td><td>{((x as any).retrieved_at || "").slice(0, 10)}</td></tr>)}</tbody></table></div></div>

      <div className="card text-sm"><h3 className="mb-2 font-semibold">Calibrations</h3>
        {p.calibrations.length === 0 ? <div className="text-amber-600">No scale was established: all centimetre values are blank and queued for review. Open an evidence frame, click two points of a known distance (e.g. a shelf board), and re-run.</div> :
          <ul className="list-disc pl-5">{p.calibrations.map(c => <li key={c.id}>{c.id} · {c.scope} · {c.method} · conf {c.confidence} · {c.note}</li>)}</ul>}</div>

      {frame && <FrameModal sid={sid} frame={frame} p={p} onClose={() => setFrame(null)} refresh={refresh} />}
    </div>
  );
}

export function FrameModal({ sid, frame, p, onClose, refresh }: { sid: string; frame: string; p: Packet; onClose: () => void; refresh: () => void }) {
  const img = useRef<HTMLImageElement>(null);
  const [pts, setPts] = useState<[number, number][]>([]);
  const [cm, setCm] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const boxes = [...p.books.filter(b => b.frame_ref === frame).map(b => ({ id: b.id, bbox: (b as any).bbox as number[] | null })), ...p.items.filter(i => i.frame_ref === frame).map(i => ({ id: i.id, bbox: (i as any).bbox as number[] | null }))];
  function click(e: React.MouseEvent) {
    const el = img.current!, r = el.getBoundingClientRect(); const sx = el.naturalWidth / r.width, sy = el.naturalHeight / r.height;
    setPts(v => (v.length >= 2 ? [[(e.clientX - r.left) * sx, (e.clientY - r.top) * sy]] : [...v, [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy]]));
  }
  async function calibrate() {
    setBusy(true); setMsg("");
    try { await api.twoPoint(sid, { frame_id: frame, p1: pts[0], p2: pts[1], real_cm: Number(cm) }); setMsg("Calibrated; packet regenerated."); await refresh(); } catch (e: any) { setMsg(e.message); }
    setBusy(false);
  }
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={onClose}>
      <div className="max-h-full max-w-5xl overflow-auto rounded-xl bg-white p-3 dark:bg-slate-900" onClick={e => e.stopPropagation()}>
        <div className="mb-2 flex items-center justify-between text-sm"><span className="font-mono">{frame}</span><button className="btn-ghost" onClick={onClose}>Close</button></div>
        <div className="relative inline-block">
          <img ref={img} src={api.frameUrl(frame)} alt={frame} className="max-h-[70vh] cursor-crosshair" onClick={click} />
          {img.current && boxes.filter(b => b.bbox).map(b => { const el = img.current!; const k = el.clientWidth / (el.naturalWidth || 1); const [x1, y1, x2, y2] = b.bbox!;
            return <div key={b.id} className="pointer-events-none absolute border-2 border-emerald-400" style={{ left: x1 * k, top: y1 * k, width: (x2 - x1) * k, height: (y2 - y1) * k }} title={b.id} />; })}
          {img.current && pts.map((q, i) => { const el = img.current!; const k = el.clientWidth / (el.naturalWidth || 1); return <div key={i} className="pointer-events-none absolute h-3 w-3 -translate-x-1.5 -translate-y-1.5 rounded-full bg-red-500" style={{ left: q[0] * k, top: q[1] * k }} />; })}
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-sm">
          <span>Manual scale: click two points a known distance apart</span>
          <input className="w-24 rounded border p-1 dark:bg-slate-800" placeholder="cm" value={cm} onChange={e => setCm(e.target.value)} />
          <button className="btn-ghost" disabled={pts.length < 2 || !Number(cm) || busy} onClick={calibrate}>{busy ? "Re-running…" : "Calibrate & regenerate"}</button>
          {msg && <span className="text-slate-500">{msg}</span>}
        </div>
      </div>
    </div>
  );
}
