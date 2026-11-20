"use client";
import { useEffect, useState } from "react";
import { api, money } from "@/lib/api";
import type { Packet } from "@/lib/types";

const na = <span className="text-slate-400">not available</span>;
const area = (m2: number | null, ft2: number | null) => (m2 == null ? na : `${m2} m² (${ft2} ft²)`);

export default function Results({ p, onNew }: { p: Packet; onNew: () => void }) {
  const t = p.totals, r = p.room, cur = p.sweep.currency, sid = p.sweep.id;
  const open = p.review_queue.filter(q => !q.resolved);
  const [c2, setC2] = useState("GB");
  const [cmp, setCmp] = useState<any>(p.second_country_pricing);
  const [busy, setBusy] = useState(false);
  const [zoom, setZoom] = useState<{ crop: string; frame: string; label: string; full: boolean } | null>(null);
  useEffect(() => { const f = (e: KeyboardEvent) => e.key === "Escape" && setZoom(null); window.addEventListener("keydown", f); return () => window.removeEventListener("keydown", f); }, []);
  const runCmp = async () => { setBusy(true); try { setCmp(await api.secondCountry(sid, c2, 10)); } finally { setBusy(false); } };
  return (
    <div className="mx-auto max-w-4xl space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-auto text-xl font-semibold">Your claim</h1>
        <a className="btn" href={api.pdfUrl(sid)} target="_blank" rel="noreferrer">Open report (PDF)</a>
        <a className="btn-ghost" href={`${api.jsonUrl(sid).replace("/packet/download", "/packet")}`} target="_blank" rel="noreferrer">View claim packet</a>
        <a className="btn-ghost" href={api.jsonUrl(sid)}>Download JSON</a>
        <button className="btn-ghost" onClick={onNew}>New claim</button>
      </div>

      <div className="card grid gap-x-8 gap-y-2 text-sm sm:grid-cols-2">
        <Row k="Books" v={`${t.book_count} (${t.books_identified} identified)`} />
        <Row k="Books – replacement cost" v={money(t.books_replacement_cost, cur)} />
        <Row k="Books – used value" v={money(t.books_used_value, cur)} />
        <Row k="Other items – replacement" v={t.items_priced ? `${money(t.items_replacement_cost_low, cur)} – ${money(t.items_replacement_cost_high, cur)}` : na} />
        <Row k="Shelf length" v={t.shelf_run_m != null ? `${t.shelf_run_m} m` : na} />
        <Row k="Floor area" v={area(r.floor_area_m2, r.floor_area_ft2)} />
        <Row k="Wall area" v={area(r.wall_area_m2, r.wall_area_ft2)} />
        <Row k="Room (L × W × H)" v={r.length_m != null ? `${r.length_m} × ${r.width_m} × ${r.height_m ?? "?"} m` : na} />
      </div>

      <div className="card overflow-x-auto">
        <h2 className="mb-2 font-semibold">Books ({p.books.length}) <span className="text-sm font-normal text-slate-500">· {p.books.filter(b => !b.title).length} unidentified</span></h2>
        <table className="w-full text-sm"><thead><tr><th></th><th>Title</th><th>Replacement</th><th>Used</th></tr></thead>
          <tbody>{p.books.map(b => (
            <tr key={b.id} className="border-t border-slate-100 dark:border-slate-800">
              <td><img src={api.cropUrl(sid, b.id)} alt="" title="Click to enlarge" onClick={() => setZoom({ crop: api.cropUrl(sid, b.id), frame: api.frameUrl(b.frame_ref), label: b.title || "Unidentified book", full: false })} className="h-14 w-10 cursor-zoom-in rounded object-cover transition hover:ring-2 hover:ring-sky-400" onError={e => ((e.target as HTMLImageElement).style.display = "none")} /></td>
              <td><div className="font-medium">{b.title || <span className="italic text-slate-400">Could not identify</span>}</div>
                {b.author && <div className="text-xs text-slate-500">{b.author}</div>}
                {b.status === "needs_appraisal" && <div className="text-xs text-violet-600">needs appraisal</div>}</td>
              <td><Price a={b.replacement_cost.amount} url={b.replacement_cost.url} cur={cur} /></td>
              <td><Price a={b.used_value.amount} url={b.used_value.url} cur={cur} /></td>
            </tr>))}</tbody></table>
      </div>

      {p.items.length > 0 && <div className="card overflow-x-auto">
        <h2 className="mb-2 font-semibold">Other items ({p.items.length})</h2>
        <table className="w-full text-sm"><thead><tr><th></th><th>Item</th><th>Replacement range</th></tr></thead>
          <tbody>{p.items.map(i => (
            <tr key={i.id} className="border-t border-slate-100 dark:border-slate-800">
              <td><img src={api.cropUrl(sid, i.id)} alt="" title="Click to enlarge" onClick={() => setZoom({ crop: api.cropUrl(sid, i.id), frame: api.frameUrl(i.frame_ref), label: i.category, full: false })} className="h-12 w-12 cursor-zoom-in rounded object-cover transition hover:ring-2 hover:ring-sky-400" onError={e => ((e.target as HTMLImageElement).style.display = "none")} /></td>
              <td className="font-medium">{i.category}</td>
              <td>{i.replacement_cost.low != null ? <>{money(i.replacement_cost.low, cur)} – {money(i.replacement_cost.high, cur)}
                {i.replacement_cost.url && <a className="ml-2 text-xs text-sky-600 underline" href={i.replacement_cost.url} target="_blank" rel="noreferrer">source</a>}</>
                : <span className="text-violet-600">{i.status === "needs_appraisal" ? "needs appraisal" : "not priced"}</span>}</td>
            </tr>))}</tbody></table>
      </div>}

      <div className="card text-sm">
        <h2 className="mb-2 font-semibold">Same books, another country <span className="text-sm font-normal text-slate-500">· proves location is a setting</span></h2>
        <div className="flex flex-wrap items-center gap-2">
          <input className="w-20 rounded border p-2 uppercase dark:bg-slate-800" maxLength={2} value={c2} onChange={e => setC2(e.target.value.toUpperCase())} />
          <button className="btn-ghost" disabled={busy} onClick={runCmp}>{busy ? "Pricing…" : "Price the first 10 identified books there"}</button></div>
        {cmp && cmp.rows && <table className="mt-2 w-full"><thead><tr><th>Book</th><th>New ({cmp.currency})</th><th>Used</th></tr></thead>
          <tbody>{cmp.rows.map((r: any) => <tr key={r.book_id} className="border-t border-slate-100 dark:border-slate-800">
            <td>{r.title}</td><td>{r.replacement_cost != null ? <>{money(r.replacement_cost, cmp.currency)}{r.url && <a className="ml-1 text-xs text-sky-600 underline" href={r.url} target="_blank" rel="noreferrer">source</a>}</> : "—"}</td>
            <td>{r.used_value != null ? money(r.used_value, cmp.currency) : "—"}</td></tr>)}</tbody></table>}
      </div>

      {open.length > 0 && <details open className="card text-sm">
        <summary className="cursor-pointer font-semibold">Needs your review ({open.length})</summary>
        <ul className="mt-2 space-y-1">{open.map((q, i) => <li key={i}><span className="font-mono text-xs text-slate-500">{q.ref_id}</span> {q.reason}</li>)}</ul>
      </details>}

      {zoom && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={() => setZoom(null)}>
        <div className="max-h-full max-w-4xl rounded-xl bg-white p-3 shadow-2xl dark:bg-slate-900" onClick={e => e.stopPropagation()}>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <b className="mr-auto">{zoom.label}</b>
            <button className="btn-ghost" onClick={() => setZoom({ ...zoom, full: !zoom.full })}>{zoom.full ? "Show the book only" : "Show the full frame"}</button>
            <button className="btn-ghost" onClick={() => setZoom(null)}>Close ✕</button>
          </div>
          <img src={zoom.full ? zoom.frame : zoom.crop} alt={zoom.label} className="mx-auto max-h-[78vh] max-w-full rounded object-contain" />
        </div>
      </div>}
    </div>
  );
}
const Row = ({ k, v }: { k: string; v: any }) => <div className="flex justify-between gap-4 border-b border-slate-100 py-1 dark:border-slate-800"><span className="text-slate-500">{k}</span><span className="font-medium">{v}</span></div>;
function Price({ a, url, cur }: { a: number | null; url: string; cur: string }) {
  if (a == null) return <span className="text-slate-400">—</span>;
  return <span>{money(a, cur)}{url && <a className="ml-1 text-xs text-sky-600 underline" href={url} target="_blank" rel="noreferrer">source</a>}</span>;
}
