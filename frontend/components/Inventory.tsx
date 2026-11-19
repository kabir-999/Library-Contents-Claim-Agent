"use client";
import { useMemo, useState } from "react";
import { api, money } from "@/lib/api";
import type { Book, Packet } from "@/lib/types";

const tone: Record<string, string> = { identified: "bg-emerald-100 text-emerald-800", needs_review: "bg-amber-100 text-amber-800", partial: "bg-amber-100 text-amber-800",
  unidentified: "bg-rose-100 text-rose-800", needs_appraisal: "bg-violet-100 text-violet-800", excluded: "bg-slate-200 text-slate-600", priced: "bg-emerald-100 text-emerald-800", unpriced: "bg-slate-200 text-slate-700" };

export default function Inventory({ p, onFrame }: { p: Packet; onFrame: (f: string) => void }) {
  const [onlyReview, setOnlyReview] = useState(false);
  const cur = p.sweep.currency, sid = p.sweep.id;
  const reviewIds = useMemo(() => new Set(p.review_queue.filter(r => !r.resolved).map(r => r.ref_id)), [p]);
  const shelves = useMemo(() => {
    const m: Record<string, Book[]> = {};
    for (const b of p.books) (m[b.shelf || "unassigned"] ||= []).push(b);
    return Object.entries(m).sort();
  }, [p]);
  return (
    <div className="space-y-4">
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={onlyReview} onChange={e => setOnlyReview(e.target.checked)} /> Show only entries needing review</label>
      {shelves.map(([shelf, books]) => {
        const st = p.totals.per_shelf[shelf];
        const rows = books.filter(b => !onlyReview || reviewIds.has(b.id));
        if (!rows.length) return null;
        return (
          <div key={shelf} className="card overflow-x-auto">
            <div className="mb-2 flex flex-wrap items-baseline gap-3"><h3 className="font-semibold">Shelf {shelf}</h3>
              <span className="text-sm text-slate-500">{books.length} books · run {st?.run_m != null ? `${st.run_m} m${st.run_complete ? "" : " (lower bound)"}` : "unavailable (no scale)"}</span></div>
            <table className="w-full text-sm"><thead><tr><th>#</th><th></th><th>Title / author / edition</th><th>Status</th><th>ID conf</th><th>H × T (cm)</th><th>Replacement</th><th>Used</th></tr></thead>
              <tbody>{rows.map(b => (
                <tr key={b.id} className="border-t border-slate-100 dark:border-slate-800">
                  <td>{b.position}</td>
                  <td><img src={api.cropUrl(sid, b.id)} alt="" className="h-14 w-8 cursor-pointer rounded object-cover" onClick={() => onFrame(b.frame_ref)} onError={e => ((e.target as HTMLImageElement).style.display = "none")} /></td>
                  <td className="max-w-xs"><div className="font-medium">{b.title || <span className="italic text-slate-400">unidentified</span>}</div>
                    <div className="text-xs text-slate-500">{b.author}{b.edition && <> · <i>{b.edition}</i></>}{b.isbn && <> · ISBN {b.isbn}</>}</div>
                    {!b.title && b.suggested_title && <div className="text-xs text-amber-600">Unconfirmed suggestion: {b.suggested_title}{b.suggested_author && ` — ${b.suggested_author}`}</div>}
                    {!b.title && b.ocr_text.length > 0 && <div className="text-xs text-slate-400">OCR: {b.ocr_text.join(" | ")}</div>}
                    {b.pricing_note && <div className="text-xs text-slate-400">{b.pricing_note}</div>}
                    {b.owner_note && <div className="text-xs text-slate-400">{b.owner_note}</div>}</td>
                  <td><span className={`badge ${tone[b.detail_status || b.status] || ""}`}>{(b.detail_status || b.status).replace("_", " ")}</span>{reviewIds.has(b.id) && <span className="badge ml-1 bg-amber-100 text-amber-800">review</span>}</td>
                  <td>{b.id_confidence ? b.id_confidence.toFixed(2) : "—"}</td>
                  <td title={b.height_measurement.method ? `${b.height_measurement.method}, conf ${b.height_measurement.confidence}` : "no scale established"}>
                    {b.spine_height_cm != null ? `${b.spine_height_cm} × ${b.spine_thickness_cm ?? "—"}` : "—"}</td>
                  <td><Price a={b.replacement_cost.amount} url={b.replacement_cost.url} src={b.replacement_cost.source} cur={cur} conv={b.replacement_cost.converted} /></td>
                  <td><Price a={b.used_value.amount} url={b.used_value.url} src={b.used_value.source} cur={cur} /></td>
                </tr>))}</tbody></table>
          </div>);
      })}
      <div className="card overflow-x-auto"><h3 className="mb-2 font-semibold">Non-book items ({p.items.length})</h3>
        <table className="w-full text-sm"><thead><tr><th></th><th>Category / description</th><th>W × H (cm)</th><th>Conf</th><th>Status</th><th>Replacement range</th><th>Sources</th></tr></thead>
          <tbody>{p.items.filter(i => !onlyReview || reviewIds.has(i.id)).map(i => (
            <tr key={i.id} className="border-t border-slate-100 dark:border-slate-800">
              <td><img src={api.cropUrl(sid, i.id)} alt="" className="h-12 w-12 cursor-pointer rounded object-cover" onClick={() => onFrame(i.frame_ref)} onError={e => ((e.target as HTMLImageElement).style.display = "none")} /></td>
              <td><div className="font-medium">{i.category}</div><div className="text-xs text-slate-500">{i.description}{i.brand_model && ` · ${i.brand_model}`}{i.material && ` · ${i.material}`}</div></td>
              <td title={i.dimension_method}>{i.dimensions_cm.w != null ? `${i.dimensions_cm.w} × ${i.dimensions_cm.h}` : "—"}</td>
              <td>{i.confidence.toFixed(2)}</td>
              <td><span className={`badge ${tone[i.detail_status || i.status] || ""}`}>{(i.detail_status || i.status).replace("_", " ")}</span>{reviewIds.has(i.id) && <span className="badge ml-1 bg-amber-100 text-amber-800">review</span>}</td>
              <td>{i.replacement_cost.low != null ? `${money(i.replacement_cost.low, cur)} – ${money(i.replacement_cost.high, cur)}` : "—"}</td>
              <td className="text-xs">{(i.replacement_cost.sources || []).slice(0, 5).map((s, k) => <a key={k} className="mr-1 text-sky-600 underline" href={s.url} target="_blank" rel="noreferrer">[{k + 1}]</a>)}
                {i.replacement_cost.retrieved_at && <div className="text-slate-400">{i.replacement_cost.retrieved_at.slice(0, 10)}</div>}</td>
            </tr>))}</tbody></table></div>
    </div>
  );
}
function Price({ a, url, src, cur, conv }: { a: number | null; url: string; src: string; cur: string; conv?: boolean }) {
  if (a == null) return <span className="text-slate-400">—</span>;
  return <div>{money(a, cur)}{conv && <span className="badge ml-1 bg-sky-100 text-sky-800">converted</span>}<div className="text-xs">{url ? <a className="text-sky-600 underline" href={url} target="_blank" rel="noreferrer">{src.slice(0, 28)}</a> : src}</div></div>;
}
