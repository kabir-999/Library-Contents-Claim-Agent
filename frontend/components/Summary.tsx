"use client";
import { useState } from "react";
import { api, money } from "@/lib/api";
import type { Packet } from "@/lib/types";

const f = (v: number | null, u = "") => (v == null ? <span className="text-amber-600">unavailable</span> : <>{v}{u}</>);

export default function Summary({ p, onFrame, refresh }: { p: Packet; onFrame: (f: string) => void; refresh: () => void }) {
  const t = p.totals, r = p.room, cur = p.sweep.currency;
  const [busy, setBusy] = useState("");
  const open = p.review_queue.filter(q => !q.resolved);
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card k="Books" v={t.book_count} s={`${t.books_identified} identified · ${t.books_unidentified} unidentified`} />
        <Card k="Replacement cost (books)" v={money(t.books_replacement_cost, cur)} s={`${t.books_priced_replacement} of ${t.book_count} priced`} />
        <Card k="Used value (books)" v={money(t.books_used_value, cur)} s={`${t.books_priced_used} of ${t.book_count} priced`} />
        <Card k="Non-book replacement range" v={`${money(t.items_replacement_cost_low, cur)} – ${money(t.items_replacement_cost_high, cur)}`} s={`${t.items_priced} of ${p.items.length} items priced`} small />
        <Card k="Shelf run" v={t.shelf_run_m != null ? `${t.shelf_run_m} m` : "unavailable"} s="calibrated spines only" />
        <Card k="Excluded from totals" v={t.excluded_from_totals} s="unpriced, appraisal or owner-excluded" />
        <Card k="Floor area" v={r.floor_area_m2 != null ? `${r.floor_area_m2} m² / ${r.floor_area_ft2} ft²` : "unavailable"} s={r.scale_method || "no method established"} small />
        <Card k="Wall area (gross)" v={r.wall_area_m2 != null ? `${r.wall_area_m2} m² / ${r.wall_area_ft2} ft²` : "unavailable"} s={r.shelved_wall_area_m2 != null ? `of which shelved: ${r.shelved_wall_area_m2} m²` : "shelved area unavailable"} small />
      </div>
      <div className="card text-sm">
        <h3 className="mb-2 font-semibold">Room</h3>
        <div className="grid grid-cols-3 gap-2"><div>Length: {f(r.length_m, " m")}</div><div>Width: {f(r.width_m, " m")}</div><div>Ceiling: {f(r.height_m, " m")}</div></div>
        <div className="mt-1 text-xs text-slate-500">Shape: {r.shape || "—"} · confidence: {r.confidence ?? "—"}</div>
        <ul className="mt-2 list-disc pl-5 text-xs text-slate-500">{r.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
        {r.evidence_frames.length > 0 && <div className="mt-1 text-xs">Evidence: {r.evidence_frames.map(e => <button key={e} className="mr-1 text-sky-600 underline" onClick={() => onFrame(e)}>{e.slice(-5)}</button>)}</div>}
      </div>
      <div className="card text-sm">
        <h3 className="mb-2 font-semibold">Review queue ({open.length} open of {p.review_queue.length})</h3>
        <div className="max-h-96 overflow-y-auto"><table className="w-full"><thead><tr><th>Ref</th><th>Category</th><th>Reason</th><th></th></tr></thead>
          <tbody>{p.review_queue.map((q, i) => (
            <tr key={i} className={`border-t border-slate-100 dark:border-slate-800 ${q.resolved ? "opacity-40" : ""}`}>
              <td className="font-mono text-xs">{q.frame_ref ? <button className="text-sky-600 underline" onClick={() => onFrame(q.frame_ref)}>{q.ref_id}</button> : q.ref_id}</td>
              <td>{q.category}</td><td>{q.reason}</td>
              <td><button className="btn-ghost !py-0.5 text-xs" disabled={busy === q.ref_id + i} onClick={async () => { setBusy(q.ref_id + i); await api.resolve(p.sweep.id, q.ref_id, !q.resolved); await refresh(); setBusy(""); }}>{q.resolved ? "reopen" : "mark reviewed"}</button></td>
            </tr>))}</tbody></table></div>
      </div>
      <div className="card text-sm"><h3 className="mb-2 font-semibold">Processing</h3>
        <div className="grid grid-cols-2 gap-x-6 sm:grid-cols-3">{p.processing.map((s, i) => <div key={i} className="flex justify-between"><span>{s.stage.replace(/_/g, " ")}</span><span className="text-slate-500">{s.seconds}s {s.status !== "ok" && `(${s.status})`}</span></div>)}</div></div>
    </div>
  );
}
function Card({ k, v, s, small }: { k: string; v: any; s?: string; small?: boolean }) {
  return <div className="card"><div className="text-xs text-slate-500">{k}</div><div className={small ? "text-base font-semibold" : "text-2xl font-semibold"}>{v}</div>{s && <div className="text-xs text-slate-500">{s}</div>}</div>;
}
