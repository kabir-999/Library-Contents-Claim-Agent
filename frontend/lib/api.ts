import type { Packet } from "./types";
export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const WS = API.replace(/^http/, "ws");
async function j<T>(r: Response): Promise<T> { if (!r.ok) throw new Error(`${r.status} ${await r.text()}`); return r.json(); }
export const api = {
  health: () => fetch(`${API}/api/health`).then(j<any>),
  markets: () => fetch(`${API}/api/markets`).then(j<{ code: string; name: string; currency: string }[]>),
  start: (b: { country: string; currency: string; device: string; camera_height_cm?: number | null }) =>
    fetch(`${API}/api/sweeps`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then(j<{ sweep_id: string; country: string; currency: string }>),
  packet: (id: string) => fetch(`${API}/api/sweeps/${id}/packet`).then(j<Packet>),
  events: (id: string, after = 0) => fetch(`${API}/api/sweeps/${id}/events?after=${after}`).then(j<any[]>),
  sweeps: () => fetch(`${API}/api/sweeps`).then(j<any[]>),
  secondCountry: (id: string, country: string, n = 10) =>
    fetch(`${API}/api/sweeps/${id}/second-country`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ country, n }) }).then(j<any>),
  resolve: (id: string, ref: string, resolved: boolean) =>
    fetch(`${API}/api/sweeps/${id}/review/${encodeURIComponent(ref)}/resolve`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ resolved }) }).then(j<any>),
  twoPoint: (id: string, b: { frame_id: string; p1: [number, number]; p2: [number, number]; real_cm: number }) =>
    fetch(`${API}/api/sweeps/${id}/calibration/two-point`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then(j<any>),
  frameUrl: (frame: string) => `${API}/api/frames/${frame}.jpg`,
  cropUrl: (id: string, ent: string) => `${API}/api/sweeps/${id}/crops/${ent}.jpg`,
  jsonUrl: (id: string) => `${API}/api/sweeps/${id}/packet/download`,
  pdfUrl: (id: string) => `${API}/api/sweeps/${id}/report.pdf`,
};
export const money = (v: number | null | undefined, cur: string) => (v == null ? "—" : new Intl.NumberFormat(undefined, { style: "currency", currency: cur || "USD" }).format(v));
