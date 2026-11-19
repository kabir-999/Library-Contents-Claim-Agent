"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Packet } from "@/lib/types";
import StartClaim, { StartResult } from "@/components/StartClaim";
import LiveSweep from "@/components/LiveSweep";
import Inventory from "@/components/Inventory";
import Summary from "@/components/Summary";
import PacketView, { FrameModal } from "@/components/PacketView";

type Tab = "inventory" | "summary" | "packet";

export default function Home() {
  const [start, setStart] = useState<StartResult | null>(null);
  const [phase, setPhase] = useState<"start" | "live" | "results">("start");
  const [packet, setPacket] = useState<Packet | null>(null);
  const [tab, setTab] = useState<Tab>("summary");
  const [frame, setFrame] = useState<string | null>(null);
  const [past, setPast] = useState<any[]>([]);
  const sid = start?.sweepId;

  useEffect(() => { api.sweeps().then(setPast).catch(() => {}); }, []);
  const load = useCallback(async (id: string) => { setPacket(await api.packet(id)); }, []);
  const refresh = useCallback(async () => { if (packet) await load(packet.sweep.id); }, [packet, load]);

  if (phase === "start") return (<>
    <StartClaim onStart={r => { setStart(r); setPhase("live"); }} />
    {past.filter(s => s.state === "complete").length > 0 && <div className="mx-auto mt-6 max-w-2xl text-sm"><h3 className="mb-1 font-medium">Completed sweeps</h3>
      {past.filter(s => s.state === "complete").map(s => <button key={s.id} className="btn-ghost mr-2 mt-1" onClick={async () => { await load(s.id); setPhase("results"); }}>{s.id}</button>)}</div>}
  </>);
  if (phase === "live" && start) return <LiveSweep start={start} onEnded={async () => { start.stream.getTracks().forEach(t => t.stop()); await load(start.sweepId); setPhase("results"); }} />;
  if (!packet) return <div>Loading…</div>;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-4 text-xl font-semibold">Claim {packet.sweep.id}</h1>
        {(["summary", "inventory", "packet"] as Tab[]).map(t => <button key={t} className={`btn-ghost ${tab === t ? "!bg-slate-900 !text-white dark:!bg-slate-100 dark:!text-slate-900" : ""}`} onClick={() => setTab(t)}>{t === "summary" ? "Claim summary" : t === "inventory" ? "Inventory" : "Claim packet"}</button>)}
        <button className="btn-ghost ml-auto" onClick={() => { setPhase("start"); setPacket(null); setStart(null); }}>New claim</button>
      </div>
      {tab === "summary" && <Summary p={packet} onFrame={setFrame} refresh={refresh} />}
      {tab === "inventory" && <Inventory p={packet} onFrame={setFrame} />}
      {tab === "packet" && <PacketView p={packet} refresh={refresh} frame={frame} setFrame={setFrame} />}
      {tab !== "packet" && frame && <FrameModal sid={packet.sweep.id} frame={frame} p={packet} onClose={() => setFrame(null)} refresh={refresh} />}
    </div>
  );
}
