"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import type { Packet } from "@/lib/types";
import StartClaim, { StartResult } from "@/components/StartClaim";
import LiveSweep from "@/components/LiveSweep";
import Results from "@/components/Results";

export default function Home() {
  const [start, setStart] = useState<StartResult | null>(null);
  const [packet, setPacket] = useState<Packet | null>(null);
  if (packet) return <Results p={packet} onNew={() => { setPacket(null); setStart(null); }} />;
  if (start) return <LiveSweep start={start} onEnded={async () => { start.stream.getTracks().forEach(t => t.stop()); setPacket(await api.packet(start.sweepId)); }} />;
  return <StartClaim onStart={setStart} />;
}
