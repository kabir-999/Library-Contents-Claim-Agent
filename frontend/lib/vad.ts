/** Mic capture -> 16 kHz mono PCM utterances with an energy VAD. Emits speech start/end immediately so the
 * agent can be interrupted (barge-in); sends one utterance blob to the server for Whisper. */
export interface VadHandlers { onStart: () => void; onEnd: () => void; onUtterance: (pcm16: Int16Array) => void; isAgentSpeaking: () => boolean }

const WORKLET = `class P extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('pcm-tap',P)`;

export async function startVad(stream: MediaStream, h: VadHandlers): Promise<() => void> {
  const ctx = new AudioContext();
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
  await ctx.audioWorklet.addModule(url);
  const src = ctx.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
  const node = new AudioWorkletNode(ctx, "pcm-tap");
  src.connect(node);
  const ratio = ctx.sampleRate / 16000;
  let noise = 0.01, speaking = false, voiced = 0, silent = 0, buf: Float32Array[] = [], pre: Float32Array[] = [];
  const frameSec = 128 / ctx.sampleRate;
  node.port.onmessage = (ev) => {
    const x: Float32Array = ev.data;
    let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    const rms = Math.sqrt(s / x.length);
    // threshold adapts to the noise floor; stricter while the agent is talking (speaker bleed)
    const thr = Math.max(0.02, noise * (h.isAgentSpeaking() ? 5 : 3));
    const loud = rms > thr;
    if (!speaking) {
      noise = 0.98 * noise + 0.02 * rms;
      pre.push(x); if (pre.length > 8) pre.shift();
      voiced = loud ? voiced + 1 : 0;
      if (voiced * frameSec > (h.isAgentSpeaking() ? 0.25 : 0.12)) { speaking = true; silent = 0; buf = [...pre]; pre = []; h.onStart(); }
    } else {
      buf.push(x);
      silent = loud ? 0 : silent + 1;
      if (silent * frameSec > 0.8) {
        speaking = false; voiced = 0; h.onEnd();
        const n = buf.reduce((a, b) => a + b.length, 0), all = new Float32Array(n); let o = 0; for (const b of buf) { all.set(b, o); o += b.length; }
        buf = [];
        const outN = Math.floor(n / ratio), out = new Int16Array(outN);
        for (let i = 0; i < outN; i++) { const v = all[Math.floor(i * ratio)]; out[i] = Math.max(-1, Math.min(1, v)) * 32767; }
        if (outN > 16000 * 0.3) h.onUtterance(out);
      }
    }
  };
  return () => { node.disconnect(); src.disconnect(); ctx.close(); URL.revokeObjectURL(url); };
}
export function pcmToB64(p: Int16Array): string {
  const u8 = new Uint8Array(p.buffer); let s = ""; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000)); return btoa(s);
}
