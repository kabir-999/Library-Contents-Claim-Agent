// Browser text-to-speech that survives phone autoplay rules (iOS Safari/Chrome need a user gesture first).
let unlocked = false;
export function pickVoice(): SpeechSynthesisVoice | undefined {
  const vs = typeof speechSynthesis !== "undefined" ? speechSynthesis.getVoices() : [];
  return vs.find(v => /en[-_](IN|GB|US)/i.test(v.lang) && /samantha|daniel|karen|moira|google|rishi|veena/i.test(v.name)) || vs.find(v => /^en/i.test(v.lang));
}
/** Call from a tap/click handler. Safe to call repeatedly. */
export function unlockSpeech(): boolean {
  if (typeof speechSynthesis === "undefined") return false;
  try {
    speechSynthesis.getVoices();
    const u = new SpeechSynthesisUtterance(" "); u.volume = 0; speechSynthesis.speak(u);   // silent utterance inside the gesture
    unlocked = true;
  } catch { /* ignore */ }
  return unlocked;
}
export const speechUnlocked = () => unlocked;

let lastCancel = 0;
/** Cancel any speech and remember when (Chrome drops an utterance spoken in the same instant as a cancel()). */
export function cancelSpeech() {
  if (typeof speechSynthesis === "undefined") return;
  try { speechSynthesis.cancel(); lastCancel = performance.now(); } catch { /* ignore */ }
}

/** Speak `text`. Returns false if speech is unavailable; calls onerr if the browser refuses or stays silent. */
export function say(text: string, onend?: () => void, onerr?: () => void): boolean {
  if (typeof speechSynthesis === "undefined") return false;
  const synth = speechSynthesis;
  const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; u.volume = 1;
  const v = pickVoice(); if (v) { u.voice = v; u.lang = v.lang; }
  let started = false, finished = false;
  u.onstart = () => { started = true; };
  u.onend = () => { finished = true; onend?.(); };
  u.onerror = (e: any) => { if (e?.error === "interrupted" || e?.error === "canceled") return; finished = true; onerr?.(); };
  const go = () => {
    try { synth.resume(); synth.speak(u); } catch { onerr?.(); return; }
    // keep Chrome from cutting long speech after ~15 s, and detect a silent refusal
    const keep = setInterval(() => { if (finished) clearInterval(keep); else if (synth.speaking) { synth.pause(); synth.resume(); } }, 10000);
    setTimeout(() => { if (!started && !finished && !synth.speaking) { clearInterval(keep); onerr?.(); } }, 2500);
  };
  if (synth.speaking || synth.pending) { cancelSpeech(); }
  const wait = Math.max(0, 180 - (performance.now() - lastCancel));   // never speak right after a cancel
  if (wait > 0) setTimeout(go, wait); else go();
  return true;
}
