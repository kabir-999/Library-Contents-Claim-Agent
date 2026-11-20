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
export function say(text: string, onend?: () => void, onerr?: () => void): boolean {
  if (typeof speechSynthesis === "undefined") return false;
  const u = new SpeechSynthesisUtterance(text); u.rate = 1.1; u.volume = 1;
  const v = pickVoice(); if (v) { u.voice = v; u.lang = v.lang; }
  u.onend = () => onend?.(); u.onerror = () => onerr?.();
  speechSynthesis.cancel(); speechSynthesis.speak(u);
  return true;
}
