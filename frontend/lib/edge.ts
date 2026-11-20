// Edge compute in the browser: decide per device what to capture, and drop frames that aren't worth sending.
// (Sharpness = variance of a Laplacian on a 96x54 grayscale copy; motion = mean abs difference to the last sent frame.)
export type Profile = { kind: "glasses" | "phone" | "desktop"; width: number; intervalMs: number; quality: number };

export function deviceProfile(forceGlasses = false): Profile {
  const ua = typeof navigator !== "undefined" ? navigator.userAgent : "";
  const q = typeof location !== "undefined" ? new URLSearchParams(location.search).get("mode") : null;
  if (forceGlasses || q === "glasses" || /Vuzix|RealWear|XREAL|Rokid|Glass|Quest|Pico|HoloLens|ThinkReality/i.test(ua))
    return { kind: "glasses", width: 640, intervalMs: 200, quality: 0.7 };      // small, low-power
  if (/Android|iPhone|iPad|Mobile/i.test(ua))
    return { kind: "phone", width: 960, intervalMs: 150, quality: 0.78 };
  return { kind: "desktop", width: 1280, intervalMs: 150, quality: 0.82 };
}

export class EdgeFilter {
  private c = typeof document !== "undefined" ? document.createElement("canvas") : null;
  private prev: Uint8ClampedArray | null = null;
  private lastSent = 0;
  sent = 0; skipped = 0;
  /** true if this frame is sharp enough and changed (or it's been >2 s since the last one). */
  check(v: HTMLVideoElement): boolean {
    if (!this.c) return true;
    const W = 96, H = 54; this.c.width = W; this.c.height = H;
    const g = this.c.getContext("2d", { willReadFrequently: true })!; g.drawImage(v, 0, 0, W, H);
    const px = g.getImageData(0, 0, W, H).data, gray = new Uint8ClampedArray(W * H);
    for (let i = 0; i < W * H; i++) gray[i] = (px[4 * i] * 77 + px[4 * i + 1] * 150 + px[4 * i + 2] * 29) >> 8;
    let s = 0, s2 = 0, n = 0;
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x, l = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - W] - gray[i + W]; s += l; s2 += l * l; n++;
    }
    const sharp = s2 / n - (s / n) ** 2;
    let diff = 99; if (this.prev) { diff = 0; for (let i = 0; i < gray.length; i++) diff += Math.abs(gray[i] - this.prev[i]); diff /= gray.length; }
    const now = performance.now(), stale = now - this.lastSent > 2000;
    const ok = sharp > 12 && (diff > 1.5 || stale);
    if (ok) { this.prev = gray; this.lastSent = now; this.sent++; } else this.skipped++;
    return ok;
  }
}
