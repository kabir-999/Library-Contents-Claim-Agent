/** Camera pitch (below horizon) and compass-style heading from DeviceOrientation (Z-X'-Y'' Euler, W3C spec).
 * The back camera looks along -z of the device frame, independent of portrait/landscape. */
export interface Orient { alpha: number | null; beta: number | null; gamma: number | null; pitch_down: number | null; heading: number | null }
export function fromEvent(e: DeviceOrientationEvent): Orient {
  if (e.alpha == null || e.beta == null || e.gamma == null) return { alpha: e.alpha, beta: e.beta, gamma: e.gamma, pitch_down: null, heading: null };
  const r = Math.PI / 180, a = e.alpha * r, b = e.beta * r, g = e.gamma * r;
  const zx = Math.cos(a) * Math.sin(g) + Math.sin(a) * Math.sin(b) * Math.cos(g);
  const zy = Math.sin(a) * Math.sin(g) - Math.cos(a) * Math.sin(b) * Math.cos(g);
  const zz = Math.cos(b) * Math.cos(g);
  const pitch_down = Math.asin(Math.max(-1, Math.min(1, zz))) / r;       // camera forward = -z; up-component = -zz
  const heading = ((Math.atan2(-zx, -zy) / r) + 360) % 360;
  return { alpha: e.alpha, beta: e.beta, gamma: e.gamma, pitch_down, heading };
}
