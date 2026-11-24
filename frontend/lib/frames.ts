// The served videos repeat every frame inside the slow-motion window `factor` times, so video time ↔ source frame
// is piecewise linear. These helpers convert both ways.
import type { VideoInfo } from "@/types/analysis";

export function frameToTime(f: number, v: VideoInfo): number {
  const { start_frame: s0, end_frame: s1, factor: k } = v.slowmo;
  let u: number;
  if (f < s0) u = f;
  else if (f <= s1) u = s0 + (f - s0) * k;
  else u = s0 + (s1 - s0 + 1) * k + (f - s1 - 1);
  const inSlow = f >= s0 && f <= s1;
  return (u + (inSlow ? k / 2 : 0.5)) / v.fps; // land in the middle of the frame's display interval
}

export function timeToFrame(t: number, v: VideoInfo): number {
  const { start_frame: s0, end_frame: s1, factor: k } = v.slowmo;
  const u = Math.floor(t * v.fps + 1e-6);
  let f: number;
  if (u < s0) f = u;
  else if (u < s0 + (s1 - s0 + 1) * k) f = s0 + Math.floor((u - s0) / k);
  else f = s1 + 1 + (u - s0 - (s1 - s0 + 1) * k);
  return Math.max(0, Math.min(v.frame_count - 1, f));
}

export function isSlowmo(f: number, v: VideoInfo): boolean {
  return f >= v.slowmo.start_frame && f <= v.slowmo.end_frame;
}
