"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { CheckDetection } from "@/components/CheckDetection";
import { Icon } from "@/components/Icons";
import { videoUrl } from "@/lib/api";
import { frameToTime, isSlowmo, timeToFrame } from "@/lib/frames";
import type { AnalysisResult, KeypointsResponse } from "@/types/analysis";

const EDGES: [string, string][] = [
  ["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"], ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"], ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"], ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
  ["nose", "left_eye"], ["nose", "right_eye"], ["left_eye", "left_ear"], ["right_eye", "right_ear"],
  ["left_ankle", "left_heel"], ["left_heel", "left_foot_index"], ["left_ankle", "left_foot_index"],
  ["right_ankle", "right_heel"], ["right_heel", "right_foot_index"], ["right_ankle", "right_foot_index"],
];
const GREEN = "#2BD576", AMBER = "#F5A524", BLUE = "#38BDF8", PINK = "#F472B6", VIOLET = "#A78BFA";
/** shoulder/hip line shorter than this × torso length = seen end-on (same rule as the backend's MIN_LINE_WIDTH) */
const MIN_LINE_WIDTH = 0.35;

export interface SeekRequest { frame: number; joint?: string | null; id: number }

type Pt = [number, number] | null;

function angleAt(a: [number, number], b: [number, number], c: [number, number]): number {
  const u = [a[0] - b[0], a[1] - b[1]], v = [c[0] - b[0], c[1] - b[1]];
  const cos = (u[0] * v[0] + u[1] * v[1]) / (Math.hypot(u[0], u[1]) * Math.hypot(v[0], v[1]));
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

function drawArc(ctx: CanvasRenderingContext2D, c: [number, number], a: [number, number], b: [number, number], r: number, color: string, label: string, s: number) {
  let a1 = Math.atan2(a[1] - c[1], a[0] - c[0]), a2 = Math.atan2(b[1] - c[1], b[0] - c[0]);
  let d = (a2 - a1 + 2 * Math.PI) % (2 * Math.PI);
  if (d > Math.PI) { [a1, a2] = [a2, a1]; d = 2 * Math.PI - d; }
  ctx.strokeStyle = color; ctx.lineWidth = 2.5 * s;
  ctx.beginPath(); ctx.arc(c[0], c[1], r, a1, a1 + d); ctx.stroke();
  const mid = a1 + d / 2, lx = c[0] + Math.cos(mid) * (r + 26 * s), ly = c[1] + Math.sin(mid) * (r + 26 * s);
  ctx.font = `600 ${14 * s}px ui-monospace, monospace`;
  const w = ctx.measureText(label).width + 12 * s;
  ctx.fillStyle = "rgba(10,11,13,0.86)"; ctx.fillRect(lx - w / 2, ly - 12 * s, w, 24 * s);
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.strokeRect(lx - w / 2, ly - 12 * s, w, 24 * s);
  ctx.fillStyle = color; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(label, lx, ly + 1);
}

export interface AngleWindow { bfc: number; release: number; fps: number }

/** The front-knee angle only means something from back foot contact until just after release, and only while the
 *  legs are apart (when they cross, which one is "front" is ambiguous). */
function kneeAngleShown(f: number, win: AngleWindow | undefined, P: (n: string) => Pt): boolean {
  if (!win) return true;
  if (f < win.bfc || f > win.release + Math.round(0.15 * win.fps)) return false;
  const la = P("left_ankle"), ra = P("right_ankle"), lh = P("left_hip"), rh = P("right_hip");
  if (!la || !ra) return false;
  if (lh && rh) {
    const leg = Math.hypot((lh[0] + rh[0]) / 2 - (la[0] + ra[0]) / 2, (lh[1] + rh[1]) / 2 - (la[1] + ra[1]) / 2);
    if (Math.abs(la[0] - ra[0]) < 0.08 * leg) return false;
  }
  return true;
}

/** Angles that mean something from in front of or behind the bowler (second clip): side bend, bowling-arm angle from
 *  vertical, and the shoulder and hip lines with their separation (only when neither line is seen end-on). */
function drawSecondViewAngles(ctx: CanvasRenderingContext2D, P: (n: string) => Pt, arm: string, s: number) {
  const lh = P("left_hip"), rh = P("right_hip"), ls = P("left_shoulder"), rs = P("right_shoulder");
  if (!(lh && rh && ls && rs)) return;
  const mh: [number, number] = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2];
  const ms: [number, number] = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
  const torso = Math.hypot(ms[0] - mh[0], ms[1] - mh[1]);
  if (torso < 5) return;
  // side bend: trunk (mid-hip → mid-shoulder) from vertical
  const up: [number, number] = [mh[0], mh[1] - torso * 0.9];
  ctx.setLineDash([4 * s, 5 * s]); ctx.strokeStyle = BLUE; ctx.lineWidth = 1.5 * s;
  ctx.beginPath(); ctx.moveTo(mh[0], mh[1]); ctx.lineTo(up[0], up[1]); ctx.stroke(); ctx.setLineDash([]);
  ctx.lineWidth = 2.5 * s; ctx.beginPath(); ctx.moveTo(mh[0], mh[1]); ctx.lineTo(ms[0], ms[1]); ctx.stroke();
  drawArc(ctx, mh, up, ms, 28 * s, BLUE, `bend ${Math.round(angleAt(up, mh, ms))}°`, s);
  // bowling arm from vertical, while the arm is up (around release)
  const sh = P(`${arm}_shoulder`), wr = P(`${arm}_wrist`);
  if (sh && wr && wr[1] < sh[1]) {
    const len = Math.hypot(wr[0] - sh[0], wr[1] - sh[1]);
    const vup: [number, number] = [sh[0], sh[1] - len];
    ctx.setLineDash([4 * s, 5 * s]); ctx.strokeStyle = AMBER; ctx.lineWidth = 1.5 * s;
    ctx.beginPath(); ctx.moveTo(sh[0], sh[1]); ctx.lineTo(vup[0], vup[1]); ctx.stroke(); ctx.setLineDash([]);
    drawArc(ctx, sh, vup, wr, 22 * s, AMBER, `arm ${Math.round(angleAt(vup, sh, wr))}°`, s);
  }
  // shoulder and hip lines, and the angle between them, only when both are seen clearly (not end-on)
  const sw = Math.hypot(ls[0] - rs[0], ls[1] - rs[1]), hw = Math.hypot(lh[0] - rh[0], lh[1] - rh[1]);
  if (sw >= MIN_LINE_WIDTH * torso && hw >= MIN_LINE_WIDTH * torso) {
    const line = (a: [number, number], b: [number, number], c: string) => {
      const dx = b[0] - a[0], dy = b[1] - a[1];
      ctx.strokeStyle = c; ctx.lineWidth = 2.5 * s;
      ctx.beginPath(); ctx.moveTo(a[0] - dx * 0.25, a[1] - dy * 0.25); ctx.lineTo(b[0] + dx * 0.25, b[1] + dy * 0.25); ctx.stroke();
    };
    line(ls, rs, PINK); line(lh, rh, VIOLET);
    const ang = (a: [number, number], b: [number, number]) => Math.atan2(b[1] - a[1], b[0] - a[0]);
    let d = Math.abs(((ang(ls, rs) - ang(lh, rh)) * 180) / Math.PI) % 180;
    d = Math.min(d, 180 - d);
    const x = Math.max(ls[0], rs[0]) + 14 * s, y = ms[1];
    const label = `sep ${Math.round(d)}°`;
    ctx.font = `600 ${14 * s}px ui-monospace, monospace`;
    const w = ctx.measureText(label).width + 12 * s;
    ctx.fillStyle = "rgba(10,11,13,0.86)"; ctx.fillRect(x, y - 12 * s, w, 24 * s);
    ctx.strokeStyle = PINK; ctx.lineWidth = 1; ctx.strokeRect(x, y - 12 * s, w, 24 * s);
    ctx.fillStyle = PINK; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(label, x + 6 * s, y + 1);
  }
}

export function drawSkeleton(ctx: CanvasRenderingContext2D, kp: KeypointsResponse, f: number, opts: { skeleton: boolean; angles: boolean; arm: string; color?: string; pulse?: { joint: string; t: number } | null; window?: AngleWindow; view?: "side" | "second" }) {
  const W = kp.width, H = kp.height;
  ctx.clearRect(0, 0, W, H);
  const fr = kp.frames[Math.max(0, Math.min(kp.frames.length - 1, f))];
  if (!fr) return;
  const s = Math.max(0.8, Math.min(W, H) / 600);
  const idx = (n: string) => kp.joints.indexOf(n);
  const P = (n: string): Pt => { const i = idx(n); const p = i >= 0 ? fr[i] : undefined; return p && p[2] >= 0.5 ? [p[0] * W, p[1] * H] : null; };
  const col = opts.color || GREEN;
  if (opts.skeleton) {
    ctx.lineCap = "round";
    for (const [a, b] of EDGES) {
      const pa = P(a), pb = P(b);
      if (!pa || !pb) continue;
      ctx.strokeStyle = a.startsWith(opts.arm) && (a.includes("elbow") || b.includes("wrist")) ? AMBER : col;
      ctx.lineWidth = 3.5 * s;
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    }
    for (const n of kp.joints) {
      const p = P(n);
      if (!p) continue;
      ctx.fillStyle = "#0A0B0D"; ctx.strokeStyle = col; ctx.lineWidth = 2 * s;
      ctx.beginPath(); ctx.arc(p[0], p[1], 4.5 * s, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
  }
  if (opts.angles && opts.view === "second") {
    drawSecondViewAngles(ctx, P, opts.arm, s);
  } else if (opts.angles) {
    const front = opts.arm === "right" ? "left" : "right";
    const hip = P(`${front}_hip`), knee = P(`${front}_knee`), ank = P(`${front}_ankle`);
    if (hip && knee && ank && kneeAngleShown(f, opts.window, P)) drawArc(ctx, knee, hip, ank, 24 * s, AMBER, `${Math.round(angleAt(hip, knee, ank))}°`, s);
    const lh = P("left_hip"), rh = P("right_hip"), ls = P("left_shoulder"), rs = P("right_shoulder");
    if (lh && rh && ls && rs) {
      const mh: [number, number] = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2], ms: [number, number] = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
      const up: [number, number] = [mh[0], mh[1] - 70 * s];
      ctx.setLineDash([4 * s, 5 * s]); ctx.strokeStyle = BLUE; ctx.lineWidth = 1.5 * s;
      ctx.beginPath(); ctx.moveTo(mh[0], mh[1]); ctx.lineTo(up[0], up[1]); ctx.stroke(); ctx.setLineDash([]);
      ctx.lineWidth = 2.5 * s; ctx.beginPath(); ctx.moveTo(mh[0], mh[1]); ctx.lineTo(ms[0], ms[1]); ctx.stroke();
      drawArc(ctx, mh, up, ms, 32 * s, BLUE, `${Math.round(angleAt(up, mh, ms))}°`, s);
    }
  }
  if (opts.pulse) {
    const p = P(opts.pulse.joint);
    if (p) {
      const k = (opts.pulse.t % 900) / 900;
      ctx.fillStyle = AMBER; ctx.beginPath(); ctx.arc(p[0], p[1], 8 * s, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = `rgba(245,165,36,${1 - k})`; ctx.lineWidth = 2.5 * s;
      ctx.beginPath(); ctx.arc(p[0], p[1], (8 + 22 * k) * s, 0, Math.PI * 2); ctx.stroke();
    }
  }
}

export interface KeyMoments { bfc: number; ffc: number; release: number }

interface Props {
  result: AnalysisResult;
  keypoints: KeypointsResponse | null;
  mediaQ: string | null;
  seek?: SeekRequest | null;
  defaultAngles?: boolean;
  /** "Correct key moments": called with the frames the user set; returns an error message or null */
  onReanalyze?: (m: KeyMoments) => Promise<string | null>;
  /** results with cached keypoints: corrections are recomputed in place (Check detection panel) */
  onRecomputed?: (r: AnalysisResult) => void;
}

type VFCVideo = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

export function VideoPlayer({ result, keypoints, mediaQ, seek, defaultAngles = false, onReanalyze, onRecomputed }: Props) {
  const v = result.video;
  const video = useRef<VFCVideo>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [frame, setFrame] = useState(0); // UI label/scrubber only, updated ≤ 10×/s while playing
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [skeleton, setSkeleton] = useState(true);
  const [angles, setAngles] = useState(defaultAngles);
  const [loadErr, setLoadErr] = useState(false);
  const frameRef = useRef(0);
  const lastUi = useRef(0);
  const pulseRef = useRef<{ joint: string; start: number } | null>(null);
  const optsRef = useRef({ skeleton, angles });
  optsRef.current = { skeleton, angles };
  useEffect(() => setAngles(defaultAngles), [defaultAngles]);

  // draw straight from refs: no React render per video frame
  const draw = useCallback(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx || !keypoints) return;
    const now = performance.now();
    const pu = pulseRef.current && now - pulseRef.current.start < 2700 ? { joint: pulseRef.current.joint, t: now - pulseRef.current.start } : null;
    drawSkeleton(ctx, keypoints, frameRef.current, { ...optsRef.current, arm: result.bowling_arm, pulse: pu,
      window: { bfc: result.events.bfc.frame, release: result.events.release.frame, fps: result.video.fps } });
    return !!pu;
  }, [keypoints, result.bowling_arm, result.events, result.video.fps]);

  const setCurrent = useCallback((f: number, forceUi: boolean) => {
    frameRef.current = f;
    draw();
    const now = performance.now();
    if (forceUi || now - lastUi.current > 100) {
      lastUi.current = now;
      setFrame(f);
    }
  }, [draw]);

  const goto = useCallback((f: number) => {
    const el = video.current;
    if (!el) return;
    el.pause();
    const ff = Math.max(0, Math.min(v.frame_count - 1, f));
    el.currentTime = frameToTime(ff, v);
    setCurrent(ff, true);
  }, [v, setCurrent]);

  useEffect(() => {
    const el = video.current;
    if (!el) return;
    let raf = 0, vfc = 0;
    const update = (t: number) => {
      const f = timeToFrame(t, v);
      if (f !== frameRef.current) setCurrent(f, el.paused);
    };
    if (el.requestVideoFrameCallback) {
      const cb = (_: number, meta: { mediaTime: number }) => { update(meta.mediaTime); vfc = el.requestVideoFrameCallback!(cb); };
      vfc = el.requestVideoFrameCallback(cb);
    } else {
      const loop = () => { update(el.currentTime); raf = requestAnimationFrame(loop); };
      raf = requestAnimationFrame(loop);
    }
    const onSettle = () => setCurrent(timeToFrame(el.currentTime, v), true);
    el.addEventListener("seeked", onSettle);
    el.addEventListener("pause", onSettle);
    return () => {
      cancelAnimationFrame(raf);
      if (el.cancelVideoFrameCallback) el.cancelVideoFrameCallback(vfc);
      el.removeEventListener("seeked", onSettle);
      el.removeEventListener("pause", onSettle);
    };
  }, [v, setCurrent]);

  useEffect(() => { draw(); }, [draw, skeleton, angles]);

  useEffect(() => {
    if (!seek) return;
    goto(seek.frame);
    pulseRef.current = seek.joint ? { joint: seek.joint, start: performance.now() } : null;
    let raf = 0;
    const anim = () => { if (draw()) raf = requestAnimationFrame(anim); };
    raf = requestAnimationFrame(anim);
    return () => cancelAnimationFrame(raf);
  }, [seek, goto, draw]);

  useEffect(() => { if (video.current) video.current.playbackRate = rate; }, [rate]);

  const toggle = useCallback(() => {
    const el = video.current;
    if (!el) return;
    if (el.paused) el.play().catch(() => {}); else el.pause();
  }, []);

  // keyboard shortcuts only while the player has focus (doesn't hijack page scrolling or screen-reader keys)
  const onKey = (e: React.KeyboardEvent) => {
    const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
    if (tag === "input" || tag === "button" || tag === "select") return;
    if (e.key === " ") { e.preventDefault(); toggle(); }
    else if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); goto(frameRef.current + (e.key === "ArrowRight" ? 1 : -1)); }
    else if (e.key === "1") goto(result.events.bfc.frame);
    else if (e.key === "2") goto(result.events.ffc.frame);
    else if (e.key === "3") goto(result.events.release.frame);
  };

  const evs = [
    { l: "BFC", g: "BFC", f: result.events.bfc.frame },
    { l: "FFC", g: "FFC", f: result.events.ffc.frame },
    { l: "RELEASE", g: "release", f: result.events.release.frame },
  ];
  const evNow = evs.find((e) => Math.abs(e.f - frame) <= 1);
  const last = Math.max(1, v.frame_count - 1);
  return (
    <div className="col" style={{ gap: 12 }} tabIndex={0} onKeyDown={onKey} aria-label="Delivery video player. Space plays or pauses, arrow keys step one frame, 1 2 3 jump to key moments.">
      <div style={{ position: "relative", aspectRatio: `${v.width} / ${v.height}`, width: `min(100%, calc(70vh * ${v.width} / ${v.height}))`, margin: "0 auto", background: "#000", borderRadius: 12, overflow: "hidden" }}>
        {mediaQ && (
          <video ref={video} src={videoUrl(result.id, mediaQ, true, result.recomputed_at)} playsInline muted preload="auto" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
            onError={() => setLoadErr(true)} onClick={toggle} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} aria-label={`Delivery video, frame ${frame} of ${last}`} />
        )}
        <canvas ref={canvas} width={keypoints?.width || v.width} height={keypoints?.height || v.height} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
        <div style={{ position: "absolute", left: 10, top: 10 }} className="row">
          <span className="chip mono" style={{ background: "var(--glass)", color: "var(--tx)" }}>F {String(frame).padStart(3, "0")} · {(frame / v.fps).toFixed(3)} s</span>
          {isSlowmo(frame, v) && <span className="chip a" style={{ background: "var(--glass)" }}>slow motion ×{v.slowmo.factor}</span>}
        </div>
        {evNow && <span className="chip" style={{ position: "absolute", right: 10, top: 10, background: "#2BD576", color: "#04140A", borderColor: "#2BD576", fontWeight: 700 }}>{evNow.l}</span>}
        {loadErr && <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", background: "rgba(0,0,0,.7)", color: "#fff", padding: 16, textAlign: "center" }}>The video couldn't be loaded from the server. Check the backend is running, then reload.</div>}
      </div>
      <div style={{ position: "relative", paddingTop: 62 }}>
        {evs.map((e, i) => (
          <button key={e.l} className="btn sm" onClick={() => goto(e.f)} aria-label={`Jump to ${e.l}, frame ${e.f}`}
            style={{ position: "absolute", top: i * 20, left: `calc(${(e.f / last) * 100}% - 28px)`, padding: "2px 8px", fontSize: 11, fontWeight: 600,
              background: frame === e.f ? "#2BD576" : "var(--s2)", color: frame === e.f ? "#04140A" : "var(--tx)" }}>
            {e.l}
          </button>
        ))}
        <input type="range" min={0} max={last} value={frame} onChange={(e) => goto(Number(e.target.value))} aria-label="Scrub frames" style={{ width: "100%", accentColor: "#2BD576" }} />
      </div>
      <div className="spread">
        <div className="row" style={{ gap: 6 }}>
          <button className="btn sm" onClick={() => goto(frameRef.current - 1)} aria-label="Previous frame"><Icon name="prev" size={16} /></button>
          <button className="btn sm primary" onClick={toggle} aria-label={playing ? "Pause" : "Play"}><Icon name={playing ? "pause" : "play"} size={16} /></button>
          <button className="btn sm" onClick={() => goto(frameRef.current + 1)} aria-label="Next frame"><Icon name="next" size={16} /></button>
          <div className="seg" role="group" aria-label="Playback speed">
            {[0.1, 0.25, 0.5, 1].map((r) => <button key={r} aria-pressed={rate === r} onClick={() => setRate(r)}>{r}×</button>)}
          </div>
        </div>
        <div className="row" style={{ gap: 6 }}>
          <button className="btn sm" aria-pressed={skeleton} onClick={() => setSkeleton(!skeleton)} style={skeleton ? { color: "var(--gtx)", borderColor: "rgba(43,213,118,.4)" } : undefined}><Icon name="skel" size={16} /> Skeleton</button>
          <button className="btn sm" aria-pressed={angles} onClick={() => setAngles(!angles)} style={angles ? { color: "var(--gtx)", borderColor: "rgba(43,213,118,.4)" } : undefined}><Icon name="angle" size={16} /> Angles</button>
        </div>
      </div>
      <p className="xs fa">Click the player, then: Space play/pause · ← → step a frame · 1 BFC · 2 FFC · 3 release. The release window plays in {v.slowmo.factor}× slow motion.</p>

      {result.can_recompute && onRecomputed
        ? <CheckDetection result={result} frame={frame} goto={goto} mediaQ={mediaQ} onRecomputed={onRecomputed} />
        : onReanalyze && <CorrectMoments result={result} frame={frame} goto={goto} onReanalyze={onReanalyze} />}
    </div>
  );
}


const MOMENTS: { k: keyof KeyMoments; label: string; hint: string }[] = [
  { k: "bfc", label: "Back foot contact", hint: "the back foot first touches the ground at the end of the bound" },
  { k: "ffc", label: "Front foot contact", hint: "the front foot (heel) first touches the ground" },
  { k: "release", label: "Release", hint: "the ball leaves the hand (arm about vertical)" },
];

/** Fix key moments by hand when the automatic ones don't match the video, then re-analyse with them. */
function CorrectMoments({ result, frame, goto, onReanalyze }: { result: AnalysisResult; frame: number; goto: (f: number) => void;
  onReanalyze: (m: KeyMoments) => Promise<string | null> }) {
  const auto: KeyMoments = { bfc: result.events.bfc.frame, ffc: result.events.ffc.frame, release: result.events.release.frame };
  const [open, setOpen] = useState(!!result.key_moments_estimated);
  const [m, setM] = useState<KeyMoments>(auto);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const changed = m.bfc !== auto.bfc || m.ffc !== auto.ffc || m.release !== auto.release;
  const fps = result.video.fps;
  const order = !(m.bfc < m.ffc && m.ffc < m.release) ? "Back foot, front foot and release must be in that order." : null;
  if (!open) {
    return (
      <button className="btn sm ghost" style={{ alignSelf: "flex-start" }} onClick={() => setOpen(true)}>
        Key moments don't match the video? Correct them
      </button>
    );
  }
  return (
    <div className="card col" style={{ background: "var(--s2)", gap: 10 }} aria-label="Correct key moments">
      <div className="spread">
        <b className="small">Correct key moments</b>
        <button className="btn sm ghost" onClick={() => { setOpen(false); setM(auto); setErr(null); }}>Close</button>
      </div>
      {result.key_moments_estimated && <p className="small" style={{ color: "var(--atx)" }}>These key moments were <b>estimated</b>, not detected:
        please check each one (press its frame number to jump there) and correct any that are off.</p>}
      <p className="small mu">Step frame by frame (the ‹ › buttons or arrow keys) to each moment, then press “Set”. The delivery is then
        analysed again with your frames, so every measurement uses them.</p>
      {MOMENTS.map(({ k, label, hint }) => (
        <div key={k} className="spread" style={{ flexWrap: "nowrap", gap: 8 }}>
          <div style={{ minWidth: 0 }}>
            <div className="small"><b>{label}</b> · <button className="btn sm ghost" style={{ padding: "0 4px" }} onClick={() => goto(m[k])}>
              F {m[k]} ({(m[k] / fps).toFixed(3)} s)</button>{m[k] !== auto[k] && <span className="xs mu"> (was F {auto[k]})</span>}</div>
            <div className="xs mu">{hint}</div>
          </div>
          <button className="btn sm" onClick={() => setM({ ...m, [k]: frame })} aria-label={`Set ${label} to frame ${frame}`}>Set to F {frame}</button>
        </div>
      ))}
      {(order || err) && <p className="small" style={{ color: "var(--rtx)" }}>{order || err}</p>}
      <div className="row" style={{ gap: 6 }}>
        <button className="btn sm primary" disabled={(!changed && !result.key_moments_estimated) || !!order || busy}
          onClick={async () => { setBusy(true); setErr(null); const e = await onReanalyze(m); if (e) { setErr(e); setBusy(false); } }}>
          {busy ? "Starting…" : !changed && result.key_moments_estimated ? "These are right: confirm them" : "Re-analyse with these moments"}</button>
        {changed && <button className="btn sm ghost" onClick={() => setM(auto)}>Reset</button>}
      </div>
    </div>
  );
}
