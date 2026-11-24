"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/** storage/debug/{id}/pose3d.json: MotionBERT's lifted pose, camera coordinates (x right, y down, z away from the
 *  camera), pelvis at the origin, one entry per video frame (null where nothing was lifted). */
export interface Pose3D {
  fps: number;
  n: number;
  joints: string[];
  bones: [string, string][];
  roles: { front_leg: string | null; back_leg: string | null; bowling_arm: string | null };
  events: { bfc: number; ffc: number; release: number };
  frames: (number[][] | null)[];
}

const ROLE_COLOR = { front: "#F05050", back: "#3C96F0", bowling: "#2BD576", other: "#9AA0A6" } as const;
type Role = keyof typeof ROLE_COLOR;
const PRESETS: { label: string; yaw: number; pitch: number }[] = [
  { label: "3/4", yaw: -0.75, pitch: 0.35 },
  { label: "Camera", yaw: 0, pitch: 0 },
  { label: "Side", yaw: Math.PI / 2, pitch: 0 },
  { label: "Front", yaw: 0, pitch: 0 },
  { label: "Top", yaw: 0, pitch: Math.PI / 2 - 0.01 },
];

function boneRole(a: string, b: string, roles: Pose3D["roles"]): Role {
  const side = (n: string) => (n.startsWith("l_") ? "left" : n.startsWith("r_") ? "right" : null);
  const limb = (n: string) => n.replace(/^[lr]_/, "");
  const s = side(b) ?? side(a);
  if (!s) return "other";
  const leg = ["knee", "ankle"].includes(limb(b));
  const arm = ["elbow", "wrist"].includes(limb(b));
  if (leg && s === roles.front_leg) return "front";
  if (leg && s === roles.back_leg) return "back";
  if (arm && s === roles.bowling_arm) return "bowling";
  return "other";
}

// Body mode: limb thickness as a fraction of the figure's standing height (roughly an adult's proportions).
const BODY_W: Record<string, number> = {
  "pelvis-r_hip": 0.09, "pelvis-l_hip": 0.09, "r_hip-r_knee": 0.085, "l_hip-l_knee": 0.085,
  "r_knee-r_ankle": 0.06, "l_knee-l_ankle": 0.06, "pelvis-spine": 0.15, "spine-thorax": 0.17, "thorax-nose": 0.055,
  "nose-head": 0.04, "thorax-l_shoulder": 0.07, "thorax-r_shoulder": 0.07, "l_shoulder-l_elbow": 0.055,
  "r_shoulder-r_elbow": 0.055, "l_elbow-l_wrist": 0.045, "r_elbow-r_wrist": 0.045,
};
const BODY_COLOR = { front: "#E25B5B", back: "#4A90D9", bowling: "#33C27A", other: "#B9BEC5" } as const;

function shade(hex: string, f: number): string {
  const n = parseInt(hex.slice(1), 16);
  const c = (v: number) => Math.max(0, Math.min(255, Math.round(f >= 0 ? v + (255 - v) * f : v * (1 + f))));
  return `rgb(${c((n >> 16) & 255)},${c((n >> 8) & 255)},${c(n & 255)})`;
}

type Proj = (p: number[]) => [number, number, number, number];

/** Solid figure: every limb a shaded capsule (dark rim, base colour, highlight along the lit side), the torso a filled
 *  shoulders-hips panel, the head a lit sphere; all drawn far to near so nearer parts cover farther ones. */
function drawBody(g: CanvasRenderingContext2D, pose: number[][], idx: Record<string, number>, P: Proj, scale: number,
                  data: Pose3D) {
  type Item = { z: number; paint: () => void };
  const items: Item[] = [];
  const capsule = (a: string, b: string, role: Role) => {
    const pa = P(pose[idx[a]]), pb = P(pose[idx[b]]);
    const wpx = (BODY_W[`${a}-${b}`] ?? 0.05) * scale * (pa[3] + pb[3]) / 2;
    const base = BODY_COLOR[role];
    items.push({
      z: (pa[2] + pb[2]) / 2,
      paint: () => {
        const dx = pb[0] - pa[0], dy = pb[1] - pa[1], len = Math.hypot(dx, dy) || 1;
        let nx = -dy / len, ny = dx / len;
        if (nx + ny > 0) { nx = -nx; ny = -ny; } // highlight on the side facing the light (upper left)
        const line = (wd: number, col: string, ox = 0, oy = 0) => {
          g.strokeStyle = col; g.lineWidth = wd;
          g.beginPath(); g.moveTo(pa[0] + ox, pa[1] + oy); g.lineTo(pb[0] + ox, pb[1] + oy); g.stroke();
        };
        line(wpx + 2, shade(base, -0.55));
        line(wpx, shade(base, -0.12));
        line(wpx * 0.55, base, nx * wpx * 0.12, ny * wpx * 0.12);
        line(wpx * 0.18, shade(base, 0.45), nx * wpx * 0.24, ny * wpx * 0.24);
      },
    });
  };
  for (const [a, b] of data.bones) if (`${a}-${b}` !== "nose-head") capsule(a, b, boneRole(a, b, data.roles));
  // torso panel between the shoulders and hips (gives the chest width when the bowler faces the camera)
  const quad = ["l_shoulder", "r_shoulder", "r_hip", "l_hip"].map((j) => P(pose[idx[j]]));
  items.push({
    z: quad.reduce((s, q) => s + q[2], 0) / 4 - 1e-4,
    paint: () => {
      g.fillStyle = shade(BODY_COLOR.other, -0.18);
      g.strokeStyle = shade(BODY_COLOR.other, -0.55);
      g.lineWidth = 1.5;
      g.beginPath();
      quad.forEach((q, i) => (i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])));
      g.closePath(); g.fill(); g.stroke();
    },
  });
  // head: a sphere centred between the nose and the head top
  const hn = pose[idx.nose], ht = pose[idx.head];
  const hc = P(hn.map((v, i) => (v + ht[i]) / 2));
  const r = 0.065 * scale * hc[3];
  items.push({
    z: hc[2],
    paint: () => {
      const grd = g.createRadialGradient(hc[0] - r * 0.35, hc[1] - r * 0.35, r * 0.1, hc[0], hc[1], r);
      grd.addColorStop(0, shade(BODY_COLOR.other, 0.5));
      grd.addColorStop(1, shade(BODY_COLOR.other, -0.35));
      g.fillStyle = grd;
      g.beginPath(); g.arc(hc[0], hc[1], r, 0, Math.PI * 2); g.fill();
      g.strokeStyle = shade(BODY_COLOR.other, -0.6); g.lineWidth = 1; g.stroke();
    },
  });
  g.lineCap = "round";
  items.sort((s, t) => s.z - t.z).forEach((it) => it.paint());
}

/** Drag to rotate (mouse or touch); presets for the camera's view, side-on, front-on and from above. */
export function Skeleton3D({ data, frame }: { data: Pose3D; frame: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  // Open at a three-quarter view from slightly above, so the depth is visible straight away.
  const [yaw, setYaw] = useState(-0.75);
  const [pitch, setPitch] = useState(0.35);
  const [spin, setSpin] = useState(false);
  const [body, setBody] = useState(true);
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);
  const idx = useMemo(() => Object.fromEntries(data.joints.map((j, i) => [j, i])), [data.joints]);

  // "Front" looks along the line of the run with the bowler coming toward the viewer: the yaw that turns the
  // back-foot → front-foot direction (at FFC) to point out of the screen.
  const runYaw = useMemo(() => {
    const ff = data.frames[data.events.ffc];
    const fl = data.roles.front_leg?.[0], bl = data.roles.back_leg?.[0];
    if (!ff || !fl || !bl) return Math.PI / 2;
    const a = ff[idx[`${fl}_ankle`]], b = ff[idx[`${bl}_ankle`]];
    const dx = a[0] - b[0], dz = -(a[2] - b[2]); // world X = x, world Z = −z
    return Math.atan2(-dx, dz);
  }, [data, idx]);

  // Size from the whole clip (the tallest the figure stands, feet to head), so it doesn't change between frames.
  const scale = useMemo(() => {
    let tall = 0.2;
    for (const f of data.frames) {
      if (!f) continue;
      const feet = Math.max(f[idx.l_ankle][1], f[idx.r_ankle][1]);
      tall = Math.max(tall, feet - Math.min(...f.map((p) => p[1])));
    }
    return tall;
  }, [data, idx]);

  const nearest = useMemo(() => {
    if (data.frames[frame]) return frame;
    let best = -1;
    for (let d = 1; d < 6; d++) {
      if (data.frames[frame - d]) { best = frame - d; break; }
      if (data.frames[frame + d]) { best = frame + d; break; }
    }
    return best;
  }, [data, frame]);

  const draw = useCallback(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const pose = nearest >= 0 ? data.frames[nearest] : null;
    // This frame's floor: under the lower foot. Everything is drawn feet-on-the-floor, turning about the point half
    // the figure's height above it.
    const floorY = pose ? Math.max(pose[idx.l_ankle][1], pose[idx.r_ankle][1]) : 0;
    const floor = -scale / 2;
    const k = (Math.min(w, h) * 0.36) / (scale / 2);
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const D = 3.5 * scale; // viewer distance: perspective, nearer parts drawn larger
    // world: X right, Y up, Z toward the camera; rotate by yaw (about Y) then pitch (about X), then perspective
    const proj = (X: number, Y: number, Z: number): [number, number, number, number] => {
      const x1 = cy * X + sy * Z, z1 = -sy * X + cy * Z;
      const y2 = cp * Y - sp * z1, z2 = sp * Y + cp * z1;
      const f = D / Math.max(D - z2, 0.2 * D);
      return [w / 2 + k * f * x1, h * 0.5 - k * f * y2, z2, k * f]; // last: pixels per unit at this depth
    };
    const P = (p: number[]) => proj(p[0], floorY - p[1] - scale / 2, -p[2]);

    // floor grid under the feet
    g.strokeStyle = "rgba(128,128,128,0.22)";
    g.lineWidth = 1;
    const step = scale / 3;
    for (let i = -4; i <= 4; i++) {
      for (const [a, b] of [[[i * step, -4 * step], [i * step, 4 * step]], [[-4 * step, i * step], [4 * step, i * step]]]) {
        const p1 = proj(a[0], floor, a[1]), p2 = proj(b[0], floor, b[1]);
        g.beginPath(); g.moveTo(p1[0], p1[1]); g.lineTo(p2[0], p2[1]); g.stroke();
      }
    }
    if (!pose) {
      g.fillStyle = "#9AA0A6";
      g.font = "13px system-ui, sans-serif";
      g.textAlign = "center";
      g.fillText("No 3D pose at this frame", w / 2, h / 2);
      return;
    }
    // shadow on the floor: the figure flattened straight down — the clearest cue of where each limb is in depth
    g.strokeStyle = body ? "rgba(0,0,0,0.3)" : "rgba(0,0,0,0.35)";
    g.lineCap = "round";
    for (const [a, b] of data.bones) {
      const pa = pose[idx[a]], pb = pose[idx[b]];
      const sa = proj(pa[0], floor, -pa[2]), sb = proj(pb[0], floor, -pb[2]);
      g.lineWidth = body ? Math.max(4, (BODY_W[`${a}-${b}`] ?? 0.05) * scale * sa[3]) : 5;
      g.beginPath(); g.moveTo(sa[0], sa[1]); g.lineTo(sb[0], sb[1]); g.stroke();
    }
    if (body) {
      drawBody(g, pose, idx, P, scale, data);
      return;
    }
    // bones, far ones first; nearer ones brighter and thicker
    const segs = data.bones.map(([a, b]) => {
      const pa = P(pose[idx[a]]), pb = P(pose[idx[b]]);
      return { pa, pb, z: (pa[2] + pb[2]) / 2, role: boneRole(a, b, data.roles) };
    }).sort((s, t) => s.z - t.z);
    for (const s of segs) {
      const near = Math.max(0, Math.min(1, 0.5 + s.z / (2 * scale)));
      g.globalAlpha = 0.45 + 0.55 * near;
      g.strokeStyle = ROLE_COLOR[s.role];
      g.lineWidth = (s.role === "other" ? 2.5 : 4) * (0.7 + 0.6 * near);
      g.beginPath(); g.moveTo(s.pa[0], s.pa[1]); g.lineTo(s.pb[0], s.pb[1]); g.stroke();
      g.fillStyle = ROLE_COLOR[s.role];
      for (const q of [s.pa, s.pb]) { g.beginPath(); g.arc(q[0], q[1], g.lineWidth * 0.8, 0, Math.PI * 2); g.fill(); }
    }
    g.globalAlpha = 1;
    const head = P(pose[idx.head]);
    g.fillStyle = ROLE_COLOR.other;
    g.beginPath(); g.arc(head[0], head[1], 7, 0, Math.PI * 2); g.fill();
  }, [data, idx, nearest, yaw, pitch, scale, body]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    if (!spin) return;
    let raf = 0, last = performance.now();
    const step = (t: number) => { setYaw((y) => y + (t - last) * 0.0008); last = t; raf = requestAnimationFrame(step); };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [spin]);
  useEffect(() => {
    const c = canvas.current;
    if (!c || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(c);
    return () => ro.disconnect();
  }, [draw]);

  return (
    <div className="col" style={{ gap: 8 }}>
      <canvas ref={canvas} role="img"
        aria-label="3D skeleton of the bowler lifted from the video: front leg red, back leg blue, bowling arm green. Drag to rotate."
        style={{ width: "100%", height: 320, background: "var(--s2)", borderRadius: 8, touchAction: "none", cursor: "grab" }}
        onPointerDown={(e) => { setSpin(false); drag.current = { x: e.clientX, y: e.clientY, yaw, pitch }; e.currentTarget.setPointerCapture(e.pointerId); }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          setYaw(d.yaw + (e.clientX - d.x) * 0.01);
          setPitch(Math.max(-1.5, Math.min(1.5, d.pitch + (e.clientY - d.y) * 0.01)));
        }}
        onPointerUp={() => { drag.current = null; }}
        onPointerCancel={() => { drag.current = null; }} />
      <div className="row" style={{ gap: 6, flexWrap: "wrap", alignItems: "center" }}>
        {PRESETS.map((p) => (
          <button key={p.label} className="btn xs" style={{ padding: "4px 10px" }}
            onClick={() => {
              setSpin(false);
              setYaw(p.label === "Front" ? runYaw : p.label === "Side" ? runYaw - Math.PI / 2 : p.yaw);
              setPitch(p.pitch);
            }}>{p.label}</button>
        ))}
        <button className="btn xs" style={{ padding: "4px 10px" }} aria-pressed={body} onClick={() => setBody(!body)}>
          {body ? "Skeleton" : "Body"}
        </button>
        <button className="btn xs" style={{ padding: "4px 10px" }} aria-pressed={spin} onClick={() => setSpin(!spin)}>
          {spin ? "Stop" : "Spin"}
        </button>
        <span className="xs mu" style={{ marginLeft: "auto" }}>
          <span style={{ color: ROLE_COLOR.front }}>■</span> front leg <span style={{ color: ROLE_COLOR.back }}>■</span> back leg{" "}
          <span style={{ color: ROLE_COLOR.bowling }}>■</span> bowling arm
        </span>
      </div>
    </div>
  );
}
