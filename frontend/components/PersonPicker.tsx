"use client";
import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/Icons";

/** Tap the bowler and/or shade people to ignore. Coordinates are fractions (0–1) of the video frame; tap time is
 *  seconds into the uploaded clip. */
export type Zone = [number, number, number, number];
export interface Guidance { tap: [number, number, number] | null; zones: Zone[] }
export const NO_GUIDANCE: Guidance = { tap: null, zones: [] };
export const MAX_ZONES = 8;

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export function PersonPicker({ url, width, height, range, value, onChange, label }: {
  url: string; width: number; height: number; range: [number, number]; value: Guidance;
  onChange: (g: Guidance) => void; label: string;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<"tap" | "zone">("tap");
  const [t, setT] = useState(() => (value.tap ? value.tap[0] : (range[0] + range[1]) / 2));
  const [draft, setDraft] = useState<Zone | null>(null);
  const [lo, hi] = range;
  // keep the scrubber inside the selected part of the clip
  useEffect(() => { setT((cur) => Math.min(hi, Math.max(lo, cur))); }, [lo, hi]);
  useEffect(() => { if (video.current) video.current.currentTime = Math.max(0, t); }, [t]);

  const at = (e: React.PointerEvent): [number, number] => {
    const r = box.current!.getBoundingClientRect();
    return [clamp01((e.clientX - r.left) / r.width), clamp01((e.clientY - r.top) / r.height)];
  };
  const down = (e: React.PointerEvent) => {
    e.preventDefault();
    const [x, y] = at(e);
    if (mode === "tap") { onChange({ ...value, tap: [Math.round(t * 1000) / 1000, x, y] }); return; }
    if (value.zones.length >= MAX_ZONES) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    setDraft([x, y, x, y]);
  };
  const move = (e: React.PointerEvent) => {
    if (!draft) return;
    const [x, y] = at(e);
    setDraft([draft[0], draft[1], x, y]);
  };
  const up = () => {
    if (!draft) return;
    const [x0, y0, x1, y1] = draft;
    const z: Zone = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
    setDraft(null);
    if (z[2] - z[0] < 0.02 || z[3] - z[1] < 0.02) return; // a click, not a drag
    if ((z[2] - z[0]) * (z[3] - z[1]) > 0.95) return; // would hide everyone
    onChange({ ...value, zones: [...value.zones, z] });
  };
  const aspect = (width || 16) / (height || 9);
  const rect = (z: Zone, key: string | number, dashed = false) => (
    <div key={key} style={{ position: "absolute", left: `${Math.min(z[0], z[2]) * 100}%`, top: `${Math.min(z[1], z[3]) * 100}%`,
      width: `${Math.abs(z[2] - z[0]) * 100}%`, height: `${Math.abs(z[3] - z[1]) * 100}%`, background: "rgba(255,77,77,.28)",
      border: `2px ${dashed ? "dashed" : "solid"} #FF4D4D`, borderRadius: 4, pointerEvents: "none" }} />
  );
  const tapVisible = value.tap && Math.abs(value.tap[0] - t) < 0.35;
  return (
    <div className="col" style={{ gap: 12 }}>
      <div className="row" role="radiogroup" aria-label={`${label} tool`}>
        <button className={`btn sm ${mode === "tap" ? "primary" : ""}`} role="radio" aria-checked={mode === "tap"} onClick={() => setMode("tap")}>
          <Icon name="user" size={16} /> Tap the bowler</button>
        <button className={`btn sm ${mode === "zone" ? "primary" : ""}`} role="radio" aria-checked={mode === "zone"} onClick={() => setMode("zone")}>
          <Icon name="x" size={16} /> Shade people to ignore</button>
        <span className="small mu">{mode === "tap" ? "Click on the bowler's body." : `Drag a box around umpires, fielders or spectators (${value.zones.length}/${MAX_ZONES}).`}</span>
      </div>
      <div ref={box} data-testid={`${label}-picker`} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={() => setDraft(null)}
        style={{ position: "relative", width: `min(100%, ${Math.round(420 * aspect)}px)`, aspectRatio: `${aspect}`, margin: "0 auto",
          background: "#000", borderRadius: 12, overflow: "hidden", cursor: "crosshair", touchAction: "none", userSelect: "none" }}>
        <video ref={video} src={url} muted playsInline preload="auto" style={{ width: "100%", height: "100%", display: "block", pointerEvents: "none" }}
          aria-label={`${label} frame at ${t.toFixed(2)} s`} />
        {value.zones.map((z, i) => rect(z, i))}
        {draft && rect(draft, "draft", true)}
        {value.tap && (
          <div aria-hidden style={{ position: "absolute", left: `${value.tap[1] * 100}%`, top: `${value.tap[2] * 100}%`, width: 28, height: 28,
            transform: "translate(-50%,-50%)", borderRadius: 99, border: "3px solid #2BD576", boxShadow: "0 0 0 3px rgba(0,0,0,.45)",
            opacity: tapVisible ? 1 : 0.45, pointerEvents: "none" }} />
        )}
        <span className="chip mono" style={{ position: "absolute", left: 12, top: 12, background: "var(--glass)", color: "var(--tx)", pointerEvents: "none" }}>{t.toFixed(2)} s</span>
      </div>
      <input type="range" min={lo} max={hi} step={0.01} value={t} onChange={(e) => setT(Number(e.target.value))} aria-label={`${label} frame time`} />
      <div className="spread small">
        <span className={value.tap ? "g" : "mu"}>
          {value.tap ? `Bowler picked at ${value.tap[0].toFixed(2)} s.` : "Optional: without a tap, the person running in is chosen automatically."}
        </span>
        <div className="row">
          {value.tap && <button className="btn sm ghost" onClick={() => onChange({ ...value, tap: null })}>Clear tap</button>}
          {value.zones.length > 0 && <button className="btn sm ghost" onClick={() => onChange({ ...value, zones: value.zones.slice(0, -1) })}>Undo zone</button>}
          {value.zones.length > 0 && <button className="btn sm ghost" onClick={() => onChange({ ...value, zones: [] })}>Clear zones</button>}
        </div>
      </div>
    </div>
  );
}

export function guidanceFields(g: Guidance): { tap: string | null; ignore: string | null } {
  const f = (v: number) => v.toFixed(4);
  return {
    tap: g.tap ? `${g.tap[0].toFixed(3)},${f(g.tap[1])},${f(g.tap[2])}` : null,
    ignore: g.zones.length ? g.zones.map((z) => z.map(f).join(",")).join(";") : null,
  };
}
