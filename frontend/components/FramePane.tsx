"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { drawSkeleton, type AngleWindow } from "@/components/VideoPlayer";
import { frameToTime, timeToFrame } from "@/lib/frames";
import type { KeypointsResponse, VideoInfo } from "@/types/analysis";

/** A video locked to a target frame, with the skeleton drawn for the frame actually on screen.
 *  Seeks are coalesced (a new target waits for the previous seek), so playback can't queue up and freeze. */
export function FramePane({ src, v, kp, target, arm, color, angles = true, angleWindow, label, overlay, view = "side" }: {
  src: string | null; v: VideoInfo; kp: KeypointsResponse | null; target: number | null; arm: string; color: string;
  angles?: boolean; angleWindow?: AngleWindow; label?: string; overlay?: string | null; view?: "side" | "second";
}) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [shown, setShown] = useState<number | null>(null);
  const pending = useRef<number | null>(null);

  const draw = useCallback((f: number | null) => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    if (f == null || !kp) { ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); return; }
    drawSkeleton(ctx, kp, f, { skeleton: true, angles, arm, color, window: angleWindow, view });
  }, [kp, angles, arm, color, angleWindow, view]);

  const seek = useCallback((f: number) => {
    pending.current = f;
    const el = video.current;
    if (!el || el.readyState < 1 || el.seeking) return; // picked up in onSeeked / onLoadedData
    el.currentTime = frameToTime(f, v);
  }, [v]);

  useEffect(() => { if (target != null) seek(target); else draw(null); }, [target, seek, draw]);
  useEffect(() => { if (shown != null && target != null) draw(shown); }, [draw, shown, target]);

  const onSeeked = () => {
    const el = video.current;
    if (!el) return;
    const f = timeToFrame(el.currentTime, v);
    setShown(f);
    if (target != null) draw(f);
    if (pending.current != null && pending.current !== f) el.currentTime = frameToTime(pending.current, v);
  };
  return (
    <div style={{ position: "relative", aspectRatio: `${v.width} / ${v.height}`, width: `min(100%, calc(60vh * ${v.width} / ${v.height}))`, margin: "0 auto", background: "#000", borderRadius: 10, overflow: "hidden" }}>
      {src && <video ref={video} src={src} muted playsInline preload="auto" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: target == null ? 0.35 : 1 }}
        onLoadedData={() => seek(pending.current ?? target ?? 0)} onSeeked={onSeeked} />}
      <canvas ref={canvas} width={kp?.width || v.width} height={kp?.height || v.height} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} aria-hidden="true" />
      <span className="chip mono" style={{ position: "absolute", left: 8, top: 8, background: "var(--glass)", color: "var(--tx)" }}>
        {label ? `${label} · ` : ""}F {target == null ? "—" : String(shown ?? target).padStart(3, "0")}
      </span>
      {overlay && <span className="chip" style={{ position: "absolute", left: "50%", top: "50%", transform: "translate(-50%,-50%)", background: "var(--glass)", color: "var(--tx)", textAlign: "center" }}>{overlay}</span>}
    </div>
  );
}
