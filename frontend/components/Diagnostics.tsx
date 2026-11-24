"use client";
import { useEffect, useRef, useState } from "react";
import { debugUrl, getDebugJson, getDebugPose3d, toApiError } from "@/lib/api";
import { type Pose3D, Skeleton3D } from "@/components/Skeleton3D";
import type { AnalysisResult } from "@/types/analysis";

/** Coach view: how the pipeline reached its result — the signals with every method's guess, the clip context with
 *  reasons, and the key frames (files from storage/debug/{id}/). */
export function Diagnostics({ result, mediaQ }: { result: AnalysisResult; mediaQ: string | null }) {
  const [open, setOpen] = useState(false);
  const [json, setJson] = useState<Record<string, unknown> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pose, setPose] = useState<Pose3D | null>(null);
  const [frame, setFrame] = useState(0);
  const [videoOk, setVideoOk] = useState(true);
  useEffect(() => {
    if (!open || !result.diagnostics) return;
    setJson(null);
    getDebugJson(result.id).then(setJson).catch((e) => setErr(toApiError(e).message));
    setPose(null);
    getDebugPose3d(result.id).then(setPose).catch(() => setPose(null)); // no 3D (PyTorch missing): viewer hidden
  }, [open, result.id, result.diagnostics, result.recomputed_at]);
  const v = result.recomputed_at;
  const vid = useRef<HTMLVideoElement>(null);
  const fps = pose?.fps || result.video.fps;
  // Follow the roles video frame by frame while it plays (timeupdate alone fires only a few times a second).
  useEffect(() => {
    const el = vid.current;
    if (!el) return;
    let raf = 0;
    const tick = () => { setFrame(Math.floor(el.currentTime * fps + 1e-3)); if (!el.paused) raf = requestAnimationFrame(tick); };
    const start = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); };
    el.addEventListener("play", start);
    el.addEventListener("seeked", tick);
    el.addEventListener("timeupdate", tick);
    return () => { cancelAnimationFrame(raf); el.removeEventListener("play", start); el.removeEventListener("seeked", tick); el.removeEventListener("timeupdate", tick); };
  }, [fps, open, videoOk, pose]);
  useEffect(() => { if (pose) setFrame(pose.events.ffc); }, [pose]);
  const goto = (f: number) => {
    setFrame(f);
    if (vid.current && videoOk) { vid.current.pause(); vid.current.currentTime = (f + 0.5) / fps; }
  };
  const nFrames = pose?.n ?? 0;
  return (
    <section className="card" style={{ padding: 0 }}>
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="spread"
        style={{ width: "100%", background: "none", border: 0, padding: "18px 20px" }}>
        <span><b>Diagnostics</b> <span className="mu small">· how the detection was made (coach view)</span></span>
        <span className="small mu">{open ? "Hide" : "Show"}</span>
      </button>
      {open && (
        <div className="col" style={{ padding: "0 20px 20px", gap: 14 }}>
          {!result.diagnostics && <p className="small mu">No diagnostics were written for this analysis. Recompute it (Check detection) to create them.</p>}
          {result.diagnostics && mediaQ && (
            <>
              <img src={debugUrl(result.id, "signals.png", mediaQ, v)} alt="Ankle speeds, wrist height and hip speed over time with the key moments and each detection method's guess"
                style={{ width: "100%", borderRadius: 8, background: "#fff" }} />
              <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                {(["bfc", "ffc", "release"] as const).map((k) => (
                  <figure key={k} style={{ margin: 0, flex: "1 1 180px" }}>
                    <img src={debugUrl(result.id, `${k}.jpg`, mediaQ, v)} alt={`${k} frame with the skeleton`} style={{ width: "100%", borderRadius: 8 }} />
                    <figcaption className="xs mu">{k.toUpperCase()} · F {result.events[k].frame} · {result.events[k].confidence}</figcaption>
                  </figure>
                ))}
              </div>
            </>
          )}
          {result.diagnostics && mediaQ && (
            <div className="col" style={{ gap: 8 }}>
              <b className="small">Roles and 3D pose</b>
              <p className="xs mu" style={{ margin: 0 }}>
                Left: the video with the skeleton coloured by role, and every frame where left/right labels were swapped
                back marked. Right: the same moment as a 3D skeleton lifted from the video (MotionBERT). Drag it to look
                from any side. 3D knee angles are not used in the report yet (see the confidence notes); the trunk and
                rotations are.
              </p>
              <div className="row" style={{ gap: 12, flexWrap: "wrap", alignItems: "flex-start" }}>
                {videoOk && (
                  <video ref={vid} src={debugUrl(result.id, "roles.mp4", mediaQ, v)} controls muted playsInline preload="metadata"
                    onError={() => setVideoOk(false)}
                    style={{ flex: "1 1 300px", minWidth: 0, maxHeight: 360, borderRadius: 8, background: "#000" }} />
                )}
                {pose && <div style={{ flex: "1 1 300px", minWidth: 0 }}><Skeleton3D data={pose} frame={frame} /></div>}
              </div>
              {!videoOk && <p className="xs mu" style={{ margin: 0 }}>The roles video isn&apos;t available for this analysis. Recompute it to create it.</p>}
              {!pose && <p className="xs mu" style={{ margin: 0 }}>No 3D pose for this analysis: install PyTorch (WINDOWS_COMMANDS.md → Perception upgrade) and recompute.</p>}
              {pose && (
                <div className="row" style={{ gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                  {(["bfc", "ffc", "release"] as const).map((k) => (
                    <button key={k} className="btn xs" style={{ padding: "4px 10px" }} onClick={() => goto(pose.events[k])}>
                      {k === "release" ? "Release" : k.toUpperCase()}
                    </button>
                  ))}
                  <input type="range" min={0} max={Math.max(0, nFrames - 1)} value={frame} aria-label="Frame"
                    onChange={(e) => goto(Number(e.target.value))} style={{ flex: "1 1 160px" }} />
                  <span className="xs mu mono">F {frame} · {(frame / fps).toFixed(2)} s</span>
                </div>
              )}
            </div>
          )}
          {result.gemini && (
            <div className="small">
              <b>Gemini video check</b>{" "}
              {result.gemini.used
                ? <span className="mu">{Object.entries(result.gemini.outcomes).filter(([, v]) => v !== "none")
                    .map(([k, v]) => `${k.replace("_", " ")}: ${v === "agree" ? "agrees" : v === "gemini" ? "Gemini's answer used" : v === "conflict" ? "disagrees (flagged)" : v}`).join(" · ") || "no usable hints"}</span>
                : <span className="mu">not used — {result.gemini.error}</span>}
            </div>
          )}
          {(result.stages?.length ?? 0) > 0 && (
            <div className="small">
              <b>Stages</b>
              <div className="row" style={{ gap: 6, flexWrap: "wrap", marginTop: 6 }}>
                {result.stages!.map((s, i) => (
                  <span key={`${s.name}${i}`} className="chip xs" title={s.detail || ""}
                    style={{ color: s.ok ? "var(--gtx)" : "var(--rtx)" }}>{s.ok ? "✓" : "✗"} {s.name}</span>
                ))}
              </div>
            </div>
          )}
          {err && <p className="small" style={{ color: "var(--rtx)" }}>{err}</p>}
          {json && (
            <details>
              <summary className="small">context.json</summary>
              <pre className="xs mono" style={{ whiteSpace: "pre-wrap", maxHeight: 420, overflow: "auto", background: "var(--s2)", padding: 12, borderRadius: 8 }}>
                {JSON.stringify(json, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
    </section>
  );
}
