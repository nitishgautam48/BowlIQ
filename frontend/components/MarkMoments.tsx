"use client";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/Icons";
import { getJobLinks, jobVideoUrl, markJob, toApiError } from "@/lib/api";

export const SLOW_OPTIONS: [number, string][] = [
  [0, "Not sure (detect automatically)"], [1, "Real time (not slow motion)"], [2, "×2 slow motion"], [4, "×4 (phone slow-mo 120 fps)"], [8, "×8 (phone slow-mo 240 fps)"],
  [10, "×10"], [16, "×16 (TV replay)"], [20, "×20 (TV replay)"], [25, "×25 (TV super slow-mo)"], [32, "×32"],
];

const MOMENTS: { k: "bfc" | "ffc" | "release"; label: string; hint: string }[] = [
  { k: "bfc", label: "Back foot contact", hint: "the back foot first touches the ground after the bound" },
  { k: "ffc", label: "Front foot contact", hint: "the front foot (heel) first touches the ground" },
  { k: "release", label: "Release", hint: "the ball leaves the hand (arm about vertical)" },
];

/** When automatic detection fails, finish the analysis by marking the three key moments on the processed clip. */
export function MarkMoments({ jobId, clip, slowSuggested, angleDoubt = false }: {
  jobId: string; clip: { fps: number; frame_count: number; width: number; height: number }; slowSuggested: boolean;
  angleDoubt?: boolean;
}) {
  const router = useRouter();
  const video = useRef<HTMLVideoElement>(null);
  const [q, setQ] = useState<string | null>(null);
  const [f, setF] = useState(0);
  const [m, setM] = useState<{ bfc: number | null; ffc: number | null; release: number | null }>({ bfc: null, ffc: null, release: null });
  const [slow, setSlow] = useState(slowSuggested ? 16 : 1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const last = clip.frame_count - 1;
  useEffect(() => { getJobLinks(jobId).then((l) => setQ(l.query)).catch((e) => setErr(toApiError(e).message)); }, [jobId]);
  const go = (n: number) => {
    const k = Math.max(0, Math.min(last, n));
    setF(k);
    if (video.current) video.current.currentTime = (k + 0.5) / clip.fps;
  };
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return;
      if (e.key === "ArrowRight") { e.preventDefault(); go(f + (e.shiftKey ? 10 : 1)); }
      if (e.key === "ArrowLeft") { e.preventDefault(); go(f - (e.shiftKey ? 10 : 1)); }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
  const all = m.bfc != null && m.ffc != null && m.release != null;
  const order = all && !(m.bfc! < m.ffc! && m.ffc! < m.release!) ? "Back foot, front foot and release must be in that order." : null;
  const submit = async () => {
    if (!all || order) return;
    setBusy(true); setErr(null);
    try {
      const j = await markJob(jobId, { bfc: m.bfc!, ffc: m.ffc!, release: m.release!, slow_motion: slow > 1 ? slow : null });
      router.push(`/processing/${j.job_id}`);
    } catch (e) { setErr(toApiError(e).message); setBusy(false); }
  };
  const aspect = clip.width / Math.max(1, clip.height);
  return (
    <section className="card col" aria-label="Mark the key moments yourself" style={{ gap: 12 }}>
      <h2 style={{ fontSize: 18 }}>Finish it by marking the key moments yourself</h2>
      <p className="small mu">Step through the clip (‹ › buttons, or the arrow keys; Shift+arrow jumps 10 frames), stop on each moment and
        press “Set”. Everything else (angles, stride, report) is then worked out automatically.</p>
      {angleDoubt && <p className="small" style={{ color: "var(--atx)" }}>Only do this if the clip really is filmed from the side: from in
        front or behind, stride and side-on angles can't be measured correctly even with the right key moments.</p>}
      <div style={{ position: "relative", width: `min(100%, ${Math.round(420 * aspect)}px)`, aspectRatio: `${aspect}`, margin: "0 auto", background: "#000", borderRadius: 10, overflow: "hidden" }}>
        {q && <video ref={video} src={jobVideoUrl(jobId, q)} muted playsInline preload="auto" style={{ width: "100%", height: "100%", display: "block" }}
          onLoadedData={() => go(0)} aria-label={`Clip frame ${f}`} />}
        <span className="chip mono" style={{ position: "absolute", left: 8, top: 8, background: "var(--glass)", color: "var(--tx)" }}>F {f} / {last}</span>
      </div>
      <input type="range" min={0} max={last} value={f} onChange={(e) => go(Number(e.target.value))} aria-label="Scrub frames" style={{ accentColor: "#2BD576" }} />
      <div className="row" style={{ gap: 6 }}>
        <button className="btn sm" onClick={() => go(f - 10)} aria-label="Back 10 frames">−10</button>
        <button className="btn sm" onClick={() => go(f - 1)} aria-label="Previous frame"><Icon name="prev" size={16} /></button>
        <button className="btn sm" onClick={() => go(f + 1)} aria-label="Next frame"><Icon name="next" size={16} /></button>
        <button className="btn sm" onClick={() => go(f + 10)} aria-label="Forward 10 frames">+10</button>
      </div>
      {MOMENTS.map(({ k, label, hint }) => (
        <div key={k} className="spread" style={{ flexWrap: "nowrap", gap: 8 }}>
          <div style={{ minWidth: 0 }}>
            <div className="small"><b>{label}</b>{m[k] != null && <> · <button className="btn sm ghost" style={{ padding: "0 4px" }} onClick={() => go(m[k]!)}>F {m[k]}</button></>}</div>
            <div className="xs mu">{hint}</div>
          </div>
          <button className={`btn sm ${m[k] == null ? "primary" : ""}`} onClick={() => setM({ ...m, [k]: f })} aria-label={`Set ${label} to frame ${f}`}>Set to F {f}</button>
        </div>
      ))}
      <label className="field">
        <span className="label">Is this clip slow motion?</span>
        <select className="input" value={slow} onChange={(e) => setSlow(Number(e.target.value))} aria-label="Slow-motion speed">
          {SLOW_OPTIONS.filter(([v]) => v > 0).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <span className="hint">Angles don't depend on this; timings and speeds do.{slowSuggested ? " This clip was detected as slow motion." : ""}</span>
      </label>
      {(order || err) && <p className="small" style={{ color: "var(--rtx)" }}>{order || err}</p>}
      <button className="btn primary" style={{ alignSelf: "flex-start" }} disabled={!all || !!order || busy} onClick={submit}>
        {busy ? "Starting…" : "Analyse with these moments"}</button>
    </section>
  );
}
