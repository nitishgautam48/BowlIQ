"use client";
import { useEffect, useState } from "react";
import { ConfPip, StatusIcon } from "@/components/Common";
import { Icon } from "@/components/Icons";
import { Gloss } from "@/components/Tooltip";
import { frameUrl } from "@/lib/api";
import { EVENT_TEXT, STATUS_TEXT, VIEW_LABEL, fmtNum, fmtVal, isSecondClip } from "@/lib/format";
import type { AnalysisResult, Finding, Metric } from "@/types/analysis";

const GLOSS_FOR: Record<string, string> = { shoulder_counter_rotation: "SCR", hip_shoulder_separation_bfc: "HSS", trunk_lateral_flexion_release: "LF" };

export function MeasureDialog({ result, metric, mediaQ, onClose, onJump }: { result: AnalysisResult; metric: Metric; mediaQ: string | null; onClose: () => void; onJump: (f: number, joint: string | null, view: string) => void }) {
  const [imgErr, setImgErr] = useState(false);
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  const sideView = !isSecondClip(metric.view);
  const canShow = sideView || !!result.front_video;
  const ev = metric.event ? EVENT_TEXT[metric.event] || metric.event : "";
  return (
    <div className="overlay" onClick={onClose} role="presentation">
      <div className="dialog" role="dialog" aria-modal="true" aria-label={`How we measured ${metric.name}`} onClick={(e) => e.stopPropagation()}>
        <div className="spread" style={{ marginBottom: 16 }}>
          <div>
            <div className="label">How we measured this</div>
            <h2 style={{ fontSize: 22 }}>{metric.name}</h2>
          </div>
          <button className="btn sm ghost" onClick={onClose} aria-label="Close"><Icon name="x" /></button>
        </div>
        <div className="grid">
          <div className="span-8">
            {metric.frame != null && canShow && !imgErr && mediaQ ? (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img src={frameUrl(result.id, metric.frame, mediaQ, metric.joint, !sideView, result.recomputed_at)} alt={`Frame ${metric.frame} with the skeleton and angles drawn`} onError={() => setImgErr(true)}
                style={{ width: "100%", borderRadius: 12, background: "#000", display: "block" }} />
            ) : (
              <div className="card mu small" style={{ minHeight: 200, display: "grid", placeItems: "center", textAlign: "center" }}>
                {metric.frame == null ? metric.reason || "Not measured" : "The frame image couldn't be loaded."}
              </div>
            )}
            {metric.frame != null && <p className="xs mu mono" style={{ marginTop: 8 }}>Frame {metric.frame} · {(metric.frame / (sideView ? result.video.fps : result.front_video?.fps || result.front_quality?.fps || result.video.fps)).toFixed(3)} s · {ev} · {VIEW_LABEL[metric.view] ?? metric.view}{sideView ? "" : " (second clip)"}</p>}
          </div>
          <div className="span-4 col" style={{ gap: 14 }}>
            <div>
              <div className="label">Value</div>
              <div className="mono" style={{ fontSize: 30, fontWeight: 600 }}>{fmtVal(metric.value, metric.unit)}</div>
              <div className="small mu">{metric.target_text ? `Target: ${metric.target_text}` : "Tracked as a trend (no target)"}</div>
              {metric.baseline != null && <div className="small mu">Your baseline: {fmtVal(metric.baseline, metric.unit)}</div>}
            </div>
            <div><div className="label">Status</div><div className="row small" style={{ gap: 6 }}><StatusIcon status={metric.status} />{STATUS_TEXT[metric.status]}</div>{metric.reason && <p className="small mu">{metric.reason}</p>}</div>
            <div><div className="label">Confidence</div><ConfPip level={metric.confidence} /><p className="small mu" style={{ marginTop: 4 }}>{metric.confidence_reason}</p></div>
            <div><div className="label">Method</div><p className="small">{metric.method}</p></div>
            <div><div className="label">View</div><p className="small">{metric.view === "3d" ? (metric.confidence_reason?.startsWith("Source: 3D lifted") ? "3D pose lifted from the video (MotionBERT). Coach view → Diagnostics shows it in 3D." : "Both clips combined into true 3D rotation (frame of the side-on clip)") : sideView ? "Side-on clip" : `Second clip, filmed ${metric.view === "behind" ? "from behind the bowler" : metric.view === "diagonal" ? "diagonally" : "front-on (behind the stumps, facing the bowler)"}`}</p></div>
            {metric.source && <div><div className="label">Research basis</div><p className="small mu">{metric.source}</p></div>}
            {metric.frame != null && canShow && <button className="btn primary" onClick={() => { onClose(); onJump(metric.frame!, metric.joint, metric.view); }}>{sideView ? "Show in video" : "Show in linked playback"}</button>}
          </div>
        </div>
      </div>
    </div>
  );
}

function BandBar({ f }: { f: Finding }) {
  const lo0 = f.target_min ?? (f.target_max != null ? Math.min(0, f.measured) : f.measured);
  const hi0 = f.target_max ?? Math.max(f.measured, lo0 + 1);
  const lo = Math.min(lo0, f.measured), hi = Math.max(hi0, f.measured);
  const pad = (hi - lo) * 0.3 || 1, A = lo - pad, B = hi + pad;
  const pc = (x: number) => `${((x - A) / (B - A)) * 100}%`;
  const bMin = f.target_min ?? A, bMax = f.target_max ?? B;
  return (
    <div style={{ position: "relative", height: 26, marginTop: 6 }} aria-label={`Measured ${fmtVal(f.measured, f.unit)}, target ${f.target_text}`} role="img">
      <div style={{ position: "absolute", top: 11, left: 0, right: 0, height: 4, borderRadius: 4, background: "var(--s3)" }} />
      <div style={{ position: "absolute", top: 9, left: pc(bMin), width: `calc(${pc(bMax)} - ${pc(bMin)})`, height: 8, borderRadius: 4, background: "rgba(43,213,118,.35)" }} />
      <div style={{ position: "absolute", top: 3, left: `calc(${pc(f.measured)} - 2px)`, width: 4, height: 20, borderRadius: 2, background: "var(--atx)" }} />
    </div>
  );
}

export function FindingCard({ f, coach, open, onToggle, onShow, onMeasure, vote, onVote, canShowSecond = false }: {
  f: Finding; coach: boolean; open: boolean; onToggle: () => void; onShow: () => void; onMeasure: () => void; vote: boolean | undefined; onVote: (v: boolean) => void;
  canShowSecond?: boolean;
}) {
  const val = fmtVal(f.measured, f.unit);
  return (
    <div className="card" style={{ padding: 0, borderColor: open ? "rgba(245,165,36,.5)" : undefined }}>
      <button onClick={onToggle} aria-expanded={open} aria-controls={`fix-${f.rank}`} style={{ width: "100%", background: "none", border: 0, padding: "18px 20px", textAlign: "left", display: "flex", gap: 14, alignItems: "center" }}>
        <span className="mono" style={{ fontSize: 22, color: "var(--fa)", width: 34 }}>0{f.rank}</span>
        <span style={{ flex: 1 }}>
          <span style={{ display: "block", fontWeight: 600, fontSize: 17 }}>{f.title}{f.kind === "injury" && <span className="chip a" style={{ marginLeft: 8 }}>Load pattern</span>}</span>
          <span className="small mu">{coach ? `${val} vs target ${f.target_text}` : f.cue || `Outside target (${f.target_text})`}</span>
        </span>
        <ConfPip level={f.confidence} reason={f.confidence_reason} label={false} tip={false} />
        <span style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform .2s" }}><Icon name="chev" /></span>
      </button>
      {open && (
        <div id={`fix-${f.rank}`} style={{ padding: "0 20px 20px 68px" }} className="col">
          <div>
            <div className="row" style={{ gap: 10 }}><span className="mono" style={{ fontSize: 26, fontWeight: 600 }}>{val}</span><span className="mu small">target {f.target_text}</span></div>
            {(f.target_min != null || f.target_max != null) && <BandBar f={f} />}
            <p className="xs mu" style={{ marginTop: 4 }}>Frame {f.frame ?? "—"}{isSecondClip(f.view) ? " of the second clip" : ""} · {VIEW_LABEL[f.view] ?? f.view} · {f.confidence} confidence: {f.confidence_reason}</p>
          </div>
          {f.why && <p className="small" style={{ order: coach ? 0 : 2 }}><b>Why it matters. </b><span className="mu">{f.why}</span></p>}
          {f.cue && <p style={{ fontSize: coach ? 15 : 18, fontWeight: 500, order: 1 }}>“{f.cue}”</p>}
          {f.drill && <p className="small" style={{ order: 3 }}><b>Drill. </b>{f.drill}</p>}
          {f.source && <p className="xs fa" style={{ order: 4 }}>Source: {f.source}</p>}
          <div className="spread" style={{ order: 5 }}>
            <div className="row" style={{ gap: 6 }}>
              {f.frame != null && (!isSecondClip(f.view) || canShowSecond) && <button className="btn sm" onClick={onShow}><Icon name="play" size={14} /> {!isSecondClip(f.view) ? `Show in video · frame ${f.frame}` : `Show in the ${VIEW_LABEL[f.view]?.toLowerCase() ?? "second"} clip · frame ${f.frame}`}</button>}
              <button className="btn sm ghost" onClick={onMeasure}><Icon name="ruler" size={14} /> How we measured this</button>
            </div>
            <div className="row" style={{ gap: 4 }}>
              <span className="xs mu">Helpful?</span>
              <button className="btn sm ghost" aria-pressed={vote === true} onClick={() => onVote(true)} style={vote === true ? { color: "var(--gtx)", background: "rgba(43,213,118,.12)" } : undefined} aria-label="Helpful"><Icon name="up" size={16} /></button>
              <button className="btn sm ghost" aria-pressed={vote === false} onClick={() => onVote(false)} style={vote === false ? { background: "var(--s3)" } : undefined} aria-label="Not helpful"><Icon name="down" size={16} /></button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

type SortKey = "default" | "name" | "value" | "status" | "confidence";

export function MetricsTable({ metrics, onMeasure }: { metrics: Metric[]; onMeasure: (m: Metric) => void }) {
  const [sort, setSort] = useState<SortKey>("default");
  const [dir, setDir] = useState(1);
  const rank = { High: 2, Medium: 1, Low: 0 } as const;
  const order = { outside: 0, in_range: 1, info: 2, not_measured: 3 } as const;
  const rows = [...metrics];
  const get: Record<Exclude<SortKey, "default">, (m: Metric) => number | string> = {
    name: (m) => m.name, value: (m) => m.value ?? -Infinity, status: (m) => order[m.status], confidence: (m) => (m.confidence ? rank[m.confidence] : -1),
  };
  if (sort !== "default") rows.sort((a, b) => { const x = get[sort](a), y = get[sort](b); return (x > y ? 1 : x < y ? -1 : 0) * dir; });
  const th = (k: SortKey, l: string) => (
    <th aria-sort={sort === k ? (dir > 0 ? "ascending" : "descending") : "none"}>
      <button onClick={() => { setDir(sort === k ? -dir : 1); setSort(k); }} style={{ color: sort === k ? "var(--tx)" : undefined }}>{l} {sort === k ? (dir > 0 ? "↑" : "↓") : ""}</button>
    </th>
  );
  return (
    <div className="tablewrap">
      <table className="t">
        <thead><tr>{th("name", "Metric")}{th("value", "Value")}<th>Target</th>{th("status", "Status")}<th>View</th>{th("confidence", "Confidence")}<th /></tr></thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.key} style={{ opacity: m.status === "not_measured" ? 0.7 : 1 }}>
              <td>{GLOSS_FOR[m.key] ? <Gloss k={GLOSS_FOR[m.key]}>{m.name}</Gloss> : m.name}</td>
              <td className="mono">{m.value == null ? "—" : <>{fmtNum(m.value, m.unit)}<span className="mu small">{m.unit === "°" ? "°" : ` ${m.unit}`}</span></>}</td>
              <td className="small mu">{m.target_text ?? "Trend only"}</td>
              <td className="small"><span className="row" style={{ gap: 6, flexWrap: "nowrap" }}><StatusIcon status={m.status} />{m.status === "not_measured" ? m.reason : m.status === "info" && m.reason ? m.reason : STATUS_TEXT[m.status]}</span></td>
              <td className="small">{VIEW_LABEL[m.view] ?? m.view}</td>
              <td><ConfPip level={m.confidence} reason={m.confidence_reason} /></td>
              <td>{m.status !== "not_measured" && <button className="btn sm ghost" onClick={() => onMeasure(m)} aria-label={`How we measured ${m.name}`}><Icon name="ruler" size={14} /></button>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function KineticChain({ result, coach, onPick }: { result: AnalysisResult; coach: boolean; onPick: (key: string | null) => void }) {
  return (
    <div className="col" style={{ gap: 0 }}>
      {result.kinetic_chain.map((c, i) => (
        <div key={c.segment}>
          <button onClick={() => onPick(c.metric)} className="row" aria-label={`${c.segment}: ${c.status}, ${c.metric_text}`}
            style={{ width: "100%", background: "var(--s2)", border: "1px solid transparent", borderRadius: 10, padding: "12px 14px", flexWrap: "nowrap", justifyContent: "space-between" }}>
            <span className="row" style={{ flexWrap: "nowrap" }}><StatusIcon status={c.status} /><span style={{ fontWeight: 500, whiteSpace: "nowrap" }}>{c.segment}</span></span>
            <span className={`small ${/\d/.test(c.metric_text) ? "mono" : ""}`} style={{ textAlign: "right", color: c.status === "warn" ? "var(--atx)" : c.status === "ok" ? "var(--gtx)" : "var(--mu)" }}>
              {coach || c.status === "unknown" ? c.metric_text : c.status === "ok" ? "On track" : "Needs work"}
            </span>
          </button>
          {i < result.kinetic_chain.length - 1 && <div style={{ width: 2, height: 10, background: "var(--line2)", marginLeft: 22 }} />}
        </div>
      ))}
    </div>
  );
}
