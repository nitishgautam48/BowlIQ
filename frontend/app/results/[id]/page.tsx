"use client";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ConfPip, ErrorBox, SkeletonBlocks } from "@/components/Common";
import { Icon } from "@/components/Icons";
import { FindingCard, KineticChain, MeasureDialog, MetricsTable } from "@/components/ResultParts";
import { useToast } from "@/components/Toast";
import { LinkedPlayback } from "@/components/LinkedPlayback";
import { VideoPlayer, type SeekRequest } from "@/components/VideoPlayer";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Diagnostics } from "@/components/Diagnostics";
import { deleteAnalysis, getAnalysis, getKeypoints, pdfUrl, reanalyze, sendFeedback, toApiError, videoUrl, type ApiError } from "@/lib/api";
import { useMediaLinks } from "@/lib/useMediaLinks";
import { VIEW_LABEL, fmtDate, fmtVal, isSecondClip } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import type { AnalysisResult, KeypointsResponse, Metric } from "@/types/analysis";

export default function Results() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const flash = useToast();
  const { settings } = useSettings();
  const [res, setRes] = useState<AnalysisResult | null>(null);
  const [kp, setKp] = useState<KeypointsResponse | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [view, setView] = useState<"bowler" | "coach">(settings.view);
  const [open, setOpen] = useState<number | null>(1);
  const [seek, setSeek] = useState<SeekRequest | null>(null);
  const [measure, setMeasure] = useState<Metric | null>(null);
  const [votes, setVotes] = useState<Record<string, boolean>>({});
  const [metricsOv, setMetricsOv] = useState<boolean | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [confirmDel, setConfirmDel] = useState(false);
  const mediaQ = useMediaLinks(id);
  const playerRef = useRef<HTMLDivElement>(null);
  useEffect(() => setView(settings.view), [settings.view]);

  const load = useCallback(() => {
    setErr(null);
    setRes(null);
    Promise.all([getAnalysis(id), getKeypoints(id)])
      .then(([r, k]) => { setRes(r); setKp(k); setVotes(r.feedback || {}); })
      .catch((e) => setErr(toApiError(e)));
  }, [id]);
  useEffect(load, [load]);
  const coach = view === "coach";
  const metricsOpen = metricsOv ?? coach;

  const linkedRef = useRef<HTMLDivElement>(null);
  const [linkSeek, setLinkSeek] = useState<{ frame: number; n: number } | null>(null);
  const show = (frame: number, joint: string | null, view = "side-on") => {
    if (isSecondClip(view)) { // measured on the second clip: jump there in Linked playback
      setLinkSeek({ frame, n: Date.now() });
      linkedRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    setSeek({ frame, joint, id: Date.now() });
    playerRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  const vote = async (rank: number, helpful: boolean) => {
    const prev = votes[String(rank)];
    setVotes({ ...votes, [String(rank)]: helpful });
    try {
      await sendFeedback(id, rank, helpful);
      flash("Thanks. Your feedback helps us rank fixes.");
    } catch (e) {
      setVotes((v) => { const n = { ...v }; if (prev === undefined) delete n[String(rank)]; else n[String(rank)] = prev; return n; });
      flash(toApiError(e).message);
    }
  };
  const remove = async () => {
    setConfirmDel(false);
    setDeleting(true);
    try {
      await deleteAnalysis(id);
      flash("Delivery deleted");
      router.push("/deliveries");
    } catch (e) {
      setDeleting(false);
      flash(toApiError(e).message);
    }
  };

  if (err) return <AppShell title="Report"><ErrorBox error={err} onRetry={err.status === 404 ? undefined : load} title={err.status === 404 ? "No report to show" : "We couldn't load this report"} />
    {err.status === 404 && <div style={{ marginTop: 16 }}><Link className="btn" href="/deliveries">My deliveries</Link></div>}</AppShell>;
  if (!res) return <AppShell title="Report"><SkeletonBlocks blocks={[[12, 48], [12, 200], [12, 64], [12, 64], [8, 420], [4, 420]]} /></AppShell>;

  const q = res.quality;
  const bs = res.ball_speed;
  const measured = res.metrics.filter((m) => m.status !== "not_measured").length;
  const needFront = res.metrics.filter((m) => m.status === "not_measured" && m.reason === "Add a front-on clip").length;
  const metricByKey = (k: string) => res.metrics.find((m) => m.key === k) || null;
  return (
    <AppShell title="Report" actions={
      <div className="seg" role="group" aria-label="Report view">
        {(["bowler", "coach"] as const).map((v) => <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>{v === "bowler" ? "Bowler" : "Coach"}</button>)}
      </div>}>
      <div className="col" style={{ gap: 16 }}>
        <div className="spread">
          <div>
            <div className="small mu">{res.bowler_name} · {fmtDate(res.created_at, true)} · {res.bowling_arm}-arm · {res.has_front ? "side-on + front-on" : "side-on"}</div>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <a className="btn sm" href={mediaQ ? pdfUrl(res.id, mediaQ) : undefined} aria-disabled={!mediaQ} target="_blank" rel="noreferrer"><Icon name="dl" size={16} /> Download report</a>
            <a className="btn sm" href={mediaQ ? videoUrl(res.id, mediaQ) : undefined} aria-disabled={!mediaQ} download={`bowliq_${res.id}.mp4`}><Icon name="film" size={16} /> Annotated video</a>
            <Link className="btn sm" href={`/compare?a=${res.id}`}><Icon name="cols" size={16} /> Compare</Link>
            <button className="btn sm danger" onClick={() => setConfirmDel(true)} disabled={deleting}><Icon name="trash" size={16} /> {deleting ? "Deleting…" : "Delete"}</button>
          </div>
        </div>

        {res.needs_review && (
          <div className="warnbox small"><b>Check the detection.</b> Some things couldn't be detected with confidence — see
            “Check detection” under the video.{(res.review_reasons?.length ?? 0) > 0 && <span className="mu"> ({res.review_reasons![0]}{res.review_reasons!.length > 1 ? ` +${res.review_reasons!.length - 1} more` : ""})</span>}</div>
        )}
        {res.warnings.length > 0 && <div className="warnbox small">{res.warnings.map((w) => <div key={w}>• {w}</div>)}</div>}
        {q.level === "Low" && <div className="warnbox small"><b>Low clip quality.</b> {q.reasons.join(" ")} Numbers below carry lower confidence.</div>}

        <div className="card grid" style={{ gap: 24 }}>
          <div className="span-8 col">
            <div className="label">Headline</div>
            <h2 style={{ fontSize: "clamp(22px,3vw,30px)" }}>{res.report.headline_insight}</h2>
            <p className="mu">{res.report.headline_sub}</p>
            {res.findings[0] && (
              <div><button className="btn sm" onClick={() => { setOpen(1); if (res.findings[0].frame != null) show(res.findings[0].frame, metricByKey(res.findings[0].metric)?.joint ?? null); }}>See fix 1 in the video</button></div>
            )}
          </div>
          <div className="span-4 col" style={{ gap: 16 }}>
            <div>
              <div className="label">Ball speed</div>
              {bs.value_kmh != null ? (
                <div className="row" style={{ alignItems: "baseline", gap: 8 }}>
                  <span className="mono" style={{ fontSize: 44, fontWeight: 600 }}>{Math.round(bs.value_kmh)}</span><span className="mu">km/h ± {bs.plus_minus?.toFixed(0)}</span>
                </div>
              ) : (
                <div><div style={{ fontSize: 18, fontWeight: 500 }}>Not measured</div><p className="small mu">{bs.reason}</p></div>
              )}
              {bs.value_kmh != null && <p className="xs fa">Tracked from the ball over {bs.frames_tracked} frames after release.</p>}
            </div>
            <div>
              <div className="label">Clip quality</div>
              <ConfPip level={q.level} reason={q.reasons.join(" ")} />
              <p className="xs mu" style={{ marginTop: 4 }}>{q.fps.toFixed(0)} fps · {q.width}×{q.height} · {VIEW_LABEL[q.view_detected] ?? q.view_detected}{res.front_quality ? ` · second clip: ${VIEW_LABEL[res.front_quality.view_detected] ?? res.front_quality.view_detected}` : ""} · body scale ±{Math.round(res.scale.rel_uncertainty * 100)}%</p>
            </div>
          </div>
        </div>

        {res.injury_flags.map((fl) => (
          <div key={fl.metric} className="warnbox">
            <div className="row" style={{ flexWrap: "nowrap", alignItems: "flex-start" }}>
              <Icon name="warn" color="var(--atx)" />
              <div className="small"><b>{fl.title}: {fmtVal(fl.measured, fl.unit)}</b> (flag at {fl.threshold}°, {fl.view}, {fl.confidence} confidence). {fl.text}.
                {fl.source && <div className="xs mu" style={{ marginTop: 4 }}>Source: {fl.source}</div>}</div>
            </div>
          </div>
        ))}

        <section className="col" aria-label="Top fixes">
          <h2 style={{ fontSize: 20 }}>{res.findings.length ? `Your ${res.findings.length === 1 ? "top fix" : `top ${res.findings.length} fixes`}` : "No fixes needed"}</h2>
          {res.findings.length === 0 && <div className="card mu small">No metric crossed a research threshold at the available confidence. {res.baseline.active ? "" : res.baseline.deliveries_until_baseline === 0 ? "Sample deliveries never build a personal baseline." : `Run-up, trunk and front-knee comparisons start once your personal baseline unlocks (${res.baseline.deliveries_until_baseline} more deliveries).`}</div>}
          {res.findings.map((f) => (
            <FindingCard key={f.rank} f={f} coach={coach} open={open === f.rank} onToggle={() => setOpen(open === f.rank ? null : f.rank)}
              onShow={() => f.frame != null && show(f.frame, metricByKey(f.metric)?.joint ?? null, f.view)} canShowSecond={!!res.front_video}
              onMeasure={() => setMeasure(metricByKey(f.metric))} vote={votes[String(f.rank)]} onVote={(v) => vote(f.rank, v)} />
          ))}
        </section>

        <div className="grid">
          <div className="card span-8" ref={playerRef}>
            <VideoPlayer result={res} keypoints={kp} mediaQ={mediaQ} seek={seek} defaultAngles={coach}
              onRecomputed={(r) => {
                setRes(r);
                getKeypoints(id).then(setKp).catch(() => {});
                flash(r.needs_review ? "Recomputed. Some things still need checking." : "Recomputed with your corrections.");
              }}
              onReanalyze={async (m) => {
                try { const j = await reanalyze(res.id, m); router.push(`/processing/${j.job_id}`); return null; }
                catch (e) { return toApiError(e).message; }
              }} />
          </div>
          <div className="span-4 col">
            <div className="card">
              <div className="label" style={{ marginBottom: 12 }}>Kinetic chain</div>
              <KineticChain result={res} coach={coach} onPick={(k) => { const m = k ? metricByKey(k) : null; if (m) setMeasure(m); }} />
            </div>
            {res.strengths.length > 0 && (
              <div className="card">
                <div className="label" style={{ marginBottom: 10 }}>Strengths</div>
                <div className="col" style={{ gap: 10 }}>
                  {res.strengths.map((s) => <div key={s.metric}><div className="row" style={{ gap: 6 }}><Icon name="ok" color="var(--gtx)" /><b className="small">{s.title}</b></div><p className="small mu">{s.detail}</p></div>)}
                </div>
              </div>
            )}
            <div className="card">
              <div className="label" style={{ marginBottom: 10 }}>Key moments</div>
              {(["bfc", "ffc", "release"] as const).map((k) => {
                const e = res.events[k];
                return (
                  <button key={k} className="spread" onClick={() => show(e.frame, null)} style={{ width: "100%", background: "none", border: 0, padding: "6px 0", flexWrap: "nowrap" }}>
                    <span className="small">{k === "bfc" ? "Back foot contact" : k === "ffc" ? "Front foot contact" : "Release"}</span>
                    <span className="row" style={{ gap: 8, flexWrap: "nowrap" }}><span className="mono small">F {e.frame} · {e.time_s.toFixed(3)} s</span><ConfPip level={e.confidence} reason={e.confidence_reason} label={false} /></span>
                  </button>
                );
              })}
              <p className="xs fa" style={{ marginTop: 6 }}>Baseline: {res.baseline.active ? `active (${res.baseline.based_on})` : res.baseline.deliveries_until_baseline === 0 ? "not used for samples" : `unlocks after ${res.baseline.deliveries_until_baseline} more ${res.baseline.deliveries_until_baseline === 1 ? "delivery" : "deliveries"}`}</p>
            </div>
          </div>
        </div>

        {res.front_video && <div ref={linkedRef}><LinkedPlayback result={res} mediaQ={mediaQ} seek={linkSeek} /></div>}

        <section className="card" style={{ padding: 0 }}>
          <button onClick={() => setMetricsOv(!metricsOpen)} aria-expanded={metricsOpen}
            className="spread" style={{ width: "100%", background: "none", border: 0, padding: "18px 20px" }}>
            <span><b>All metrics</b> <span className="mu small">· {measured} measured{needFront ? ` · ${needFront} need a front-on clip` : ""}</span></span>
            <span style={{ transform: metricsOpen ? "rotate(180deg)" : "none" }}><Icon name="chev" /></span>
          </button>
          {metricsOpen && <div style={{ padding: "0 12px 12px" }}><MetricsTable metrics={res.metrics} onMeasure={setMeasure} /></div>}
        </section>

        {coach && <Diagnostics result={res} mediaQ={mediaQ} />}

        {res.report.strengths_text && res.strengths.length > 0 && <p className="small mu">{res.report.strengths_text}</p>}
        <div className="card tight">
          <div className="label" style={{ marginBottom: 6 }}>Limits of this report</div>
          <p className="small mu">{res.report.limits_note}</p>
          <p className="xs fa" style={{ marginTop: 6 }}>Report text: {res.report.generated_by === "gemini" ? "written by Gemini from the measured numbers only (validated)" : "built-in template"}.</p>
        </div>
      </div>
      {measure && <MeasureDialog result={res} metric={measure} mediaQ={mediaQ} onClose={() => setMeasure(null)} onJump={show} />}
      {confirmDel && <ConfirmDialog title="Delete this delivery?" danger confirmLabel="Delete"
        body={`This removes the delivery from ${res.bowler_name}'s profile, including its video and report files. This can't be undone.`}
        onConfirm={remove} onCancel={() => setConfirmDel(false)} />}
    </AppShell>
  );
}
