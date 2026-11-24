"use client";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ErrorBox } from "@/components/Common";
import { BowlerFigure } from "@/components/Figure";
import { MarkMoments } from "@/components/MarkMoments";
import { getJob, toApiError, type ApiError } from "@/lib/api";
import { useToast } from "@/components/Toast";
import type { JobState, Stage } from "@/types/analysis";

const STAGES: [Stage, string][] = [
  ["Checking quality", "Frame rate, resolution, camera motion, view"],
  ["Detecting skeleton", "17 joints on every frame"],
  ["Finding key moments", "Back foot contact, front foot contact, release"],
  ["Measuring", "Angles, stride, speeds and ball tracking"],
  ["Writing report", "Ranking fixes, annotated video and PDF"],
];

export default function Processing() {
  const { jobId } = useParams<{ jobId: string }>();
  const router = useRouter();
  const flash = useToast();
  const [job, setJob] = useState<JobState | null>(null);
  const [netErr, setNetErr] = useState<ApiError | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const t0 = useRef(Date.now());
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    let delay = 1000;
    const poll = async () => {
      try {
        const j = await getJob(jobId);
        if (!alive) return;
        setJob(j);
        setNetErr(null);
        delay = 1000;
        if (j.status === "done" && j.result) {
          flash(`Saved to ${j.result.bowler_name}'s profile`);
          router.replace(`/results/${j.result.id}`);
          return;
        }
        if (j.status === "failed") return;
      } catch (e) {
        if (!alive) return;
        const err = toApiError(e);
        setNetErr(err);
        if (err.status === 404 || err.status === 401) return;
        delay = Math.min(10_000, delay * 2); // back off while the server is unreachable
      }
      timer = setTimeout(poll, delay);
    };
    poll();
    const tick = setInterval(() => setElapsed(Math.round((Date.now() - t0.current) / 1000)), 1000);
    return () => { alive = false; clearTimeout(timer); clearInterval(tick); };
  }, [jobId, router, flash]);

  const failed = job?.status === "failed";
  const act = job ? (job.stage === "Done" ? 5 : Math.max(0, STAGES.findIndex(([s]) => s === job.stage))) : 0;
  const pct = job?.status === "done" ? 100 : job?.progress ?? 0;
  const mm = Math.floor(elapsed / 60), ss = String(elapsed % 60).padStart(2, "0");
  return (
    <AppShell title="New analysis">
      <div className="grid" style={{ maxWidth: 980 }}>
        <div className="card span-6" style={{ display: "grid", placeItems: "center", minHeight: 320 }}>
          <BowlerFigure size={280} />
        </div>
        <div className="card span-6 col" style={{ gap: 18 }}>
          <div>
            <h2 style={{ fontSize: 22 }}>{netErr?.status === 404 ? "Job not found" : failed ? "Analysis stopped" : job?.status === "queued" ? "Waiting in the queue" : "Analyzing your delivery"}</h2>
            <p className="mu small">{mm}:{ss} elapsed · usually 30–90 s on CPU, depending on clip length</p>
          </div>
          <div className="col" style={{ gap: 6 }}>
            <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}><div style={{ width: `${pct}%`, background: failed ? "var(--rtx)" : undefined }} /></div>
            <span className="mono small">{pct}%</span>
          </div>
          <ol style={{ listStyle: "none", margin: 0, padding: 0 }} className="col">
            {STAGES.map(([s, sub], i) => {
              const done = i < act, running = i === act && !failed && job?.status === "processing";
              const stop = failed && i === act;
              return (
                <li key={s} className="row" style={{ flexWrap: "nowrap", alignItems: "flex-start" }}>
                  <span style={{ width: 22, height: 22, borderRadius: 99, flexShrink: 0, marginTop: 1, display: "grid", placeItems: "center", fontSize: 12, color: "#04140A",
                    border: `1.5px solid ${stop ? "var(--rtx)" : i <= act ? "#2BD576" : "var(--line2)"}`, background: done ? "#2BD576" : "transparent",
                    animation: running ? "biqPulse 1.2s ease-out infinite" : "none" }}>{done ? "✓" : ""}</span>
                  <div className="col" style={{ gap: 0, flex: 1 }}>
                    <span style={{ fontWeight: running ? 600 : 500, color: i <= act ? "var(--tx)" : "var(--mu)" }}>{s}</span>
                    <span className="xs mu">{sub}</span>
                  </div>
                  <span className="xs" style={{ color: stop ? "var(--rtx)" : "var(--mu)" }}>{done ? "Done" : running ? "Running" : stop ? "Stopped" : ""}</span>
                </li>
              );
            })}
          </ol>
          {failed && job?.error && (
            <>
              <ErrorBox error={job.error} title={`We couldn't finish: ${job.error.message}`} />
              {job.can_mark && job.clip && <MarkMoments jobId={job.job_id} clip={job.clip} slowSuggested={job.error.code === "SLOW_MOTION_CLIP"}
                angleDoubt={job.error.code === "WRONG_CAMERA_ANGLE"} />}
              <div className="row"><Link href="/new" className="btn primary">Try another clip</Link><Link href="/deliveries" className="btn">My deliveries</Link></div>
            </>
          )}
          {netErr && <ErrorBox error={netErr} title={netErr.status === 404 ? "This job doesn't exist" : "Lost contact with the server; retrying…"} />}
          {netErr?.status === 404 && <div className="row"><Link href="/new" className="btn primary">Start a new analysis</Link></div>}
        </div>
      </div>
    </AppShell>
  );
}
