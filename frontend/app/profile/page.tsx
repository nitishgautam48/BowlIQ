"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Empty, ErrorBox, SkeletonBlocks } from "@/components/Common";
import { RepeatSummary } from "@/components/RepeatSummary";
import { TrendChart } from "@/components/TrendChart";
import { getBenchmarks, getHistory, toApiError, type ApiError } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { SAMPLE_BOWLER } from "@/lib/useSample";
import type { BenchmarkRule, BowlerHistory, HistoryPoint } from "@/types/analysis";

/** One series per metric, in the unit most deliveries used (e.g. run-up speed in m/s when a height was given). */
function series(pts: HistoryPoint[], key: string) {
  const counts = new Map<string, number>();
  pts.forEach((p) => { const u = p.metrics[key]?.unit; if (u && p.metrics[key]?.value != null) counts.set(u, (counts.get(u) ?? 0) + 1); });
  const unit = [...counts].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "";
  const skipped = pts.filter((p) => p.metrics[key] && p.metrics[key].unit !== unit && p.metrics[key].value != null).length;
  return {
    unit,
    skipped,
    points: pts.map((p) => ({ date: p.created_at, value: p.metrics[key]?.unit === unit ? p.metrics[key].value : null })),
  };
}

const CHARTS: [string, number][] = [
  ["runup_speed", 1], ["stride_length_pct_height", 1], ["front_knee_flexion_ffc_to_release", 0], ["front_knee_angle_release", 0],
  ["trunk_forward_flexion_release", 0], ["release_height_pct_height", 1], ["shoulder_counter_rotation", 0], ["trunk_lateral_flexion_release", 0],
];

export default function Profile() {
  const { settings, ready } = useSettings();
  const [hist, setHist] = useState<BowlerHistory | null>(null);
  const [rules, setRules] = useState<Record<string, BenchmarkRule> | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const load = useCallback(() => {
    if (!ready) return;
    setErr(null);
    setHist(null);
    Promise.all([getHistory(settings.bowler), getBenchmarks()]).then(([h, r]) => { setHist(h); setRules(r); }).catch((e) => setErr(toApiError(e)));
  }, [settings.bowler, ready]);
  useEffect(load, [load]);
  const pts = hist?.deliveries ?? [];
  return (
    <AppShell title="Bowler profile">
      <div className="col" style={{ gap: 16 }}>
        {err && <ErrorBox error={err} onRetry={load} title={`We couldn't load ${settings.bowler}'s profile`} />}
        {!hist && !err && <SkeletonBlocks blocks={[[12, 72], [12, 300], [6, 240], [6, 240]]} />}
        {hist && pts.length === 0 && (
          <Empty title={`No deliveries for ${settings.bowler} yet`} body="Trends appear after the first analysis. Your personal baseline unlocks after 10 deliveries.">
            <Link href="/new" className="btn primary">Start an analysis</Link>
          </Empty>
        )}
        {hist && pts.length > 0 && rules && (
          <>
            <div className="card spread">
              <div>
                <h2 style={{ fontSize: 24 }}>{hist.name}</h2>
                <p className="mu small">{pts.length} {pts.length === 1 ? "delivery" : "deliveries"} analysed</p>
              </div>
              <div style={{ minWidth: 260 }}>
                <div className="label">Personal baseline</div>
                {hist.name === SAMPLE_BOWLER ? <p className="small mu">Synthetic sample deliveries never build a baseline.</p> : hist.baseline_active ? <p className="small g">Active: run-up, trunk and front-knee metrics are judged against your best deliveries.</p> : (
                  <>
                    <div className="bar" style={{ margin: "6px 0" }}><div style={{ width: `${(pts.length / 10) * 100}%` }} /></div>
                    <p className="small mu">Unlocks after {hist.deliveries_until_baseline} more {hist.deliveries_until_baseline === 1 ? "delivery" : "deliveries"}.</p>
                  </>
                )}
              </div>
            </div>
            <RepeatSummary name={hist.name} />
            <p className="small mu">{hover != null && pts[hover] ? `Showing ${new Date(pts[hover].created_at).toLocaleDateString()} on every chart.` : "Hover a chart to see the same delivery across all metrics."}</p>
            <div className="grid">
              <div className="span-12">
                <TrendChart title="Ball speed" unit="km/h" dec={0} points={pts.map((p) => ({ date: p.created_at, value: p.ball_speed_kmh }))} hover={hover} setHover={setHover} height={220} />
              </div>
              {CHARTS.map(([k, dec]) => {
                const r = rules[k];
                if (!r) return null;
                const sr = series(pts, k);
                const unit = sr.unit || r.unit;
                return (
                  <div key={k} className="span-6">
                    <TrendChart title={`${r.name}${unit ? ` (${unit})` : ""}`} unit={unit} dec={dec} points={sr.points}
                      band={unit === r.unit ? r.band ?? null : null} hover={hover} setHover={setHover} />
                    {sr.skipped > 0 && <p className="xs fa" style={{ marginTop: 4 }}>{sr.skipped} {sr.skipped === 1 ? "delivery" : "deliveries"} measured in a different unit (height not given) are not plotted.</p>}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </AppShell>
  );
}
