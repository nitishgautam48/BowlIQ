"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { Logo, ThemeToggle } from "@/components/AppShell";
import { ErrorBox, SkeletonBlocks } from "@/components/Common";
import { getBenchmarks, toApiError, type ApiError } from "@/lib/api";
import { GLOSSARY } from "@/lib/format";
import type { BenchmarkRule } from "@/types/analysis";

const METHOD = [
  ["01", "Quality check", "Frame rate, resolution, length, rotation, camera motion and camera angle. Unreadable clips and clips under 1 s are stopped; everything else lowers confidence with a reason."],
  ["02", "Pose estimation", "MediaPipe Pose Landmarker (heavy) finds 33 landmarks on every frame; we keep 17. Short gaps are interpolated, then Savitzky–Golay smoothing."],
  ["03", "Event detection", "Back foot contact, front foot contact and release are found by rules on the ankle and wrist paths, then sanity-checked."],
  ["04", "Measure and rank", "Plain geometry at each event, judged against research rules or your personal baseline, ranked by deviation × weight × confidence."],
];
const LIMITS = [
  "One camera cannot measure rotation towards or away from it. Front-on rotation metrics are 2D projections of 3D rotations and are capped at Medium confidence.",
  "At 30 fps an event can be up to 33 ms off. 120 fps or more is recommended.",
  "Loose clothing, low light and motion blur reduce joint confidence.",
  "Body height is estimated from nose-to-ankle × 1.07, so metres and ball speed carry that scale error.",
  "Ball speed is measured from the ball itself and needs your height; it is not reported when the ball can't be followed for 4 frames.",
  "Targets are population ranges. Your coach may set different ones for you.",
];
const TYPE_TEXT: Record<string, string> = { band: "Target range", max: "Flag at or above", higher_better: "Higher is better (vs your baseline)", monitor: "Trend only" };

export default function About() {
  const [rules, setRules] = useState<Record<string, BenchmarkRule> | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const load = () => { setErr(null); getBenchmarks().then(setRules).catch((e) => setErr(toApiError(e))); };
  useEffect(load, []);
  return (
    <div>
      <header className="mk-top">
        <Logo />
        <nav className="row" style={{ gap: 8 }}><Link href="/new" className="btn primary sm">Start an analysis</Link><ThemeToggle /></nav>
      </header>
      <main className="mk col" style={{ gap: 48, padding: "32px 24px 72px" }}>
        <section>
          <h1 style={{ fontSize: "clamp(30px,5vw,48px)", marginBottom: 12 }}>How BowlIQ measures</h1>
          <p className="mu" style={{ maxWidth: 680, fontSize: 17 }}>AI does the perception (finding joints and key moments). Every measurement is plain geometry, and the report text only explains numbers it is given.</p>
        </section>
        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))", gap: 16 }}>
          {METHOD.map(([n, t, d]) => (
            <div key={n} className="card"><div className="mono g">{n}</div><h3 style={{ fontSize: 17, margin: "6px 0" }}>{t}</h3><p className="mu small">{d}</p></div>
          ))}
        </section>
        <section>
          <h2 style={{ fontSize: 28, marginBottom: 16 }}>Metrics, rules and sources</h2>
          {err && <ErrorBox error={err} onRetry={load} title="We couldn't load the metric definitions" />}
          {!rules && !err && <SkeletonBlocks blocks={[[12, 280]]} />}
          {rules && (
            <div className="card tablewrap" style={{ padding: 8 }}>
              <table className="t">
                <thead><tr><th>Metric</th><th>View</th><th>How it is measured</th><th>Rule</th><th>Source</th></tr></thead>
                <tbody>
                  {Object.entries(rules).map(([k, r]) => (
                    <tr key={k}>
                      <td><b>{r.name}</b><div className="fa xs mono">{k}</div></td>
                      <td className="small">{r.view}</td>
                      <td className="small mu">{r.method}</td>
                      <td className="small">{TYPE_TEXT[r.type]}{r.band ? `: ${r.band[0]}–${r.band[1]} ${r.unit}` : ""}{r.threshold != null ? `: ${r.threshold}${r.unit}` : ""}</td>
                      <td className="xs mu">{r.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 24 }}>
          <div>
            <h2 style={{ fontSize: 24, marginBottom: 12 }}>Glossary</h2>
            <div className="card col">
              {Object.values(GLOSSARY).map((g) => { const [a, b] = g.split(": "); return <p key={a} className="small"><b>{a}</b><span className="mu"> — {b}</span></p>; })}
            </div>
          </div>
          <div>
            <h2 style={{ fontSize: 24, marginBottom: 12 }}>Known limits</h2>
            <div className="card col">{LIMITS.map((l) => <p key={l} className="small mu">• {l}</p>)}</div>
          </div>
        </section>
      </main>
    </div>
  );
}
