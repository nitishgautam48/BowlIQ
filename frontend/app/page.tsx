"use client";
import Link from "next/link";
import { Logo, ThemeToggle } from "@/components/AppShell";
import { BowlerFigure } from "@/components/Figure";
import { ErrorBox } from "@/components/Common";
import { useSampleAnalysis } from "@/lib/useSample";

const STEPS = [
  ["01", "Upload a side-on clip", "One delivery, filmed from about 10 m at hip height. Add a front-on clip for alignment."],
  ["02", "Select the delivery", "Trim to the last run-up strides through follow-through."],
  ["03", "We track 17 joints", "On every frame, then find back foot contact, front foot contact and release."],
  ["04", "You get up to 3 ranked fixes", "Each with a coaching cue, a drill, the frame it was measured on and the research behind it."],
];
const FILM = [
  ["Position", "Side-on, square to the popping crease, about 10 m away"],
  ["Height", "Hip height, about 1 m off the ground"],
  ["Frame rate", "60 fps minimum, 120 fps or more preferred"],
  ["Framing", "Whole body in view from the last run-up strides to follow-through"],
  ["Camera", "On a tripod. No zoom or panning."],
];
const TRUST = [
  ["Measured from 2D video", "Angles come from one camera, not a motion-capture lab. Every number carries a High, Medium or Low confidence rating and the reason for it."],
  ["Check the working", "Every measurement links to the frame it came from, with the angle drawn on your video."],
  ["Not a diagnosis", "Load patterns are prompts to talk to a coach or physio. We never diagnose injuries."],
];

export default function Home() {
  const sample = useSampleAnalysis();
  return (
    <div>
      <header className="mk-top">
        <Logo />
        <nav className="row" style={{ gap: 8 }}>
          <Link href="/about" className="btn ghost sm">How it works</Link>
          <Link href="/deliveries" className="btn ghost sm">My deliveries</Link>
          <ThemeToggle />
        </nav>
      </header>
      <section className="mk" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 32, alignItems: "center", padding: "48px 24px 72px" }}>
        <div className="col" style={{ gap: 20 }}>
          <span className="chip g" style={{ alignSelf: "flex-start" }}>Fast-bowling biomechanics</span>
          <h1 style={{ fontSize: "clamp(34px,6vw,60px)", letterSpacing: "-0.02em" }}>Find the pace your action is leaving behind.</h1>
          <p className="mu" style={{ fontSize: 18, maxWidth: 520 }}>
            Upload one delivery. BowlIQ tracks your skeleton frame by frame, measures stride, front-leg brace, trunk and release, and ranks the fixes most likely to add pace or reduce load.
          </p>
          <div className="row">
            <Link href="/new" className="btn primary lg">Start an analysis</Link>
            <button className="btn lg" onClick={sample.run} disabled={sample.busy}>
              {sample.busy ? "Uploading sample…" : "Analyze a sample clip"}
            </button>
          </div>
          {sample.error && <ErrorBox error={sample.error} />}
        </div>
        <div style={{ display: "flex", justifyContent: "center" }}><BowlerFigure size={380} /></div>
      </section>
      <section className="mk" style={{ paddingBottom: 72 }}>
        <h2 style={{ fontSize: 32, marginBottom: 24 }}>How it works</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))", gap: 16 }}>
          {STEPS.map(([n, t, d]) => (
            <div key={n} className="card"><div className="mono g" style={{ marginBottom: 8 }}>{n}</div><h3 style={{ fontSize: 17, marginBottom: 6 }}>{t}</h3><p className="mu small">{d}</p></div>
          ))}
        </div>
      </section>
      <section className="mk" style={{ paddingBottom: 72, display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 24 }}>
        <div>
          <h2 style={{ fontSize: 28, marginBottom: 16 }}>Film it right</h2>
          <div className="card" style={{ padding: 0 }}>
            {FILM.map(([k, v], i) => (
              <div key={k} className="row" style={{ padding: "14px 20px", borderTop: i ? "1px solid var(--line)" : "none", flexWrap: "nowrap" }}>
                <span className="mu small" style={{ width: 110, flexShrink: 0 }}>{k}</span><span className="small">{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h2 style={{ fontSize: 28, marginBottom: 16 }}>Honest numbers</h2>
          <div className="col">
            {TRUST.map(([t, d]) => (
              <div key={t} className="card tight"><h3 style={{ fontSize: 16, marginBottom: 4 }}>{t}</h3><p className="mu small">{d}</p></div>
            ))}
          </div>
        </div>
      </section>
      <footer className="mk" style={{ padding: "24px", borderTop: "1px solid var(--line)" }}>
        <p className="fa small">BowlIQ is a personal, non-commercial project. Runs locally on your computer.</p>
      </footer>
    </div>
  );
}
