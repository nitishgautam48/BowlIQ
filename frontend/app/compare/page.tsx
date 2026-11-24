"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Empty, ErrorBox, SkeletonBlocks, StatusIcon } from "@/components/Common";
import { Icon } from "@/components/Icons";
import { FramePane } from "@/components/FramePane";
import { compare, getDeliveries, getKeypoints, toApiError, videoUrl, type ApiError } from "@/lib/api";
import { fmtDate, fmtVal } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import { useMediaLinks } from "@/lib/useMediaLinks";
import type { AnalysisResult, CompareResponse, DeliverySummary, KeypointsResponse } from "@/types/analysis";

function Pane({ tag, res, relS, color }: { tag: string; res: AnalysisResult; relS: number; color: string }) {
  const [kp, setKp] = useState<KeypointsResponse | null>(null);
  const mediaQ = useMediaLinks(res.id);
  const v = res.video;
  const target = Math.max(0, Math.min(v.frame_count - 1, res.events.release.frame + Math.round(relS * v.fps)));
  useEffect(() => { getKeypoints(res.id).then(setKp).catch(() => setKp(null)); }, [res.id]);
  const win = useMemo(() => ({ bfc: res.events.bfc.frame, release: res.events.release.frame, fps: v.fps }), [res.events, v.fps]);
  return (
    <div className="card col">
      <div className="spread">
        <div className="row"><span className="chip" style={{ color, borderColor: color }}>{tag}</span><b className="small">{fmtDate(res.created_at, true)}</b></div>
        <span className="mono small">{res.ball_speed.value_kmh != null ? `${Math.round(res.ball_speed.value_kmh)} km/h` : "speed —"}</span>
      </div>
      <FramePane src={mediaQ ? videoUrl(res.id, mediaQ, true) : null} v={v} kp={kp} target={target} arm={res.bowling_arm} color={color} angleWindow={win} />
      <p className="small mu">{res.findings[0]?.title ?? "No fixes flagged"}</p>
    </div>
  );
}

function Picker({ onPick }: { onPick: (a: string, b: string) => void }) {
  const { settings, ready } = useSettings();
  const [rows, setRows] = useState<DeliverySummary[] | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  useEffect(() => { if (ready) getDeliveries(settings.bowler).then((r) => { setRows(r); setA(r[0]?.id ?? ""); setB(r[1]?.id ?? ""); }).catch((e) => setErr(toApiError(e))); }, [ready, settings.bowler]);
  if (err) return <ErrorBox error={err} title="We couldn't load your deliveries" />;
  if (!rows) return <SkeletonBlocks blocks={[[12, 120]]} />;
  if (rows.length < 2) return <Empty title="You need two deliveries to compare" body={`${settings.bowler} has ${rows.length} analysed ${rows.length === 1 ? "delivery" : "deliveries"}. Upload another to compare them side by side.`} />;
  const sel = (v: string, set: (s: string) => void, label: string) => (
    <label className="field" style={{ flex: 1, minWidth: 220 }}><span className="label">{label}</span>
      <select className="input" value={v} onChange={(e) => set(e.target.value)}>
        {rows.map((r) => <option key={r.id} value={r.id}>{fmtDate(r.created_at, true)}{r.ball_speed_kmh != null ? ` · ${Math.round(r.ball_speed_kmh)} km/h` : ""}</option>)}
      </select></label>
  );
  return (
    <div className="card row" style={{ alignItems: "flex-end" }}>
      {sel(a, setA, "Delivery A")}{sel(b, setB, "Delivery B")}
      <button className="btn primary" disabled={!a || !b || a === b} onClick={() => onPick(a, b)}>Compare</button>
    </div>
  );
}

function CompareInner() {
  const sp = useSearchParams();
  const router = useRouter();
  const a = sp.get("a"), b = sp.get("b");
  const [data, setData] = useState<CompareResponse | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [relMs, setRelMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.25);
  const load = useCallback(() => {
    if (!a || !b) return;
    setErr(null); setData(null);
    compare(a, b).then(setData).catch((e) => setErr(toApiError(e)));
  }, [a, b]);
  useEffect(load, [load]);
  // both clips aligned on their release, in real time (they can have different frame rates)
  const range = data ? {
    lo: -Math.floor(1000 * Math.min(data.a.events.release.frame / data.a.video.fps, data.b.events.release.frame / data.b.video.fps)),
    hi: Math.floor(1000 * Math.min((data.a.video.frame_count - 1 - data.a.events.release.frame) / data.a.video.fps,
                                    (data.b.video.frame_count - 1 - data.b.events.release.frame) / data.b.video.fps)),
  } : { lo: 0, hi: 0 };
  useEffect(() => {
    if (!playing || !data) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      // advance with the clock; panes skip frames they can't decode in time instead of freezing
      setRelMs((r) => (r + dt * speed > range.hi ? range.lo : r + dt * speed));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, data, speed, range.lo, range.hi]);
  const pick = (x: string, y: string) => router.push(`/compare?a=${x}&b=${y}`);
  return (
    <div className="col" style={{ gap: 16 }}>
      {(!a || !b) && <Picker onPick={pick} />}
      {a && b && err && <ErrorBox error={err} onRetry={load} title="We couldn't load these deliveries" />}
      {a && b && !data && !err && <SkeletonBlocks blocks={[[6, 300], [6, 300], [12, 300]]} />}
      {data && (
        <>
          <div className="grid">
            <div className="span-6"><Pane tag="A" res={data.a} relS={relMs / 1000} color="#2BD576" /></div>
            <div className="span-6"><Pane tag="B" res={data.b} relS={relMs / 1000} color="#38BDF8" /></div>
          </div>
          <div className="card col">
            <div className="spread">
              <div className="row">
                <button className="btn sm primary" onClick={() => setPlaying(!playing)}><Icon name={playing ? "pause" : "play"} size={16} /> {playing ? "Pause both" : "Play both"}</button>
                <button className="btn sm" onClick={() => { setPlaying(false); setRelMs(0); }}>Jump to release</button>
                <div className="seg" role="group" aria-label="Playback speed">
                  {[0.1, 0.25, 0.5, 1].map((sp) => <button key={sp} aria-pressed={speed === sp} onClick={() => setSpeed(sp)}>{sp}×</button>)}
                </div>
              </div>
              <span className="mono small">{Math.abs(relMs) < 1 ? "At release" : `Release ${relMs > 0 ? "+" : "−"}${Math.abs(Math.round(relMs))} ms`}</span>
            </div>
            <input type="range" min={range.lo} max={range.hi} step={5} value={Math.round(relMs)} onChange={(e) => { setPlaying(false); setRelMs(Number(e.target.value)); }} aria-label="Milliseconds relative to release" style={{ accentColor: "#2BD576" }} />
            <p className="xs fa">Both deliveries are aligned on their release moment in real time (so clips filmed at different frame rates stay in step). Playback runs at the selected fraction of real speed.</p>
          </div>
          <div className="card" style={{ padding: 8 }}>
            <div className="tablewrap">
              <table className="t">
                <thead><tr><th>Metric</th><th>A</th><th>B</th><th>A − B</th><th>Verdict for A</th></tr></thead>
                <tbody>
                  {data.diffs.map((d) => (
                    <tr key={d.key}>
                      <td>{d.name}</td>
                      <td className="mono small">{fmtVal(d.a, d.unit)}</td>
                      <td className="mono small">{fmtVal(d.b, d.unit)}</td>
                      <td className="mono small">{d.delta == null ? "—" : `${d.delta >= 0 ? "+" : "−"}${fmtVal(Math.abs(d.delta), d.unit)}`}</td>
                      <td className="small"><span className="row" style={{ gap: 6, flexWrap: "nowrap" }}><StatusIcon status={d.verdict} />{d.note}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <button className="btn" style={{ alignSelf: "flex-start" }} onClick={() => router.push("/compare")}>Choose other deliveries</button>
        </>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <AppShell title="Compare">
      <Suspense fallback={<SkeletonBlocks blocks={[[12, 120]]} />}><CompareInner /></Suspense>
    </AppShell>
  );
}
