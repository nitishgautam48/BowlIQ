"use client";
import { useEffect, useMemo, useState } from "react";
import { FramePane } from "@/components/FramePane";
import { Icon } from "@/components/Icons";
import { frontVideoUrl, getKeypoints, setSync, toApiError, videoUrl } from "@/lib/api";
import { VIEW_LABEL } from "@/lib/format";
import type { AnalysisResult, KeypointsResponse, SyncInfo } from "@/types/analysis";

const METHOD: Record<SyncInfo["method"], string> = {
  audio: "Synced on sound",
  motion: "Synced on body movement",
  release: "Synced on the release moment",
  none: "Not synced automatically",
};

/** Phase 2: the side-on and front-on clips played side by side, frame-locked through the sync offset
 *  (front_time = side_time + offset), with the skeleton on each. The offset can be nudged and saved. */
export function LinkedPlayback({ result, mediaQ, seek }: { result: AnalysisResult; mediaQ: string | null; seek?: { frame: number; n: number } | null }) {
  const sv = result.video;
  const fv = result.front_video!;
  const [sync, setSyncState] = useState<SyncInfo | null>(result.sync ?? null);
  const auto = sync?.offset_s ?? 0;
  const saved = sync?.manual_offset_s ?? auto;
  const [offset, setOffset] = useState(saved);
  const [sideKp, setSideKp] = useState<KeypointsResponse | null>(null);
  const [frontKp, setFrontKp] = useState<KeypointsResponse | null>(null);
  const [tMs, setTMs] = useState(Math.round((result.events.release.frame / sv.fps) * 1000));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.25);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { getKeypoints(result.id).then(setSideKp).catch(() => setSideKp(null)); }, [result.id]);
  useEffect(() => { getKeypoints(result.id, "front").then(setFrontKp).catch(() => setFrontKp(null)); }, [result.id]);
  useEffect(() => { setOffset(saved); }, [saved]);

  const sideEnd = ((sv.frame_count - 1) / sv.fps) * 1000;
  // with a sync, play only the stretch both phones recorded (side-clip time, ms); otherwise the whole side clip
  const synced = sync?.manual_offset_s != null || (sync?.offset_s != null && sync.method !== "none");
  const ovLo = Math.max(0, -offset * 1000), ovHi = Math.min(sideEnd, ((fv.frame_count - 1) / fv.fps - offset) * 1000);
  const [lo, hi] = synced && ovHi - ovLo > 200 ? [Math.ceil(ovLo), Math.floor(ovHi)] : [0, Math.floor(sideEnd)];
  // "show me" for a measurement made on the second clip: go to that frame of the second clip
  useEffect(() => {
    if (!seek) return;
    setPlaying(false);
    setTMs(Math.max(0, Math.min(sideEnd, (seek.frame / fv.fps - offset) * 1000)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seek?.n]);
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      setTMs((t) => (t + dt * speed > hi || t < lo ? lo : t + dt * speed));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, lo, hi]);

  const sideF = Math.max(0, Math.min(sv.frame_count - 1, Math.round((tMs / 1000) * sv.fps)));
  const frontT = sideF / sv.fps + offset;
  const frontRaw = Math.round(frontT * fv.fps);
  const frontF = frontRaw >= 0 && frontRaw < fv.frame_count ? frontRaw : null;
  const frontMsg = frontF != null ? null
    : !synced ? "Not lined up yet: use the nudge buttons below"
    : frontRaw < 0 ? "The second phone wasn't recording yet" : "The second phone had stopped recording";
  const step = (n: number) => { setPlaying(false); setTMs(Math.round(((Math.max(0, Math.min(sv.frame_count - 1, sideF + n))) / sv.fps) * 1000)); };
  const nudge = (frames: number) => setOffset((o) => Math.round((o + frames / fv.fps) * 10000) / 10000);
  const dirty = Math.abs(offset - saved) > 1e-4;
  const save = async (value: number | null) => {
    setSaving(true); setMsg(null);
    try {
      const s = await setSync(result.id, value);
      setSyncState(s);
      setMsg(value == null ? "Back to the automatic alignment." : "Alignment saved.");
    } catch (e) { setMsg(toApiError(e).message); }
    finally { setSaving(false); }
  };
  const sideWin = useMemo(() => ({ bfc: result.events.bfc.frame, release: result.events.release.frame, fps: sv.fps }), [result.events, sv.fps]);
  const fe = result.front_events;
  const jumps = (["bfc", "ffc", "release"] as const).map((k) => [k, result.events[k].frame] as const);
  const offsetMs = Math.round(offset * 1000);
  return (
    <section className="card col" aria-label="Linked playback" style={{ gap: 14 }}>
      <div className="spread">
        <div>
          <h2 style={{ fontSize: 18 }}>Linked playback</h2>
          <p className="small mu">Both clips frame-locked: the same instant of the delivery in each view.</p>
        </div>
        {sync && (
          <span className={`chip ${sync.manual_offset_s != null ? "" : sync.confidence === "High" ? "g" : sync.confidence === "Medium" ? "a" : "r"}`}>
            {sync.manual_offset_s != null ? "Aligned by hand" : `${METHOD[sync.method]} · ${sync.confidence}`}
          </span>
        )}
      </div>
      <div className="grid">
        <div className="span-6 col" style={{ gap: 6 }}>
          <FramePane src={mediaQ ? videoUrl(result.id, mediaQ, true) : null} v={sv} kp={sideKp} target={sideF} arm={result.bowling_arm}
            color="#2BD576" angleWindow={sideWin} label="Side-on" />
        </div>
        <div className="span-6 col" style={{ gap: 6 }}>
          <FramePane src={mediaQ ? frontVideoUrl(result.id, mediaQ) : null} v={fv} kp={frontKp} target={frontF} arm={result.bowling_arm}
            color="#38BDF8" angles view="second" label={VIEW_LABEL[result.front_quality?.view_detected ?? ""] ?? "Second clip"} overlay={frontMsg} />
        </div>
      </div>
      <div className="row" style={{ gap: 8 }}>
        <button className="btn sm primary" onClick={() => setPlaying(!playing)}><Icon name={playing ? "pause" : "play"} size={16} /> {playing ? "Pause" : "Play both"}</button>
        <button className="btn sm" onClick={() => step(-1)} aria-label="Previous frame"><Icon name="prev" size={16} /></button>
        <button className="btn sm" onClick={() => step(1)} aria-label="Next frame"><Icon name="next" size={16} /></button>
        {jumps.map(([k, f]) => <button key={k} className="btn sm" onClick={() => { setPlaying(false); setTMs(Math.round((f / sv.fps) * 1000)); }}>{k === "bfc" ? "Back foot" : k === "ffc" ? "Front foot" : "Release"}</button>)}
        <div className="seg" role="group" aria-label="Playback speed">
          {[0.1, 0.25, 0.5, 1].map((sp) => <button key={sp} aria-pressed={speed === sp} onClick={() => setSpeed(sp)}>{sp}×</button>)}
        </div>
      </div>
      <input type="range" min={lo} max={hi} step={1} value={Math.round(Math.min(hi, Math.max(lo, tMs)))} onChange={(e) => { setPlaying(false); setTMs(Number(e.target.value)); }}
        aria-label="Time in the side-on clip" style={{ accentColor: "#2BD576" }} />
      <p className="xs fa">Second clip: <span style={{ color: "#38BDF8" }}>bend</span> = side bend of the trunk from vertical,
        <span style={{ color: "#F5A524" }}> arm</span> = bowling arm from vertical (shown while the arm is up),
        <span style={{ color: "#F472B6" }}> sep</span> = angle between the shoulder and hip lines (hidden when they overlap end-on). Flat 2D
        angles; the report's 3D values use both clips.</p>
      <div className="spread small">
        <span className="mono">Side F {sideF} · {(sideF / sv.fps).toFixed(3)} s ↔ Front {frontF == null ? "—" : `F ${frontF} · ${frontT.toFixed(3)} s`}</span>
        {fe?.release && <span className="mu">Front-clip release: F {fe.release.frame}</span>}
      </div>
      <div className="card col" style={{ background: "var(--s2)", gap: 8 }}>
        <div className="spread">
          <b className="small">Alignment</b>
          <span className="mono small" aria-live="polite">front = side {offsetMs >= 0 ? "+" : "−"} {Math.abs(offsetMs)} ms</span>
        </div>
        {sync && <p className="small mu">{sync.detail}</p>}
        <div className="row" style={{ gap: 6 }}>
          <span className="small mu">If the views don't match, nudge the front-on clip:</span>
          <button className="btn sm" onClick={() => nudge(-5)}>−5 frames</button>
          <button className="btn sm" onClick={() => nudge(-1)}>−1 frame</button>
          <button className="btn sm" onClick={() => nudge(1)}>+1 frame</button>
          <button className="btn sm" onClick={() => nudge(5)}>+5 frames</button>
        </div>
        <div className="row" style={{ gap: 6 }}>
          <button className="btn sm primary" disabled={!dirty || saving || !sync} onClick={() => save(offset)}>Save alignment</button>
          {sync?.manual_offset_s != null && <button className="btn sm ghost" disabled={saving} onClick={() => save(null)}>Use automatic</button>}
          {dirty && <button className="btn sm ghost" onClick={() => setOffset(saved)}>Undo changes</button>}
          {msg && <span className="small mu" role="status">{msg}</span>}
        </div>
        <p className="xs fa">Tip: pause on the frame where the front foot lands in the side-on view, then nudge until it lands in the front-on view too.
          For automatic sync next time, clap once loudly where both phones can hear it, about the same distance from each.</p>
      </div>
    </section>
  );
}
