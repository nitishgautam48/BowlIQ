"use client";
import { useEffect, useState } from "react";
import { ConfPip } from "@/components/Common";
import { personUrl, recompute, toApiError } from "@/lib/api";
import type { AnalysisResult, Detection, Overrides } from "@/types/analysis";

const VIEWS: { v: NonNullable<Overrides["view"]>; l: string }[] = [
  { v: "side-on", l: "Side-on" }, { v: "front-on", l: "Front-on (facing the bowler)" },
  { v: "behind", l: "From behind the bowler" }, { v: "oblique", l: "Diagonal / oblique" },
];
const VIEW_NAME: Record<string, string> = { "side-on": "side-on", "front-on": "front-on", behind: "from behind", oblique: "diagonal", diagonal: "diagonal" };
const DIR_NAME: Record<string, string> = { left_to_right: "left → right", right_to_left: "right → left" };
const MOMENTS = [
  { k: "bfc", f: "bfc_frame", label: "BFC", long: "Back foot contact", hint: "the back foot first touches the ground after the bound" },
  { k: "ffc", f: "ffc_frame", label: "FFC", long: "Front foot contact", hint: "the front foot (heel) first touches the ground" },
  { k: "release", f: "release_frame", label: "Release", long: "Release", hint: "the ball leaves the hand (arm about vertical)" },
] as const;

/** Shows what was detected (camera view, arm, direction, bowler, key moments) with confidence, and lets the user
 *  correct any of it. "Recompute" re-runs everything after the pose stage in a few seconds (no re-upload). */
export function CheckDetection({ result, frame, goto, mediaQ, onRecomputed }: {
  result: AnalysisResult; frame: number; goto: (f: number) => void; mediaQ: string | null;
  onRecomputed: (r: AnalysisResult) => void;
}) {
  const ctx = result.context;
  const lowCtx = ctx ? [ctx.view, ctx.bowling_arm, ctx.direction, ctx.bowler].some((d) => d.confidence === "Low" && d.source === "auto") : false;
  const lowEv = (["bfc", "ffc", "release"] as const).some((k) => result.events[k].confidence === "Low");
  const attention = !!result.needs_review || lowCtx || lowEv;
  const [open, setOpen] = useState(attention);
  const [ov, setOv] = useState<Overrides>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setOv({}); setErr(null); }, [result.recomputed_at]);
  const fps = result.video.fps;
  const cur = (k: "bfc" | "ffc" | "release") => (ov[`${k}_frame` as const] ?? result.events[k].frame);
  const order = !(cur("bfc") < cur("ffc") && cur("ffc") < cur("release")) ? "Back foot, front foot and release must be in that order." : null;
  const changed = Object.keys(ov).length > 0;
  const prior = result.overrides || {};

  const run = async (body: Overrides & { clear?: boolean }) => {
    setBusy(true);
    setErr(null);
    try {
      onRecomputed(await recompute(result.id, body));
    } catch (e) {
      setErr(toApiError(e).message);
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button className="btn sm ghost" style={{ alignSelf: "flex-start" }} onClick={() => setOpen(true)}>
        Check detection (camera view, arm, bowler, key moments)
      </button>
    );
  }
  const row = (label: string, d: Detection | undefined, shown: string, control: React.ReactNode) => (
    <div className="spread" style={{ gap: 8, alignItems: "flex-start" }}>
      <div style={{ minWidth: 0, flex: 1 }}>
        <div className="small row" style={{ gap: 8 }}><b>{label}</b> <span>{shown}</span>
          {d && <ConfPip level={d.source === "user" ? "High" : d.confidence} reason={d.reasons.join(" ")} label={false} />}
          {d?.source === "user" && <span className="chip xs">set by you</span>}</div>
        {d && <div className="xs mu">{d.reasons.join(" ")}</div>}
      </div>
      {control}
    </div>
  );
  return (
    <div className="card col" style={{ background: "var(--s2)", gap: 12 }} aria-label="Check detection">
      <div className="spread">
        <b className="small">Check detection</b>
        <button className="btn sm ghost" onClick={() => { setOpen(false); setOv({}); setErr(null); }}>Close</button>
      </div>
      {result.needs_review && (result.review_reasons?.length ?? 0) > 0 && (
        <div className="warnbox small"><b>Please check:</b>{result.review_reasons!.map((r) => <div key={r}>• {r}</div>)}</div>
      )}
      <p className="xs mu">Correct anything that's wrong, then press Recompute: the measurements, report, video and PDF are redone
        from the stored skeleton in a few seconds.</p>

      {ctx && row("Camera view", ctx.view, VIEW_NAME[String(ctx.view.value)] ?? String(ctx.view.value),
        <select value={ov.view ?? ""} onChange={(e) => setOv({ ...ov, view: (e.target.value || undefined) as Overrides["view"] })} aria-label="Camera view">
          <option value="">Keep</option>{VIEWS.map((x) => <option key={x.v} value={x.v}>{x.l}</option>)}
        </select>)}
      {ctx && row("Bowling arm", ctx.bowling_arm, `${ctx.bowling_arm.value}-arm`,
        <select value={ov.bowling_arm ?? ""} onChange={(e) => setOv({ ...ov, bowling_arm: (e.target.value || undefined) as Overrides["bowling_arm"] })} aria-label="Bowling arm">
          <option value="">Keep</option><option value="right">Right arm</option><option value="left">Left arm</option>
        </select>)}
      {ctx && row("Direction", ctx.direction, DIR_NAME[String(ctx.direction.value)] ?? String(ctx.direction.value),
        <select value={ov.direction ?? ""} onChange={(e) => setOv({ ...ov, direction: (e.target.value || undefined) as Overrides["direction"] })} aria-label="Run-up direction">
          <option value="">Keep</option><option value="left_to_right">Left → right</option><option value="right_to_left">Right → left</option>
        </select>)}

      <div className="col" style={{ gap: 6 }}>
        <b className="small">Key moments</b>
        <p className="xs mu">Step to the moment on the scrubber above (‹ › or arrow keys), then press its button.</p>
        {MOMENTS.map(({ k, f, label, long, hint }) => {
          const e = result.events[k];
          const methods = Object.entries(e.methods || {}).filter(([m]) => m !== "user");
          return (
            <div key={k} className="spread" style={{ flexWrap: "nowrap", gap: 8 }}>
              <div style={{ minWidth: 0 }}>
                <div className="small row" style={{ gap: 6 }}><b>{long}</b>
                  <button className="btn sm ghost" style={{ padding: "0 4px" }} onClick={() => goto(cur(k))}>F {cur(k)} ({(cur(k) / fps).toFixed(3)} s)</button>
                  <ConfPip level={e.confidence} reason={e.confidence_reason} label={false} />
                  {ov[f] !== undefined && <span className="xs mu">(was F {e.frame})</span>}</div>
                <div className="xs mu">{hint}{methods.length > 0 && ` · methods: ${methods.map(([m, v]) => `${m.replace("_", " ")} ${v ?? "–"}`).join(", ")}`}</div>
              </div>
              <button className="btn sm" onClick={() => setOv({ ...ov, [f]: frame })} aria-label={`Set ${long} to frame ${frame}`}>Set as {label}</button>
            </div>
          );
        })}
      </div>

      {ctx && ctx.candidates.length > 1 && (
        <div className="col" style={{ gap: 6 }}>
          <div className="small row" style={{ gap: 8 }}><b>Choose bowler</b>
            <ConfPip level={ctx.bowler.source === "user" ? "High" : ctx.bowler.confidence} reason={ctx.bowler.reasons.join(" ")} label={false} />
            <span className="xs mu">{ctx.bowler.reasons.join(" ")}</span></div>
          <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
            {ctx.candidates.map((c) => {
              const chosen = (ov.bowler_track_id ?? ctx.bowler.value) === c.id;
              return (
                <button key={c.id} onClick={() => setOv({ ...ov, bowler_track_id: c.id })} aria-pressed={chosen}
                  className="btn sm col" style={{ padding: 4, gap: 2, borderColor: chosen ? "#2BD576" : undefined, borderWidth: chosen ? 2 : 1 }}
                  aria-label={`Person ${c.id + 1}, run-up score ${c.score.toFixed(2)}`}>
                  {mediaQ && <img src={personUrl(result.id, c.id, mediaQ)} alt={`Person ${c.id + 1}`} style={{ height: 96, borderRadius: 6 }} />}
                  <span className="xs">Person {c.id + 1}{chosen ? " ✓" : ""}</span>
                  <span className="xs mu mono">{c.score.toFixed(2)}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="spread" style={{ gap: 8 }}>
        <label className="small row" style={{ gap: 6 }}>Height (cm)
          <input type="number" min={100} max={230} style={{ width: 80 }} value={ov.height_cm ?? result.bowler_height_cm ?? ""}
            onChange={(e) => setOv({ ...ov, height_cm: e.target.value ? Number(e.target.value) : undefined })} aria-label="Bowler height in centimetres" />
        </label>
      </div>
      {(order || err) && <p className="small" style={{ color: "var(--rtx)" }}>{order || err}</p>}
      <div className="row" style={{ gap: 6 }}>
        <button className="btn sm primary" disabled={!changed || !!order || busy} onClick={() => run(ov)}>
          {busy ? "Recomputing…" : "Recompute"}</button>
        {changed && <button className="btn sm ghost" disabled={busy} onClick={() => setOv({})}>Reset</button>}
        {Object.keys(prior).length > 0 && <button className="btn sm ghost" disabled={busy} onClick={() => run({ clear: true })}>
          Undo all my corrections</button>}
        {!changed && result.needs_review && <button className="btn sm ghost" disabled={busy}
          onClick={() => run({ bfc_frame: result.events.bfc.frame, ffc_frame: result.events.ffc.frame, release_frame: result.events.release.frame })}>
          Key moments are right: confirm</button>}
      </div>
    </div>
  );
}
