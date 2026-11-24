"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Icon } from "@/components/Icons";
import { getSummary, toApiError } from "@/lib/api";
import { VIEW_LABEL, fmtVal } from "@/lib/format";
import type { BowlerSummaryOut, RepeatIssue } from "@/types/analysis";

const PATTERN: Record<RepeatIssue["pattern"], { label: string; cls: string }> = {
  consistent: { label: "Consistent", cls: "r" },
  occasional: { label: "Occasional", cls: "a" },
  "one-off": { label: "One-off", cls: "" },
};

/** Phase 5: what keeps happening across the bowler's recent deliveries, not just in one ball. */
export function RepeatSummary({ name }: { name: string }) {
  const [last, setLast] = useState(10);
  const [data, setData] = useState<BowlerSummaryOut | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setErr(null); getSummary(name, last).then(setData).catch((e) => setErr(toApiError(e).message)); }, [name, last]);
  if (err) return <div className="card small mu">{err}</div>;
  if (!data) return null;
  return (
    <section className="card col" aria-label="Across your recent deliveries" style={{ gap: 14 }}>
      <div className="spread">
        <div>
          <h2 style={{ fontSize: 18 }}>Across your last {data.deliveries} {data.deliveries === 1 ? "delivery" : "deliveries"}</h2>
          <p className="small mu">What keeps happening, not what happened once. Only medium- and high-confidence measurements count.</p>
        </div>
        <div className="seg" role="group" aria-label="Number of deliveries">
          {[5, 10, 20].map((n) => <button key={n} aria-pressed={last === n} onClick={() => setLast(n)}>Last {n}</button>)}
        </div>
      </div>
      <p style={{ fontSize: 16 }}>{data.summary}</p>
      {data.ball_speed && <p className="small mu">Ball speed: typically {data.ball_speed.median} km/h, best {data.ball_speed.best} km/h ({data.ball_speed.measured} measured).</p>}
      {data.issues.length > 0 && (
        <div className="col" style={{ gap: 10 }}>
          {data.issues.map((i) => (
            <div key={i.metric} className="card col" style={{ background: "var(--s2)", gap: 6 }}>
              <div className="spread">
                <div className="row" style={{ gap: 8 }}>
                  <span className={`chip ${PATTERN[i.pattern].cls}`}>{PATTERN[i.pattern].label}</span>
                  <b>{i.title}</b>
                  {i.kind === "injury" && <span className="chip a">Load pattern</span>}
                </div>
                <span className="mono small" aria-label={`${i.count} of ${i.measured} deliveries`}>{i.count} of {i.measured}</span>
              </div>
              <div className="bar" aria-hidden="true"><div style={{ width: `${Math.round(i.share * 100)}%`, background: i.pattern === "consistent" ? "var(--rtx)" : undefined }} /></div>
              <p className="small mu">Typically {fmtVal(i.typical, i.unit)}{i.target_text ? ` (target ${i.target_text})` : ""}.
                {i.uncertain ? ` ${i.uncertain} more low-confidence ${i.uncertain === 1 ? "measurement wasn't" : "measurements weren't"} counted.` : ""}</p>
              {i.pattern !== "one-off" && i.drill && <p className="small"><b>Drill. </b>{i.drill}</p>}
              {i.pattern !== "one-off" && i.cue && <p className="small">“{i.cue}”</p>}
              <Link className="btn sm ghost" style={{ alignSelf: "flex-start" }} href={`/results/${i.example.id}`}>
                <Icon name="play" size={14} /> See the clearest example ({fmtVal(i.example.value, i.unit)}{i.example.view ? `, ${VIEW_LABEL[i.example.view] ?? i.example.view}` : ""})
              </Link>
            </div>
          ))}
        </div>
      )}
      {data.strengths.length > 0 && (
        <div className="col" style={{ gap: 4 }}>
          <div className="label">Consistently good</div>
          {data.strengths.map((s) => <p key={s.metric} className="small"><Icon name="ok" color="var(--gtx)" /> {s.text}</p>)}
        </div>
      )}
      {data.trends.length > 0 && (
        <details>
          <summary className="small" style={{ cursor: "pointer" }}>Personal trends ({data.trends.length} measurements without a research target)</summary>
          <div className="tablewrap" style={{ marginTop: 8 }}>
            <table className="t">
              <thead><tr><th>Measurement</th><th>Typical</th><th>Spread</th><th>Older → newer</th></tr></thead>
              <tbody>
                {data.trends.map((t) => (
                  <tr key={t.metric}>
                    <td className="small">{t.name}</td>
                    <td className="mono small">{fmtVal(t.typical, t.unit)}</td>
                    <td className="mono small">±{fmtVal(t.spread, t.unit)}</td>
                    <td className="small">{t.direction === "steady" ? "Steady" : `${t.direction === "up" ? "Up" : "Down"} ${fmtVal(Math.abs(t.change), t.unit)}`} <span className="mu">({t.measured})</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </section>
  );
}
