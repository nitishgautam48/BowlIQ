"use client";
import { useState } from "react";
import { shortDate } from "@/lib/format";

export interface TrendPoint { date: string; value: number | null }

export function TrendChart({ title, unit, dec, points, band, baseline, height = 180, hover, setHover }: {
  title: string; unit: string; dec: number; points: TrendPoint[]; band?: [number, number] | null; baseline?: number | null;
  height?: number; hover: number | null; setHover: (i: number | null) => void;
}) {
  const [table, setTable] = useState(false);
  const W = 640, H = height, L = 48, R = 16, T = 20, B = 26;
  const vals = points.map((p) => p.value).filter((v): v is number => v != null);
  const fm = (v: number) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(dec);
  const u = unit === "°" ? "°" : ` ${unit}`;
  const header = (
    <div className="spread" style={{ marginBottom: 8 }}>
      <div><b>{title}</b>{vals.length > 1 && <span className="small mu"> · {fm(vals[vals.length - 1] - vals[0]).replace(/^(?!−)/, "+")}{u} since first</span>}</div>
      <button className="btn sm ghost" aria-pressed={table} onClick={() => setTable(!table)}>{table ? "View as chart" : "View as table"}</button>
    </div>
  );
  if (vals.length === 0) return <div className="card">{header}<p className="small mu">Not measured in any delivery yet.</p></div>;
  let lo = Math.min(...vals, ...(band || []), ...(baseline != null ? [baseline] : [])), hi = Math.max(...vals, ...(band || []), ...(baseline != null ? [baseline] : []));
  if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
  const n = points.length;
  const x = (i: number) => (n === 1 ? (L + W - R) / 2 : L + (i * (W - L - R)) / (n - 1));
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  let d = "";
  points.forEach((p, i) => { if (p.value != null) d += `${d === "" ? "M" : "L"}${x(i).toFixed(1)} ${y(p.value).toFixed(1)} `; });
  const ticks = [lo + pad, (lo + hi) / 2, hi - pad];
  const hv = hover != null && points[hover] ? points[hover] : null;
  return (
    <div className="card">
      {header}
      {table ? (
        <div className="tablewrap" style={{ maxHeight: 240, overflowY: "auto" }}>
          <table className="t"><thead><tr><th>Date</th><th>Value</th><th>Target</th></tr></thead>
            <tbody>{points.map((p, i) => <tr key={i}><td className="small">{shortDate(p.date)}</td><td className="mono small">{p.value == null ? "—" : fm(p.value) + u}</td>
              <td className="small mu">{band && p.value != null ? (p.value >= band[0] && p.value <= band[1] ? "In target" : "Outside") : "—"}</td></tr>)}</tbody></table>
        </div>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label={`${title} over ${n} deliveries`} onMouseLeave={() => setHover(null)}>
          {band && <><rect x={L} width={W - L - R} y={y(band[1])} height={y(band[0]) - y(band[1])} fill="rgba(43,213,118,0.10)" />
            <text x={L + 6} y={y(band[1]) + 14} style={{ fill: "var(--gtx)", fontSize: 11 }}>target</text></>}
          {baseline != null && <><line x1={L} x2={W - R} y1={y(baseline)} y2={y(baseline)} stroke="var(--ctx)" strokeDasharray="4 4" />
            <text x={W - R} y={y(baseline) - 4} textAnchor="end" style={{ fill: "var(--ctx)", fontSize: 11 }}>baseline</text></>}
          {ticks.map((t, i) => <g key={i}><line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--line)" />
            <text x={L - 8} y={y(t) + 4} textAnchor="end" style={{ fill: "var(--mu)", fontSize: 11 }}>{fm(t)}</text></g>)}
          <path d={d} fill="none" stroke="#2BD576" strokeWidth="2.2" />
          {points.map((p, i) => p.value != null && <circle key={i} cx={x(i)} cy={y(p.value)} r={i === hover ? 5 : 3} fill="#2BD576" />)}
          {[0, n - 1].filter((v, i, a) => a.indexOf(v) === i).map((i) => <text key={i} x={x(i)} y={H - 6} textAnchor="middle" style={{ fill: "var(--mu)", fontSize: 11 }}>{shortDate(points[i].date)}</text>)}
          {points.map((_, i) => <rect key={i} x={x(i) - (W - L - R) / Math.max(1, n - 1) / 2} y={0} width={(W - L - R) / Math.max(1, n - 1)} height={H} fill="transparent" onMouseEnter={() => setHover(i)} />)}
          {hv && hover != null && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B} stroke="var(--line2)" />
              <rect x={Math.min(x(hover) + 8, W - 120)} y={T} width={110} height={40} rx={6} fill="var(--s2)" stroke="var(--line2)" />
              <text x={Math.min(x(hover) + 8, W - 120) + 10} y={T + 17} style={{ fill: "var(--tx)", fontSize: 13, fontWeight: 600 }}>{hv.value == null ? "—" : fm(hv.value) + u}</text>
              <text x={Math.min(x(hover) + 8, W - 120) + 10} y={T + 32} style={{ fill: "var(--mu)", fontSize: 11 }}>{shortDate(hv.date)}</text>
            </g>
          )}
        </svg>
      )}
    </div>
  );
}
