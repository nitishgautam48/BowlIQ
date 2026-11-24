"use client";
import type { Confidence } from "@/types/analysis";
import { CONF_COLOR } from "@/lib/format";
import { Tip } from "@/components/Tooltip";
import type { ApiError } from "@/lib/api";
import { Icon } from "@/components/Icons";

/** Three-bar confidence pip with an explanatory tooltip. */
export function ConfPip({ level, reason, label = true, tip = true }: { level: Confidence | null; reason?: string; label?: boolean; tip?: boolean }) {
  if (!level) return <span className="mu small">—</span>;
  const on = { High: [1, 1, 1], Medium: [1, 1, 0.25], Low: [1, 0.25, 0.25] }[level];
  const color = CONF_COLOR[level];
  const body = (
    <span className="row" style={{ gap: 6, color, display: "inline-flex", flexWrap: "nowrap" }} aria-label={`${level} confidence. ${reason || ""}`}>
      <span style={{ display: "inline-flex", gap: 2, alignItems: "flex-end" }}>
        {on.map((o, i) => (
          <span key={i} style={{ width: 4, height: 6 + i * 3, borderRadius: 1, background: "currentColor", opacity: o }} />
        ))}
      </span>
      {label && <span className="small" style={{ fontWeight: 500 }}>{level}</span>}
    </span>
  );
  return reason && tip ? <Tip text={`${level} confidence. ${reason}`}>{body}</Tip> : body;
}

export function ErrorBox({ error, onRetry, title }: { error: ApiError | { message: string; fix: string; code?: string }; onRetry?: () => void; title?: string }) {
  return (
    <div className="err" role="alert">
      <div className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
        <Icon name="fail" color="var(--rtx)" size={20} />
        <div className="col" style={{ gap: 6, flex: 1 }}>
          <strong>{title || error.message}</strong>
          {title && <span>{error.message}</span>}
          <span className="mu small"><b style={{ color: "var(--tx)" }}>How to fix: </b>{error.fix}</span>
          {error.code && <span className="fa xs mono">{error.code}</span>}
          {onRetry && <div><button className="btn sm" onClick={onRetry}>Try again</button></div>}
        </div>
      </div>
    </div>
  );
}

export function SkeletonBlocks({ blocks }: { blocks: [number, number][] }) {
  return (
    <div className="grid" aria-busy="true" aria-label="Loading">
      {blocks.map(([span, h], i) => (
        <div key={i} className="skel" style={{ gridColumn: `span ${span}`, height: h }} />
      ))}
    </div>
  );
}

export function Empty({ title, body, children }: { title: string; body: string; children?: React.ReactNode }) {
  return (
    <div className="card" style={{ textAlign: "center", padding: "56px 24px" }}>
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}><Icon name="film" size={40} color="var(--fa)" /></div>
      <h2 style={{ fontSize: 22, marginBottom: 8 }}>{title}</h2>
      <p className="mu" style={{ maxWidth: 460, margin: "0 auto 20px" }}>{body}</p>
      {children}
    </div>
  );
}

export function StatusIcon({ status }: { status: string }) {
  if (status === "in_range" || status === "ok" || status === "better") return <Icon name="ok" color="var(--gtx)" />;
  if (status === "outside" || status === "warn" || status === "worse") return <Icon name="warn" color="var(--atx)" />;
  if (status === "not_measured" || status === "not_comparable") return <Icon name="same" color="var(--fa)" />;
  return <Icon name="info" color="var(--mu)" />;
}
