import type { Confidence, Metric } from "@/types/analysis";

const DEC: Record<string, number> = { "°": 0, "% height": 1, "m/s": 1, "% height/s": 0, "km/h": 0, "%": 0, ms: 0, "% leg": 1 };

export function fmtNum(v: number, unit: string): string {
  const d = DEC[unit] ?? 1;
  const s = Math.abs(v).toFixed(d);
  return (v < 0 ? "−" : "") + s;
}

export function fmtVal(v: number | null | undefined, unit: string): string {
  if (v == null) return "—";
  return unit === "°" ? `${fmtNum(v, unit)}°` : `${fmtNum(v, unit)} ${unit}`;
}

export function fmtDate(iso: string, withTime = false): string {
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  const date = d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  return withTime ? `${date} · ${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}` : date;
}

export function shortDate(iso: string): string {
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

export const STATUS_TEXT: Record<Metric["status"], string> = {
  in_range: "In range",
  outside: "Outside target",
  not_measured: "Not measured",
  info: "Info",
};

export const CONF_COLOR: Record<Confidence, string> = { High: "var(--gtx)", Medium: "var(--tx)", Low: "var(--atx)" };

export const EVENT_TEXT: Record<string, string> = {
  BFC: "Back foot contact",
  FFC: "Front foot contact",
  release: "Release",
  "run-up": "Run-up",
};

export const GLOSSARY: Record<string, string> = {
  BFC: "Back foot contact (BFC): the frame your back foot lands in the delivery stride.",
  FFC: "Front foot contact (FFC): the frame your front foot lands. The front leg should brace from here to release.",
  release: "Release: the frame the ball leaves your hand (highest point of the bowling wrist after FFC).",
  SCR: "Shoulder counter-rotation: how far your shoulders twist away from the batter between back foot and front foot contact.",
  HSS: "Hip–shoulder separation: the angle between your hip line and shoulder line at back foot contact, seen front-on.",
  LF: "Lateral flexion: how far your trunk bends sideways at release, seen front-on.",
};

/** Camera view of a measurement. "side-on" = the main clip; the others are the second clip's detected angle. */
export const VIEW_LABEL: Record<string, string> = {
  "side-on": "Side-on", "front-on": "Front-on", behind: "Behind bowler", diagonal: "Diagonal", "3d": "3D",
};
/** Measured on the second clip's frames. 3D metrics are timed on the side-on clip, so they show its frames. */
export const isSecondClip = (view: string) => view !== "side-on" && view !== "3d";
