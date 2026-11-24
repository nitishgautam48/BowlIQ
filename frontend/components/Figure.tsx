// Decorative illustration of a bowler at release (not data).
export function BowlerFigure({ size = 320 }: { size?: number }) {
  const P: Record<string, [number, number]> = {
    head: [297, 102], neck: [285, 122], rEl: [282, 78], rWr: [280, 38], lEl: [290, 160], lWr: [270, 180],
    pel: [240, 192], rKn: [215, 235], rAn: [190, 265], lKn: [278, 236], lAn: [285, 290],
  };
  const B: [string, string, number][] = [["neck", "pel", 1], ["neck", "rEl", 1], ["rEl", "rWr", 1], ["neck", "lEl", 0.5], ["lEl", "lWr", 0.5],
    ["pel", "rKn", 0.5], ["rKn", "rAn", 0.5], ["pel", "lKn", 1], ["lKn", "lAn", 1], ["neck", "head", 1]];
  return (
    <svg viewBox="120 0 260 320" width={size} height={size} aria-hidden="true" style={{ maxWidth: "100%", height: "auto" }}>
      <defs>
        <radialGradient id="hg"><stop offset="0%" stopColor="#2BD576" stopOpacity="0.22" /><stop offset="100%" stopColor="#2BD576" stopOpacity="0" /></radialGradient>
      </defs>
      <circle cx="240" cy="180" r="150" fill="url(#hg)" />
      <line x1="120" x2="380" y1="293" y2="293" stroke="var(--line2)" strokeWidth="1.5" />
      {B.map(([a, b, o], i) => (
        <line key={i} x1={P[a][0]} y1={P[a][1]} x2={P[b][0]} y2={P[b][1]} stroke="#2BD576" strokeWidth="3.2" strokeLinecap="round" opacity={o} />
      ))}
      <circle cx={P.head[0]} cy={P.head[1]} r="11" fill="var(--bg)" stroke="#2BD576" strokeWidth="2.6" />
      {Object.entries(P).filter(([k]) => k !== "head").map(([k, [x, y]]) => (
        <circle key={k} cx={x} cy={y} r="3.6" fill="var(--bg)" stroke="#2BD576" strokeWidth="2" />
      ))}
      <circle cx="280" cy="30" r="5" fill="#F87171" />
    </svg>
  );
}
