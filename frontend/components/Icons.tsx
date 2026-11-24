const P: Record<string, string> = {
  new: "M12 5v14M5 12h14",
  list: "M4 6h16M4 12h16M4 18h10",
  user: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4 21a8 8 0 0 1 16 0",
  cols: "M4 4h7v16H4zM13 4h7v16h-7z",
  set: "M4 7h10M18 7h2M4 17h4M12 17h8M16 5v4M10 15v4",
  ok: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M8 12l3 3 5-6",
  warn: "M12 3 22 20H2Z M12 10v4 M12 17h.01",
  fail: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M9 9l6 6M15 9l-6 6",
  same: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M8 12h8",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z M12 11v6 M12 7.5h.01",
  moon: "M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z",
  sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  play: "M7 4v16l13-8Z",
  pause: "M6 4h4v16H6zM14 4h4v16h-4z",
  skel: "M12 7a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM12 7v7M8 21l4-7 4 7M7 10h10",
  angle: "M4 20h16M4 20 16 6M10 20a6 6 0 0 0-2-4.5",
  chev: "M6 9l6 6 6-6",
  prev: "M15 6l-6 6 6 6",
  next: "M9 6l6 6-6 6",
  up: "M7 10v11M7 10l4-7c1.5 0 2.5 1 2.5 2.5V9h5.5a2 2 0 0 1 2 2.3l-1.3 7.5a2 2 0 0 1-2 1.7H7M3 10h4v11H3z",
  down: "M17 14V3M17 14l-4 7c-1.5 0-2.5-1-2.5-2.5V15H5a2 2 0 0 1-2-2.3l1.3-7.5a2 2 0 0 1 2-1.7H17M21 14h-4V3h4z",
  dl: "M12 4v12M7 11l5 5 5-5M5 20h14",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  upload: "M12 16V4M7 9l5-5 5 5M5 20h14",
  film: "M4 4h16v16H4zM4 8h16M4 16h16M8 4v4M16 4v4M8 16v4M16 16v4",
  x: "M6 6l12 12M18 6 6 18",
  ruler: "M3 17 17 3l4 4L7 21zM7 13l2 2M10 10l2 2M13 7l2 2",
};

export function Icon({ name, size = 18, color = "currentColor", strokeWidth = 1.8 }: { name: keyof typeof P | string; size?: number; color?: string; strokeWidth?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ flexShrink: 0 }}>
      <path d={P[name] || ""} />
    </svg>
  );
}
