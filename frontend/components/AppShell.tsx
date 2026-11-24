"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/Icons";
import { useSettings } from "@/lib/settings";
import { getBowlers, getHealth, getToken, setToken } from "@/lib/api";
import type { BowlerSummary } from "@/types/analysis";

const NAV: [string, string, string, string][] = [
  ["/new", "New analysis", "New", "new"],
  ["/deliveries", "My deliveries", "Deliveries", "list"],
  ["/profile", "Bowler profiles", "Profile", "user"],
  ["/compare", "Compare", "Compare", "cols"],
  ["/settings", "Settings", "Settings", "set"],
];

function current(path: string): string {
  if (path.startsWith("/processing")) return "/new";
  if (path.startsWith("/results")) return "/deliveries";
  return NAV.find(([h]) => path.startsWith(h))?.[0] ?? "";
}

export function Logo() {
  return (
    <Link href="/" className="row" style={{ gap: 10, color: "var(--tx)", flexWrap: "nowrap" }} aria-label="BowlIQ home">
      <span style={{ width: 30, height: 30, borderRadius: 99, background: "#2BD576", color: "#04140A", display: "grid", placeItems: "center", fontWeight: 700 }}>B</span>
      <span className="lbl" style={{ fontWeight: 600, fontSize: 17 }}>BowlIQ</span>
    </Link>
  );
}

export function BowlerPicker() {
  const { settings, update } = useSettings();
  const [open, setOpen] = useState(false);
  const [bowlers, setBowlers] = useState<BowlerSummary[]>([]);
  const [name, setName] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    getBowlers().then(setBowlers).catch(() => setBowlers([]));
    const off = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", off);
    return () => document.removeEventListener("mousedown", off);
  }, [open]);
  const pick = (n: string) => { update({ bowler: n }); setOpen(false); };
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button className="btn sm" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span style={{ width: 22, height: 22, borderRadius: 99, background: "var(--s3)", display: "grid", placeItems: "center", fontSize: 11, fontWeight: 600 }}>
          {settings.bowler.slice(0, 2).toUpperCase()}
        </span>
        {settings.bowler}
        <Icon name="chev" size={14} />
      </button>
      {open && (
        <div className="card tight" role="listbox" style={{ position: "absolute", right: 0, top: 42, width: 260, zIndex: 80, boxShadow: "0 12px 32px rgba(0,0,0,.35)" }}>
          <div className="label" style={{ marginBottom: 8 }}>Bowler</div>
          {bowlers.length === 0 && <p className="mu small" style={{ marginBottom: 8 }}>No saved bowlers yet.</p>}
          {bowlers.map((b) => (
            <button key={b.name} role="option" aria-selected={b.name === settings.bowler} className="nav-item"
              style={{ width: "100%", border: 0, background: b.name === settings.bowler ? "var(--s2)" : "transparent", justifyContent: "space-between" }}
              onClick={() => pick(b.name)}>
              <span style={{ color: "var(--tx)" }}>{b.name}</span>
              <span className="xs">{b.deliveries} {b.deliveries === 1 ? "delivery" : "deliveries"}</span>
            </button>
          ))}
          <form className="row" style={{ marginTop: 8, flexWrap: "nowrap" }} onSubmit={(e) => { e.preventDefault(); if (name.trim()) { pick(name.trim()); setName(""); } }}>
            <input className="input" placeholder="New bowler name" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} style={{ padding: "6px 10px" }} />
            <button className="btn sm" type="submit">Add</button>
          </form>
        </div>
      )}
    </div>
  );
}

export function ThemeToggle() {
  const { settings, update } = useSettings();
  const dark = settings.theme === "dark";
  return (
    <button className="btn sm ghost" aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={() => update({ theme: dark ? "light" : "dark" })}>
      <Icon name={dark ? "moon" : "sun"} />
    </button>
  );
}

function OfflineBanner() {
  const [down, setDown] = useState(false);
  useEffect(() => {
    let alive = true;
    const check = () => getHealth().then(() => alive && setDown(false)).catch(() => alive && setDown(true));
    check();
    const t = setInterval(check, 15000);
    return () => { alive = false; clearInterval(t); };
  }, []);
  if (!down) return null;
  return (
    <div className="warnbox" style={{ borderRadius: 0, borderLeft: 0, borderRight: 0, padding: "8px 24px", fontSize: 14 }} role="alert">
      <b>Backend offline.</b> Start it with <span className="mono">uvicorn app.main:app</span> in the backend folder. Saved reports load again once it is running.
    </div>
  );
}

/** Redirects to /login when no access token is stored. Renders nothing until checked (no flash of app UI). */
function useAuthGuard(): boolean {
  const router = useRouter();
  const path = usePathname();
  const [ok, setOk] = useState(false);
  useEffect(() => {
    if (getToken()) setOk(true);
    else router.replace(`/login?next=${encodeURIComponent(path)}`);
  }, [router, path]);
  return ok;
}

export function SignOut() {
  return (
    <button className="btn sm ghost" onClick={() => { setToken(null); window.location.assign("/login"); }} aria-label="Sign out">
      Sign out
    </button>
  );
}

export function AppShell({ title, actions, children }: { title: string; actions?: React.ReactNode; children: React.ReactNode }) {
  const path = usePathname();
  const cur = current(path);
  const authed = useAuthGuard();
  if (!authed) return null;
  return (
    <>
      <title>{`${title} · BowlIQ`}</title>
      <aside className="shell-side" aria-label="Main navigation">
        <div style={{ padding: "4px 4px 20px" }}><Logo /></div>
        {NAV.map(([href, label, , icon]) => (
          <Link key={href} href={href} className="nav-item" aria-current={cur === href ? "page" : undefined} title={label}>
            <Icon name={icon} /><span className="lbl">{label}</span>
          </Link>
        ))}
        <div style={{ flex: 1 }} />
        <Link href="/about" className="nav-item" title="How it works"><Icon name="info" /><span className="lbl">How it works</span></Link>
      </aside>
      <div className="shell-main">
        <header className="shell-top">
          <h1 style={{ fontSize: 18, fontWeight: 600 }}>{title}</h1>
          <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
            {actions}
            <BowlerPicker />
            <ThemeToggle />
            <SignOut />
          </div>
        </header>
        <OfflineBanner />
        <main className="shell-content">{children}</main>
      </div>
      <nav className="tabbar" aria-label="Main navigation">
        {NAV.map(([href, , short, icon]) => (
          <Link key={href} href={href} aria-current={cur === href ? "page" : undefined}><Icon name={icon} />{short}</Link>
        ))}
      </nav>
    </>
  );
}
