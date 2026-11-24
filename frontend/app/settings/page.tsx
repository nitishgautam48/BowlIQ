"use client";
import { useCallback, useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ErrorBox, StatusIcon } from "@/components/Common";
import { useToast } from "@/components/Toast";
import { API_URL, getHealth, getMe, toApiError, type ApiError } from "@/lib/api";
import { SignOut } from "@/components/AppShell";
import { useSettings, type Settings } from "@/lib/settings";
import type { HealthResponse } from "@/types/analysis";

export default function SettingsPage() {
  const { settings, update } = useSettings();
  const flash = useToast();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [owner, setOwner] = useState<string | null>(null);
  useEffect(() => { getMe().then((m) => setOwner(m.owner)).catch(() => setOwner(null)); }, []);
  const load = useCallback(() => { setErr(null); getHealth().then(setHealth).catch((e) => setErr(toApiError(e))); }, []);
  useEffect(load, [load]);
  const seg = <K extends keyof Settings>(label: string, hint: string, key: K, opts: [Settings[K], string][]) => (
    <div className="spread" style={{ padding: "16px 0", borderTop: "1px solid var(--line)" }}>
      <div><b>{label}</b><p className="small mu">{hint}</p></div>
      <div className="seg" role="group" aria-label={label}>
        {opts.map(([v, l]) => <button key={l} aria-pressed={settings[key] === v} onClick={() => { if (settings[key] !== v) { update({ [key]: v } as Partial<Settings>); flash("Settings saved"); } }}>{l}</button>)}
      </div>
    </div>
  );
  const rows: [string, boolean, string][] = health ? [
    ["ffmpeg", health.ffmpeg, health.details.ffmpeg],
    ["Pose model", health.model_file, health.details.model_file],
    ["Database", health.database, health.details.database],
    ["Pose engine", health.pose_device !== "unknown", health.details.pose_device],
    ["Video encoder", health.video_encoder !== "unknown", health.details.video_encoder],
    ["Gemini report writer", health.gemini_key, health.details.gemini],
  ] : [];
  return (
    <AppShell title="Settings">
      <div className="col" style={{ gap: 16, maxWidth: 820 }}>
        <div className="card" style={{ paddingTop: 8 }}>
          {seg("Default report view", "Bowler shows cues first. Coach shows angles, frames and confidence.", "view", [["bowler", "Bowler"], ["coach", "Coach"]])}
          {seg("Theme", "Applies on this device.", "theme", [["dark", "Dark"], ["light", "Light"]])}
          {seg("Glossary hints", "Explain terms like BFC and FFC on hover.", "hints", [[true, "On"], [false, "Off"]])}
        </div>
        <div className="card spread">
          <div><b>Account</b><p className="small mu">Signed in as <span className="mono">{owner ?? "…"}</span>. Your deliveries are private to this account.</p></div>
          <SignOut />
        </div>
        <div className="card">
          <div className="spread" style={{ marginBottom: 8 }}><b>Backend status</b><button className="btn sm" onClick={load}>Re-check</button></div>
          <p className="small mu" style={{ marginBottom: 12 }}>Server: <span className="mono">{API_URL}</span></p>
          {err && <ErrorBox error={err} />}
          {rows.map(([k, ok, d]) => (
            <div key={k} className="row" style={{ padding: "8px 0", borderTop: "1px solid var(--line)", flexWrap: "nowrap" }}>
              <StatusIcon status={ok ? "ok" : k.startsWith("Gemini") ? "info" : "warn"} />
              <span style={{ width: 180, flexShrink: 0 }}>{k}</span><span className="small mu">{d}</span>
            </div>
          ))}
        </div>
      </div>
    </AppShell>
  );
}
