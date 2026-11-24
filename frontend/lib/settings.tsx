"use client";
// Per-device preferences (theme, default report view, glossary hints, current bowler) kept in localStorage.
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export interface Settings {
  theme: "dark" | "light";
  view: "bowler" | "coach";
  hints: boolean;
  bowler: string;
  heightCm: number | null;
}

const DEFAULTS: Settings = { theme: "dark", view: "bowler", hints: true, bowler: "Me", heightCm: null };
const KEY = "bowliq.settings.v1";

interface Ctx {
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
  ready: boolean;
}

const SettingsContext = createContext<Ctx>({ settings: DEFAULTS, update: () => {}, ready: false });

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) setSettings({ ...DEFAULTS, ...JSON.parse(raw) });
    } catch {
      /* private mode: keep defaults */
    }
    setReady(true);
  }, []);
  useEffect(() => {
    document.body.dataset.theme = settings.theme;
    document.documentElement.dataset.theme = settings.theme;
  }, [settings.theme]);
  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((s) => {
      const next = { ...s, ...patch };
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);
  const value = useMemo(() => ({ settings, update, ready }), [settings, update, ready]);
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export const useSettings = () => useContext(SettingsContext);
