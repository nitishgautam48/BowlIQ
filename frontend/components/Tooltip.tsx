"use client";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { useSettings } from "@/lib/settings";
import { GLOSSARY } from "@/lib/format";

interface TipState { text: string; x: number; y: number }
const Ctx = createContext<(text: string | null, el?: HTMLElement) => void>(() => {});

export function TooltipProvider({ children }: { children: React.ReactNode }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = useCallback((text: string | null, el?: HTMLElement) => {
    if (!text || !el) return setTip(null);
    const r = el.getBoundingClientRect();
    setTip({ text, x: Math.max(8, Math.min(window.innerWidth - 296, r.left)), y: r.bottom + 8 });
  }, []);
  useEffect(() => {
    const off = () => setTip(null);
    window.addEventListener("scroll", off, true);
    return () => window.removeEventListener("scroll", off, true);
  }, []);
  return (
    <Ctx.Provider value={show}>
      {children}
      {tip && <div className="tip" role="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </Ctx.Provider>
  );
}

/** Wraps children in a span that shows `text` on hover/focus. */
export function Tip({ text, children, glossary = false }: { text: string; children: React.ReactNode; glossary?: boolean }) {
  const show = useContext(Ctx);
  const { settings } = useSettings();
  if (glossary && !settings.hints) return <>{children}</>;
  return (
    <span
      tabIndex={0}
      style={glossary ? { textDecoration: "underline dotted", textUnderlineOffset: 3, cursor: "help" } : { cursor: "help" }}
      onMouseEnter={(e) => show(text, e.currentTarget)}
      onMouseLeave={() => show(null)}
      onFocus={(e) => show(text, e.currentTarget)}
      onBlur={() => show(null)}
      onClick={(e) => { e.stopPropagation(); show(text, e.currentTarget); }}
      role="button"
      aria-label={text}
    >
      {children}
    </span>
  );
}

export function Gloss({ k, children }: { k: keyof typeof GLOSSARY | string; children: React.ReactNode }) {
  const text = GLOSSARY[k];
  if (!text) return <>{children}</>;
  return <Tip text={text} glossary>{children}</Tip>;
}
