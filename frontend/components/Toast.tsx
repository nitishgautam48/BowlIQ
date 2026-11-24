"use client";
import { createContext, useCallback, useContext, useRef, useState } from "react";

interface ToastAction { label: string; fn: () => void }
interface ToastState { msg: string; action?: ToastAction }
const Ctx = createContext<(msg: string, action?: ToastAction) => void>(() => {});

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [t, setT] = useState<ToastState | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flash = useCallback((msg: string, action?: ToastAction) => {
    setT({ msg, action });
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setT(null), 4500);
  }, []);
  return (
    <Ctx.Provider value={flash}>
      {children}
      {t && (
        <div className="toast" role="status" aria-live="polite">
          <span>{t.msg}</span>
          {t.action && (
            <button onClick={() => { const a = t.action; setT(null); a?.fn(); }}>{t.action.label}</button>
          )}
          <button onClick={() => setT(null)} aria-label="Dismiss notification" style={{ textDecoration: "none", fontSize: 16, lineHeight: 1 }}>×</button>
        </div>
      )}
    </Ctx.Provider>
  );
}

export const useToast = () => useContext(Ctx);
