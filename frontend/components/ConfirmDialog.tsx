"use client";
import { useEffect, useRef } from "react";

/** Accessible confirmation dialog (replaces window.confirm). */
export function ConfirmDialog({ title, body, confirmLabel, onConfirm, onCancel, danger = false }: {
  title: string; body: string; confirmLabel: string; onConfirm: () => void; onCancel: () => void; danger?: boolean;
}) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    cancelRef.current?.focus();
    const k = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onCancel]);
  return (
    <div className="overlay" onClick={onCancel} role="presentation">
      <div className="dialog" role="alertdialog" aria-modal="true" aria-labelledby="cd-title" aria-describedby="cd-body"
        onClick={(e) => e.stopPropagation()} style={{ maxWidth: 440 }}>
        <h2 id="cd-title" style={{ fontSize: 20, marginBottom: 8 }}>{title}</h2>
        <p id="cd-body" className="mu small" style={{ marginBottom: 20 }}>{body}</p>
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button ref={cancelRef} className="btn" onClick={onCancel}>Cancel</button>
          <button className={`btn ${danger ? "danger" : "primary"}`} onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}
