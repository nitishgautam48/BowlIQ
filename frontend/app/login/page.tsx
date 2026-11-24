"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { Logo, ThemeToggle } from "@/components/AppShell";
import { ErrorBox } from "@/components/Common";
import { checkToken, setToken, toApiError, type ApiError } from "@/lib/api";

function LoginForm() {
  const router = useRouter();
  const next = useSearchParams().get("next");
  const [token, setTok] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);
  const safeNext = next && next.startsWith("/") && !next.startsWith("//") ? next : "/deliveries";
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await checkToken(token.trim());
      setToken(token.trim());
      router.replace(safeNext);
    } catch (e2) {
      setErr(toApiError(e2));
      setBusy(false);
    }
  };
  return (
    <form onSubmit={submit} className="card col" style={{ gap: 16, maxWidth: 480, width: "100%" }} aria-labelledby="login-title">
      <h1 id="login-title" style={{ fontSize: 24 }}>Sign in</h1>
      <p className="mu small">
        Paste your BowlIQ access token. When the backend starts for the first time it prints the token in its console
        and saves it in <span className="mono">backend/storage/.api_token</span>.
      </p>
      <label className="field">
        <span className="label">Access token</span>
        <input className="input mono" type="password" autoComplete="current-password" value={token} onChange={(e) => setTok(e.target.value)} required minLength={16} />
      </label>
      {err && <ErrorBox error={err} title={err.status === 401 ? "That token isn't valid" : undefined} />}
      <button className="btn primary" type="submit" disabled={busy || token.trim().length < 16}>{busy ? "Checking…" : "Sign in"}</button>
    </form>
  );
}

export default function Login() {
  return (
    <div>
      <title>Sign in · BowlIQ</title>
      <header className="mk-top"><Logo /><ThemeToggle /></header>
      <main className="mk" style={{ display: "flex", justifyContent: "center", padding: "48px 16px" }}>
        <Suspense fallback={null}><LoginForm /></Suspense>
      </main>
    </div>
  );
}
