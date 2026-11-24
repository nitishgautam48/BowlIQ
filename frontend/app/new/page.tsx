"use client";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ErrorBox } from "@/components/Common";
import { Icon } from "@/components/Icons";
import { SLOW_OPTIONS } from "@/components/MarkMoments";
import { guidanceFields, NO_GUIDANCE, PersonPicker, type Guidance } from "@/components/PersonPicker";
import { analyze, getBowlers, getHealth, toApiError, type ApiError } from "@/lib/api";
import { useSettings } from "@/lib/settings";
import { useSampleAnalysis } from "@/lib/useSample";
import type { BowlerSummary } from "@/types/analysis";

const MAX_BYTES = 200 * 1024 * 1024;
type Check = { label: string; value: string; status: "ok" | "warn" | "fail"; note: string };
interface ClipInfo { file: File; url: string; duration: number; width: number; height: number; checks: Check[] }

function readClip(file: File): Promise<ClipInfo> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const checks: Check[] = [];
    const ext = file.name.toLowerCase().split(".").pop() || "";
    if (!["mp4", "mov"].includes(ext)) checks.push({ label: "Format", value: ext.toUpperCase() || "Unknown", status: "fail", note: "Only MP4 or MOV files are accepted." });
    if (file.size > MAX_BYTES) checks.push({ label: "Size", value: `${(file.size / 1048576).toFixed(0)} MB`, status: "fail", note: "The limit is 200 MB. Trim the clip or export at a lower bitrate." });
    const v = document.createElement("video");
    v.preload = "metadata";
    v.muted = true;
    v.onloadedmetadata = () => {
      const d = v.duration, w = v.videoWidth, h = v.videoHeight;
      checks.push({ label: "Resolution", value: `${w}×${h}`, status: Math.min(w, h) >= 720 ? "ok" : "warn", note: Math.min(w, h) >= 720 ? "" : "Below 720p: joints may be harder to find." });
      checks.push({ label: "Length", value: `${d.toFixed(1)} s`, status: d < 1 ? "fail" : "ok", note: d < 1 ? "A clip under 1 s can't contain a full delivery." : "You can trim it in the next step." });
      checks.push({ label: "Orientation", value: w >= h ? "Landscape" : "Portrait", status: w >= h ? "ok" : "warn", note: w >= h ? "" : "Landscape keeps more of the run-up in view." });
      checks.push({ label: "Frame rate", value: "Checked on upload", status: "ok", note: "Browsers can't read frame rate; the server checks it (60 fps or more recommended)." });
      resolve({ file, url, duration: d, width: w, height: h, checks });
    };
    v.onerror = () => {
      checks.push({ label: "File", value: "Unreadable", status: "fail", note: "Your browser can't decode this file. Export it as MP4 (H.264) and try again." });
      resolve({ file, url, duration: 0, width: 0, height: 0, checks });
    };
    v.src = url;
  });
}

function DropZone({ title, tag, hint, clip, onFile, onClear }: { title: string; tag: string; hint: string; clip: ClipInfo | null; onFile: (f: File) => void; onClear: () => void }) {
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  if (clip) {
    return (
      <div className="card">
        <div className="spread" style={{ marginBottom: 12 }}>
          <div className="row"><Icon name="film" /><b>{title}</b><span className="chip">{tag}</span></div>
          <button className="btn sm ghost" onClick={onClear} aria-label={`Remove ${title}`}><Icon name="x" size={16} /> Remove</button>
        </div>
        <p className="small" style={{ marginBottom: 10, wordBreak: "break-all" }}>{clip.file.name} · {(clip.file.size / 1048576).toFixed(1)} MB</p>
        <div className="col" style={{ gap: 6 }} aria-label={`${title} checks`}>
          {clip.checks.map((c) => (
            <div key={c.label} className="row" style={{ flexWrap: "nowrap", alignItems: "flex-start" }}>
              <Icon name={c.status === "ok" ? "ok" : c.status === "warn" ? "warn" : "fail"} color={c.status === "ok" ? "var(--gtx)" : c.status === "warn" ? "var(--atx)" : "var(--rtx)"} />
              <span className="small" style={{ width: 96, flexShrink: 0 }}>{c.label}</span>
              <span className="small mono" style={{ width: 130, flexShrink: 0 }}>{c.value}</span>
              <span className="small mu">{c.note}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className={`dz ${drag ? "drag" : ""}`} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) onFile(f); }}>
      <div className="row" style={{ justifyContent: "center", marginBottom: 8 }}><b>{title}</b><span className="chip">{tag}</span></div>
      <p className="mu small" style={{ marginBottom: 14 }}>{hint}</p>
      <button className="btn" onClick={() => input.current?.click()} aria-label={`Choose ${title}`}><Icon name="upload" /> Choose file or drop it here</button>
      <input ref={input} type="file" accept="video/mp4,video/quicktime,.mp4,.mov" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
    </div>
  );
}

function describe(g: Guidance, auto: string): string {
  const parts = [g.tap ? `tapped at ${g.tap[0].toFixed(2)} s` : auto];
  if (g.zones.length) parts.push(`${g.zones.length} ignore zone${g.zones.length > 1 ? "s" : ""}`);
  return parts.join(" · ");
}

function useThumbs(url: string | null, duration: number, n = 12): string[] {
  const [thumbs, setThumbs] = useState<string[]>([]);
  useEffect(() => {
    if (!url || !duration) return;
    let cancelled = false;
    const v = document.createElement("video");
    v.muted = true;
    v.preload = "auto";
    v.src = url;
    const canvas = document.createElement("canvas");
    const out: string[] = [];
    const grab = (i: number) => {
      if (cancelled || i >= n) { if (!cancelled) setThumbs(out.slice()); return; }
      v.currentTime = Math.min(duration - 0.01, ((i + 0.5) / n) * duration);
    };
    v.onseeked = () => {
      canvas.width = 160;
      canvas.height = Math.round((160 * v.videoHeight) / Math.max(1, v.videoWidth));
      canvas.getContext("2d")?.drawImage(v, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((b) => {
        if (cancelled) return;
        out.push(b ? URL.createObjectURL(b) : "");
        setThumbs(out.slice());
        grab(out.length);
      }, "image/jpeg", 0.6);
    };
    v.onloadeddata = () => grab(0);
    return () => { cancelled = true; v.removeAttribute("src"); v.load(); out.forEach((u) => u && URL.revokeObjectURL(u)); };
  }, [url, duration, n]);
  return thumbs;
}

function Trimmer({ clip, trim, setTrim }: { clip: ClipInfo; trim: [number, number]; setTrim: (t: [number, number]) => void }) {
  const D = clip.duration;
  const thumbs = useThumbs(clip.url, D);
  const preview = useRef<HTMLVideoElement>(null);
  const strip = useRef<HTMLDivElement>(null);
  const [focus, setFocus] = useState<"start" | "end">("start");
  const [dragging, setDragging] = useState<"start" | "end" | null>(null);
  const [a, b] = trim;
  const tf = focus === "start" ? a : b;
  useEffect(() => { if (preview.current) preview.current.currentTime = Math.min(tf, D - 0.01); }, [tf, D]);
  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => {
      const r = strip.current?.getBoundingClientRect();
      if (!r) return;
      const t = Math.round(Math.max(0, Math.min(D, ((e.clientX - r.left) / r.width) * D)) * 100) / 100;
      if (dragging === "start") setTrim([Math.min(t, b - 1), b]); else setTrim([a, Math.max(t, a + 1)]);
    };
    const up = () => setDragging(null);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
  }, [dragging, a, b, D, setTrim]);
  const key = (which: "start" | "end") => (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const d = e.key === "ArrowRight" ? 0.05 : -0.05;
    setFocus(which);
    if (which === "start") setTrim([Math.max(0, Math.min(b - 1, +(a + d).toFixed(2))), b]);
    else setTrim([a, Math.min(D, Math.max(a + 1, +(b + d).toFixed(2)))]);
  };
  const pct = (t: number) => `${(t / D) * 100}%`;
  return (
    <div className="col" style={{ gap: 16 }}>
      <div style={{ position: "relative", aspectRatio: `${clip.width || 16} / ${clip.height || 9}`, maxHeight: 420, background: "#000", borderRadius: 12, overflow: "hidden", margin: "0 auto", width: "100%" }}>
        <video ref={preview} src={clip.url} muted playsInline preload="auto" style={{ width: "100%", height: "100%", objectFit: "contain" }} aria-label={`Preview at ${tf.toFixed(2)} s`} />
        <span className="chip mono" style={{ position: "absolute", left: 12, top: 12, background: "var(--glass)", color: "var(--tx)" }}>{focus === "start" ? "Start" : "End"} · {tf.toFixed(2)} s</span>
      </div>
      <div ref={strip} style={{ position: "relative", height: 64, borderRadius: 10, overflow: "hidden", display: "flex", background: "var(--s2)", userSelect: "none", touchAction: "none" }}>
        {Array.from({ length: 12 }).map((_, i) => (
          <div key={i} style={{ flex: 1, backgroundImage: thumbs[i] ? `url("${thumbs[i]}")` : undefined, backgroundSize: "cover", backgroundPosition: "center",
            opacity: ((i + 0.5) / 12) * D >= a && ((i + 0.5) / 12) * D <= b ? 1 : 0.3, borderRight: "1px solid var(--bg)" }} />
        ))}
        <div style={{ position: "absolute", top: 0, bottom: 0, left: pct(a), width: `${((b - a) / D) * 100}%`, border: "2px solid #2BD576", borderRadius: 8, pointerEvents: "none" }} />
        {(["start", "end"] as const).map((w) => (
          <div key={w} role="slider" tabIndex={0} aria-label={`Trim ${w}`} aria-valuemin={0} aria-valuemax={D} aria-valuenow={w === "start" ? a : b}
            onKeyDown={key(w)} onPointerDown={(e) => { e.preventDefault(); setFocus(w); setDragging(w); }} onFocus={() => setFocus(w)}
            style={{ position: "absolute", top: 0, bottom: 0, left: `calc(${pct(w === "start" ? a : b)} - 8px)`, width: 16, cursor: "ew-resize", display: "grid", placeItems: "center" }}>
            <div style={{ width: 6, height: 36, borderRadius: 3, background: "#2BD576" }} />
          </div>
        ))}
      </div>
      <div className="spread small">
        <span className="mono">{a.toFixed(2)} s → {b.toFixed(2)} s</span>
        <span className="mu">{(b - a).toFixed(2)} s selected · arrow keys nudge by 0.05 s</span>
        <button className="btn sm" onClick={() => setTrim([0, D])}>Use whole clip</button>
      </div>
    </div>
  );
}

export default function NewAnalysis() {
  const router = useRouter();
  const { settings, update } = useSettings();
  const sample = useSampleAnalysis();
  const [geminiVideo, setGeminiVideo] = useState(false);
  useEffect(() => { getHealth().then((h) => setGeminiVideo(!!h.gemini_video)).catch(() => {}); }, []);
  const [step, setStep] = useState(1);
  const [side, setSide] = useState<ClipInfo | null>(null);
  const [front, setFront] = useState<ClipInfo | null>(null);
  const [trim, setTrim] = useState<[number, number]>([0, 0]);
  const [sideG, setSideG] = useState<Guidance>(NO_GUIDANCE);
  const [frontG, setFrontG] = useState<Guidance>(NO_GUIDANCE);
  const [name, setName] = useState("");
  const [height, setHeight] = useState("");
  const [slow, setSlow] = useState(0);
  const [camView, setCamView] = useState<"" | "side-on" | "front-on" | "behind">("");
  const [armSel, setArmSel] = useState<"" | "right" | "left">("");
  const [bowlers, setBowlers] = useState<BowlerSummary[]>([]);
  const [upload, setUpload] = useState<number | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => { getBowlers().then(setBowlers).catch(() => {}); }, []);
  // release blob URLs when clips are replaced/removed or the page unmounts (memory leak fix)
  useEffect(() => () => { if (side) URL.revokeObjectURL(side.url); }, [side]);
  useEffect(() => () => { if (front) URL.revokeObjectURL(front.url); }, [front]);
  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => { setName(settings.bowler); if (settings.heightCm) setHeight(String(settings.heightCm)); }, [settings.bowler, settings.heightCm]);
  const pick = async (which: "side" | "front", f: File) => {
    const info = await readClip(f);
    if (which === "side") { setSide(info); setTrim([0, info.duration]); setSideG(NO_GUIDANCE); } else { setFront(info); setFrontG(NO_GUIDANCE); }
  };
  const h = height.trim() === "" ? null : Number(height);
  const heightErr = h != null && (Number.isNaN(h) || h < 120 || h > 230) ? "Enter a height between 120 and 230 cm." : null;
  const blocked = !side || side.checks.some((c) => c.status === "fail") || (front?.checks.some((c) => c.status === "fail") ?? false);
  const warned = (side?.checks.some((c) => c.status === "warn") ?? false) || (front?.checks.some((c) => c.status === "warn") ?? false);
  const dur = trim[1] - trim[0];
  // a tap outside the trimmed part would be rejected by the server: drop it when the trim moves away from it
  useEffect(() => {
    if (sideG.tap && (sideG.tap[0] < trim[0] - 0.05 || sideG.tap[0] > trim[1] + 0.05)) setSideG((g) => ({ ...g, tap: null }));
  }, [trim, sideG.tap]);
  const trimErr = side && (dur < 1 ? "Select at least 1 s." : dur > 30 ? "Select 30 s or less." : null);
  const steps = ["Upload", "Select the delivery", "Confirm"];
  const start = async () => {
    if (!side) return;
    setError(null);
    setUpload(0);
    const n = name.trim() || "Me";
    update({ bowler: n, heightCm: h });
    try {
      const whole = trim[0] <= 0.001 && trim[1] >= side.duration - 0.001;
      abortRef.current = new AbortController();
      const sg = guidanceFields(sideG), fg = guidanceFields(frontG);
      const job = await analyze({ sideVideo: side.file, frontVideo: front?.file, bowlerName: n, bowlerHeightCm: h,
        trimStartS: whole ? null : trim[0], trimEndS: whole ? null : trim[1],
        slowMotion: slow >= 1 ? slow : null, cameraView: camView || null, bowlingArm: armSel || null,
        sideTap: sg.tap, sideIgnore: sg.ignore, frontTap: front ? fg.tap : null, frontIgnore: front ? fg.ignore : null },
        setUpload, abortRef.current.signal);
      router.push(`/processing/${job.job_id}`);
    } catch (e) {
      setError(toApiError(e));
      setUpload(null);
    }
  };
  const rows = useMemo(() => [
    ["Side-on clip", side ? `${side.file.name} · ${side.width}×${side.height}` : "", 1],
    ["Delivery", `${trim[0].toFixed(2)} s → ${trim[1].toFixed(2)} s (${dur.toFixed(2)} s)`, 2],
    ["Bowler selection", describe(sideG, "automatic (the person running in)"), 2],
    ["Front-on clip", front ? front.file.name : "Not added. Alignment and side-bend will be skipped.", 1],
    ...(front ? [["Front-clip selection", describe(frontG, "automatic"), 2] as [string, string, number]] : []),
    ["Bowler", name.trim() || "Me", 1],
    ["Height", h ? `${h} cm` : "Not given: stride in % of height only, no ball speed", 1],
    ["Speed", slow > 1 ? `Slow motion ×${slow}` : slow === 1 ? "Real time" : "Detect automatically", 1],
    ["Camera view", camView || "Detect automatically", 1],
    ["Bowling arm", armSel ? `${armSel[0].toUpperCase()}${armSel.slice(1)}-arm` : "Detect automatically", 1],
  ] as [string, string, number][], [side, front, trim, dur, name, h, sideG, frontG, slow, camView, armSel]);

  return (
    <AppShell title="New analysis">
      <div className="col" style={{ gap: 24, maxWidth: 920 }}>
        {geminiVideo && (
          <p className="small mu" role="note">
            <Icon name="warn" size={14} /> Gemini video check is on: the main clip (trimmed, reduced, without sound) is sent to Google so
            Gemini can point out the bowler, camera view and bowling arm. All measurements are still made on this computer.
            Switch it off with <span className="mono">BOWLIQ_GEMINI_VIDEO=0</span> in backend/.env.
          </p>
        )}
        <ol className="row" style={{ listStyle: "none", padding: 0, margin: 0, gap: 8 }} aria-label="Steps">
          {steps.map((l, i) => {
            const n = i + 1, done = n < step, act = n === step;
            return (
              <li key={l} aria-current={act ? "step" : undefined} className="row" style={{ gap: 8, flex: 1, minWidth: 150 }}>
                <span style={{ width: 26, height: 26, borderRadius: 99, display: "grid", placeItems: "center", fontSize: 13, fontWeight: 600,
                  border: `1.5px solid ${n <= step ? "#2BD576" : "var(--line2)"}`, background: done ? "#2BD576" : "transparent", color: done ? "#04140A" : n <= step ? "var(--gtx)" : "var(--mu)" }}>{done ? "✓" : n}</span>
                <span style={{ fontWeight: act ? 600 : 500, color: n <= step ? "var(--tx)" : "var(--mu)", fontSize: 14 }}>{l}</span>
              </li>
            );
          })}
        </ol>

        {step === 1 && (
          <>
            <DropZone title="Side-on clip" tag="Required" hint="Film side-on, at 90° to the pitch, about 10 m away. MP4 or MOV, up to 200 MB." clip={side}
              onFile={(f) => pick("side", f)} onClear={() => setSide(null)} />
            <DropZone title="Front-on clip" tag="Optional" hint="Film from behind the stumps at the same time as the side-on phone. Clap once, loudly, after both start recording so the two clips are lined up automatically. Adds hip–shoulder alignment, counter-rotation, side bend and linked playback." clip={front}
              onFile={(f) => pick("front", f)} onClear={() => setFront(null)} />
            <div className="card" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(240px,1fr))", gap: 16 }}>
              <label className="field">
                <span className="label">Bowler</span>
                <input className="input" list="bowler-list" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} placeholder="Your name" />
                <datalist id="bowler-list">{bowlers.map((b) => <option key={b.name} value={b.name} />)}</datalist>
                <span className="hint">Deliveries are saved to this bowler's profile.</span>
              </label>
              <label className="field">
                <span className="label">Camera view of the main clip</span>
                <select className="input" value={camView} onChange={(e) => setCamView(e.target.value as typeof camView)}>
                  <option value="">Detect automatically</option>
                  <option value="side-on">Side-on (from the side)</option>
                  <option value="front-on">Front-on (facing the bowler)</option>
                  <option value="behind">Behind the bowler</option>
                </select>
                <span className="hint">Side-on gives the full analysis; front-on / behind give side bend, arm angle, head and foot alignment.</span>
              </label>
              <label className="field">
                <span className="label">Bowling arm</span>
                <select className="input" value={armSel} onChange={(e) => setArmSel(e.target.value as typeof armSel)}>
                  <option value="">Detect automatically</option>
                  <option value="right">Right-arm</option>
                  <option value="left">Left-arm</option>
                </select>
                <span className="hint">Setting it removes one possible mistake.</span>
              </label>
              <label className="field">
                <span className="label">Is the side-on clip slow motion?</span>
                <select className="input" value={slow} onChange={(e) => setSlow(Number(e.target.value))}>
                  {SLOW_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
                <span className="hint">TV replays are usually ×16–×25; phone slow-mo ×4 (120 fps) or ×8 (240 fps).</span>
              </label>
              <label className="field">
                <span className="label">Your height (cm)</span>
                <input className="input" inputMode="decimal" value={height} onChange={(e) => setHeight(e.target.value)} placeholder="Optional, e.g. 178" aria-invalid={!!heightErr} />
                <span className="hint" style={heightErr ? { color: "var(--rtx)" } : undefined}>{heightErr || "Needed for stride length in metres and ball speed"}</span>
              </label>
            </div>
            <div className="spread">
              <button className="btn ghost" onClick={sample.run} disabled={sample.busy}>{sample.busy ? "Uploading sample…" : "Or analyze a sample clip"}</button>
              <div className="row">
                {(!side || blocked || warned) && <span className="small" style={{ color: blocked ? "var(--mu)" : "var(--atx)" }}>
                  {!side ? "Add a side-on clip to continue." : blocked ? "Fix the failed check to continue." : "You can continue. Some results will be less precise."}</span>}
                <button className="btn primary" disabled={blocked || !!heightErr} onClick={() => setStep(2)}>Continue</button>
              </div>
            </div>
            {sample.error && <ErrorBox error={sample.error} />}
          </>
        )}

        {step === 2 && side && (
          <>
            <div className="card">
              <h2 style={{ fontSize: 18, marginBottom: 4 }}>Select the delivery</h2>
              <p className="mu small" style={{ marginBottom: 16 }}>Keep the last 3–4 run-up strides through to follow-through. Drag the handles or use the arrow keys.</p>
              <Trimmer clip={side} trim={trim} setTrim={setTrim} />
            </div>
            <div className="card">
              <h2 style={{ fontSize: 18, marginBottom: 4 }}>Who is the bowler?</h2>
              <p className="mu small" style={{ marginBottom: 16 }}>If other people are in view (umpire, batter, fielders, people walking past), tap the bowler on any frame
                where the whole body is visible, and shade anyone who should be ignored. This makes sure only the bowler is analysed.</p>
              <PersonPicker label="Side clip" url={side.url} width={side.width} height={side.height} range={trim} value={sideG} onChange={setSideG} />
            </div>
            {front && (
              <div className="card">
                <h2 style={{ fontSize: 18, marginBottom: 4 }}>Bowler in the front-on clip</h2>
                <p className="mu small" style={{ marginBottom: 16 }}>Optional. Tap the bowler in the front-on clip too if other people are in view.</p>
                <PersonPicker label="Front clip" url={front.url} width={front.width} height={front.height} range={[0, front.duration]} value={frontG} onChange={setFrontG} />
              </div>
            )}
            <div className="spread">
              <button className="btn" onClick={() => setStep(1)}>Back</button>
              <div className="row">{trimErr && <span className="small a">{trimErr}</span>}<button className="btn primary" disabled={!!trimErr} onClick={() => setStep(3)}>Continue</button></div>
            </div>
          </>
        )}

        {step === 3 && side && (
          <>
            <div className="card" style={{ padding: 0 }}>
              {rows.map(([k, v, s], i) => (
                <div key={k} className="spread" style={{ padding: "14px 20px", borderTop: i ? "1px solid var(--line)" : "none", flexWrap: "nowrap" }}>
                  <span className="mu small" style={{ width: 120, flexShrink: 0 }}>{k}</span>
                  <span className="small" style={{ flex: 1, wordBreak: "break-word" }}>{v}</span>
                  <button className="btn sm ghost" onClick={() => setStep(s)}>Change</button>
                </div>
              ))}
            </div>
            {error && <ErrorBox error={error} />}
            {upload != null && (
              <div className="col" style={{ gap: 6 }}>
                <div className="bar"><div style={{ width: `${Math.round(upload * 100)}%` }} /></div>
                <span className="row small mu" style={{ justifyContent: "space-between" }}>Uploading… {Math.round(upload * 100)}%
                  <button className="btn sm ghost" onClick={() => abortRef.current?.abort()}>Cancel upload</button></span>
              </div>
            )}
            <div className="spread">
              <button className="btn" onClick={() => setStep(2)} disabled={upload != null}>Back</button>
              <button className="btn primary lg" onClick={start} disabled={upload != null}>Analyze delivery</button>
            </div>
          </>
        )}
      </div>
    </AppShell>
  );
}
