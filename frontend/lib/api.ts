// HTTP client for the BowlIQ FastAPI backend (v1). Every request carries the access token and the client header;
// every error surfaces the backend's { code, message, fix }.
import type {
  AnalysisResult, AnalyzeInput, ApiErrorBody, BenchmarkRule, BowlerHistory, BowlerSummaryOut, BowlerSummary, CompareResponse,
  DeliverySummary, HealthResponse, JobCreated, JobState, KeypointsResponse, MediaLinks, Me, Overrides, SampleClip, SyncInfo,
} from "@/types/analysis";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
const V1 = `${API_URL}/api/v1`;
const TOKEN_KEY = "bowliq.token.v1";
const UPLOAD_TIMEOUT_MS = 15 * 60_000;

export class ApiError extends Error implements ApiErrorBody {
  code: string;
  fix: string;
  status: number;
  constructor(body: ApiErrorBody, status = 0) {
    super(body.message);
    this.code = body.code;
    this.fix = body.fix;
    this.status = status;
  }
}

const NETWORK_ERROR: ApiErrorBody = {
  code: "NETWORK",
  message: `Can't reach the BowlIQ server at ${API_URL}.`,
  fix: "Start the backend (cd backend, then `uvicorn app.main:app`) and check NEXT_PUBLIC_API_URL.",
};

// ───────────── token storage ─────────────
export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}
export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* private mode */
  }
}

function authHeaders(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  const t = getToken();
  if (t) h.set("Authorization", `Bearer ${t}`);
  h.set("X-BowlIQ-Client", "web");
  return h;
}

function onUnauthorized(): void {
  if (typeof window === "undefined") return;
  setToken(null);
  if (!window.location.pathname.startsWith("/login")) {
    window.location.assign(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
  }
}

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    if (body && typeof body.code === "string") return new ApiError(body as ApiErrorBody, res.status);
  } catch {
    /* fall through */
  }
  return new ApiError({ code: `HTTP_${res.status}`, message: `The server answered ${res.status}.`, fix: "Try again." }, res.status);
}

async function request<T>(path: string, init?: RequestInit & { noRedirect?: boolean }): Promise<T> {
  if (!init?.noRedirect && !getToken()) {
    onUnauthorized(); // not signed in: don't send a request that can only fail
    throw new ApiError({ code: "UNAUTHORIZED", message: "You are not signed in.", fix: "Sign in with your access token." }, 401);
  }
  let res: Response;
  try {
    res = await fetch(`${V1}${path}`, { cache: "no-store", ...init, headers: authHeaders(init?.headers) });
  } catch (e) {
    if ((e as Error)?.name === "AbortError") throw e;
    throw new ApiError(NETWORK_ERROR);
  }
  if (res.status === 401 && !init?.noRedirect) onUnauthorized();
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

export function toApiError(e: unknown): ApiError {
  if (e instanceof ApiError) return e;
  return new ApiError({ code: "CLIENT_ERROR", message: e instanceof Error ? e.message : String(e), fix: "Reload the page and try again." });
}

/** Verify a token without storing it. */
export async function checkToken(token: string): Promise<Me> {
  let res: Response;
  try {
    res = await fetch(`${V1}/me`, { cache: "no-store", headers: { Authorization: `Bearer ${token}`, "X-BowlIQ-Client": "web" } });
  } catch {
    throw new ApiError(NETWORK_ERROR);
  }
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as Me;
}

/** Upload clips and start an analysis (XHR for real upload progress). Abortable, with a 15-minute timeout. */
export function analyze(input: AnalyzeInput, onUploadProgress?: (fraction: number) => void, signal?: AbortSignal): Promise<JobCreated> {
  const form = new FormData();
  form.append("side_video", input.sideVideo, input.sideVideo.name);
  if (input.frontVideo) form.append("front_video", input.frontVideo, input.frontVideo.name);
  if (input.bowlerName) form.append("bowler_name", input.bowlerName);
  if (input.bowlerHeightCm != null) form.append("bowler_height_cm", String(input.bowlerHeightCm));
  if (input.trimStartS != null) form.append("trim_start_s", input.trimStartS.toFixed(3));
  if (input.trimEndS != null) form.append("trim_end_s", input.trimEndS.toFixed(3));
  if (input.sideTap) form.append("side_tap", input.sideTap);
  if (input.sideIgnore) form.append("side_ignore", input.sideIgnore);
  if (input.frontVideo && input.frontTap) form.append("front_tap", input.frontTap);
  if (input.frontVideo && input.frontIgnore) form.append("front_ignore", input.frontIgnore);
  if (input.slowMotion && input.slowMotion >= 1) form.append("slow_motion", String(input.slowMotion));
  if (input.cameraView) form.append("camera_view", input.cameraView);
  if (input.bowlingArm) form.append("bowling_arm", input.bowlingArm);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${V1}/analyze`);
    xhr.responseType = "json";
    xhr.timeout = UPLOAD_TIMEOUT_MS;
    authHeaders().forEach((v, k) => xhr.setRequestHeader(k, v));
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable && onUploadProgress) onUploadProgress(ev.loaded / ev.total);
    };
    const onAbort = () => xhr.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    xhr.onabort = () => reject(new ApiError({ code: "CANCELLED", message: "Upload cancelled.", fix: "Start the upload again." }));
    xhr.ontimeout = () => reject(new ApiError({ code: "TIMEOUT", message: "The upload took too long.", fix: "Check your connection or trim the clip, then try again." }));
    xhr.onerror = () => reject(new ApiError(NETWORK_ERROR));
    xhr.onload = () => {
      signal?.removeEventListener("abort", onAbort);
      const body = xhr.response;
      if (xhr.status === 401) onUnauthorized();
      if (xhr.status >= 200 && xhr.status < 300 && body?.job_id) resolve(body as JobCreated);
      else if (body && typeof body.code === "string") reject(new ApiError(body as ApiErrorBody, xhr.status));
      else reject(new ApiError({ code: `HTTP_${xhr.status}`, message: `Upload failed (${xhr.status}).`, fix: "Try again." }, xhr.status));
    };
    xhr.send(form);
  });
}

const enc = encodeURIComponent;
export const getMe = () => request<Me>("/me");
export const getJob = (jobId: string) => request<JobState>(`/jobs/${enc(jobId)}`);
export const getAnalysis = (id: string) => request<AnalysisResult>(`/analysis/${enc(id)}`);
export const getKeypoints = (id: string, view: "side" | "front" = "side") =>
  request<KeypointsResponse>(`/analysis/${enc(id)}/keypoints${view === "front" ? "?view=front" : ""}`);
export const getJobLinks = (jobId: string) => request<MediaLinks>(`/jobs/${enc(jobId)}/links`);
export const markJob = (jobId: string, m: { bfc: number; ffc: number; release: number; slow_motion?: number | null }) =>
  request<JobCreated>(`/jobs/${enc(jobId)}/mark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(m),
  });
export const reanalyze = (id: string, m: { bfc: number; ffc: number; release: number; slow_motion?: number | null }) =>
  request<JobCreated>(`/analysis/${enc(id)}/reanalyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(m),
  });
export const recompute = (id: string, body: Overrides & { clear?: boolean }) =>
  request<AnalysisResult>(`/analysis/${enc(id)}/recompute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
export const getDebugJson = (id: string) => request<Record<string, unknown>>(`/analysis/${enc(id)}/debug/context.json`);
export const getDebugPose3d = (id: string) => request<import("@/components/Skeleton3D").Pose3D>(`/analysis/${enc(id)}/debug/pose3d.json`);
export const setSync = (id: string, offsetS: number | null) =>
  request<SyncInfo>(`/analysis/${enc(id)}/sync`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ offset_s: offsetS }),
  });
export const getLinks = (id: string) => request<MediaLinks>(`/analysis/${enc(id)}/links`);
export const deleteAnalysis = (id: string) => request<{ deleted: string }>(`/analysis/${enc(id)}`, { method: "DELETE" });
export const sendFeedback = (id: string, findingRank: number, helpful: boolean) =>
  request<{ ok: boolean }>(`/analysis/${enc(id)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ finding_rank: findingRank, helpful }),
  });
export const getDeliveries = (bowler?: string) => request<DeliverySummary[]>(`/deliveries${bowler ? `?bowler=${enc(bowler)}` : ""}`);
export const getBowlers = () => request<BowlerSummary[]>("/bowlers");
export const getHistory = (name: string) => request<BowlerHistory>(`/bowlers/${enc(name)}/history`);
export const getSummary = (name: string, last = 10) => request<BowlerSummaryOut>(`/bowlers/${enc(name)}/summary?last=${last}`);
export const compare = (a: string, b: string) => request<CompareResponse>(`/compare?a=${enc(a)}&b=${enc(b)}`);
export const getHealth = () => request<HealthResponse>("/health", { noRedirect: true });
export const getBenchmarks = () => request<Record<string, BenchmarkRule>>("/benchmarks");
export const getSamples = () => request<SampleClip[]>("/samples");

export async function downloadSample(name: string): Promise<File> {
  let res: Response;
  try {
    res = await fetch(`${V1}/samples/${enc(name)}`, { cache: "no-store", headers: authHeaders() });
  } catch {
    throw new ApiError(NETWORK_ERROR);
  }
  if (res.status === 401) onUnauthorized();
  if (!res.ok) throw await parseError(res);
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type || "video/mp4" });
}

// Media URLs carry a short-lived signed query (from getLinks) because <video>/<img>/links can't send headers.
const withQ = (url: string, q: string) => `${url}${url.includes("?") ? "&" : "?"}${q}`;
// `v` (the result's recomputed_at) changes the URL after a correction, so the browser doesn't reuse a cached copy
const ver = (v?: string | null) => (v ? `v=${enc(v)}&` : "");
export const videoUrl = (id: string, q: string, clean = false, v?: string | null) =>
  withQ(`${V1}/analysis/${enc(id)}/video${clean ? "?clean=true" : ""}`, ver(v) + q);
export const debugUrl = (id: string, name: string, q: string, v?: string | null) => withQ(`${V1}/analysis/${enc(id)}/debug/${name}`, ver(v) + q);
export const personUrl = (id: string, cid: number, q: string) => withQ(`${V1}/analysis/${enc(id)}/person/${cid}`, q);
export const jobVideoUrl = (jobId: string, q: string) => withQ(`${V1}/jobs/${enc(jobId)}/video`, q);
export const frontVideoUrl = (id: string, q: string) => withQ(`${V1}/analysis/${enc(id)}/video?view=front`, q);
export const frameUrl = (id: string, n: number, q: string, joint?: string | null, second = false, v?: string | null) => {
  const qs = [joint ? `joint=${enc(joint)}` : "", second ? "view=front" : "", v ? `v=${enc(v)}` : ""].filter(Boolean).join("&");
  return withQ(`${V1}/analysis/${enc(id)}/frame/${n}${qs ? `?${qs}` : ""}`, q);
};
export const pdfUrl = (id: string, q: string) => withQ(`${V1}/analysis/${enc(id)}/pdf`, q);
