"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { analyze, ApiError, downloadSample, getSamples, getToken, toApiError } from "@/lib/api";

// The generated sample figure is drawn at 1.80 m, so 180 cm is its true height (enables metres and ball speed).
const SAMPLE_HEIGHT_CM = 180;
// Samples are saved under a reserved bowler so they never enter your own history or personal baseline.
export const SAMPLE_BOWLER = "Sample (synthetic)";

/** Downloads the sample clip from backend/samples/ and uploads it through the normal /api/analyze path. */
export function useSampleAnalysis() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const run = async () => {
    if (!getToken()) {
      router.push("/login?next=/");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const list = await getSamples();
      if (!list.length) {
        throw new ApiError({ code: "NO_SAMPLE", message: "No sample clip was found in backend/samples/.",
          fix: "Run `python scripts/make_sample.py` in the backend folder, then try again." });
      }
      const file = await downloadSample(list[0].name);
      const job = await analyze({ sideVideo: file, bowlerName: SAMPLE_BOWLER, bowlerHeightCm: SAMPLE_HEIGHT_CM });
      router.push(`/processing/${job.job_id}`);
    } catch (e) {
      setError(toApiError(e));
      setBusy(false);
    }
  };
  return { run, busy, error };
}
