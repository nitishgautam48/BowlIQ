"use client";
import { useEffect, useState } from "react";
import { getLinks } from "@/lib/api";

/** Signed query string for an analysis's media URLs, refreshed before it expires. */
export function useMediaLinks(id: string | null | undefined): string | null {
  const [q, setQ] = useState<string | null>(null);
  useEffect(() => {
    if (!id) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = () =>
      getLinks(id)
        .then((l) => {
          if (!alive) return;
          setQ(l.query);
          const ms = Math.max(60_000, (l.expires_at - 300) * 1000 - Date.now());
          timer = setTimeout(load, ms);
        })
        .catch(() => alive && setQ(null));
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, [id]);
  return q;
}
