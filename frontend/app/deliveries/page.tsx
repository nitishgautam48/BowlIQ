"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ConfPip, Empty, ErrorBox, SkeletonBlocks } from "@/components/Common";
import { Icon } from "@/components/Icons";
import { useToast } from "@/components/Toast";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { deleteAnalysis, getDeliveries, toApiError, type ApiError } from "@/lib/api";
import { fmtDate } from "@/lib/format";
import { useSettings } from "@/lib/settings";
import type { DeliverySummary } from "@/types/analysis";

export default function Deliveries() {
  const { settings, ready } = useSettings();
  const router = useRouter();
  const flash = useToast();
  const [rows, setRows] = useState<DeliverySummary[] | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [all, setAll] = useState(false);
  const [sel, setSel] = useState<string[]>([]);
  const [pending, setPending] = useState<DeliverySummary | null>(null);
  const load = useCallback(() => {
    if (!ready) return;
    setErr(null);
    setRows(null);
    getDeliveries(all ? undefined : settings.bowler).then(setRows).catch((e) => setErr(toApiError(e)));
  }, [all, settings.bowler, ready]);
  useEffect(load, [load]);
  const toggle = (id: string) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s.slice(-1), id]));
  const remove = async (d: DeliverySummary) => {
    setPending(null);
    try {
      await deleteAnalysis(d.id);
      setRows((r) => r?.filter((x) => x.id !== d.id) ?? null);
      setSel((s) => s.filter((x) => x !== d.id));
      flash("Delivery deleted");
    } catch (e) {
      flash(toApiError(e).message);
    }
  };
  return (
    <AppShell title="My deliveries">
      <div className="col" style={{ gap: 16 }}>
        <div className="spread">
          <div className="seg" role="group" aria-label="Whose deliveries">
            <button aria-pressed={!all} onClick={() => setAll(false)}>{settings.bowler}</button>
            <button aria-pressed={all} onClick={() => setAll(true)}>All bowlers</button>
          </div>
          <div className="row">
            <span className="small mu">{sel.length === 2 ? "Two selected" : "Select two to compare"}</span>
            <button className="btn sm" disabled={sel.length !== 2} onClick={() => router.push(`/compare?a=${sel[0]}&b=${sel[1]}`)}><Icon name="cols" size={16} /> Compare</button>
            <Link href="/new" className="btn sm primary"><Icon name="new" size={16} /> New analysis</Link>
          </div>
        </div>
        {err && <ErrorBox error={err} onRetry={load} title="We couldn't load your deliveries" />}
        {!rows && !err && <SkeletonBlocks blocks={[[12, 64], [12, 64], [12, 64], [12, 64]]} />}
        {rows && rows.length === 0 && (
          <Empty title="No deliveries yet" body="Upload a side-on clip of one delivery to get your first report. It takes about a minute.">
            <Link href="/new" className="btn primary">Start an analysis</Link>
          </Empty>
        )}
        {rows && rows.length > 0 && (
          <div className="card" style={{ padding: 8 }}>
            <div className="tablewrap">
              <table className="t">
                <thead><tr><th style={{ width: 36 }} /><th>Date</th>{all && <th>Bowler</th>}<th>Ball speed</th><th>Top fix</th><th>Quality</th><th /></tr></thead>
                <tbody>
                  {rows.map((d) => (
                    <tr key={d.id}>
                      <td><input type="checkbox" checked={sel.includes(d.id)} onChange={() => toggle(d.id)} aria-label={`Select delivery from ${fmtDate(d.created_at, true)}`} /></td>
                      <td><Link href={`/results/${d.id}`} style={{ color: "var(--tx)", fontWeight: 500 }}>{fmtDate(d.created_at, true)}</Link>{d.has_front && <span className="chip" style={{ marginLeft: 8 }}>+ front-on</span>}</td>
                      {all && <td className="small">{d.bowler_name}</td>}
                      <td className="mono">{d.ball_speed_kmh != null ? `${Math.round(d.ball_speed_kmh)} km/h` : <span className="mu small">—</span>}</td>
                      <td className="small">{d.top_finding ?? <span className="mu">No fixes flagged</span>}</td>
                      <td><ConfPip level={d.quality} /></td>
                      <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                        <Link href={`/results/${d.id}`} className="btn sm ghost">Open</Link>
                        <button className="btn sm ghost" onClick={() => setPending(d)} aria-label={`Delete delivery from ${fmtDate(d.created_at, true)}`}><Icon name="trash" size={15} /></button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
      {pending && <ConfirmDialog title="Delete this delivery?" danger confirmLabel="Delete"
        body={`The delivery from ${fmtDate(pending.created_at, true)} and its video and report files will be removed. This can't be undone.`}
        onConfirm={() => remove(pending)} onCancel={() => setPending(null)} />}
    </AppShell>
  );
}
