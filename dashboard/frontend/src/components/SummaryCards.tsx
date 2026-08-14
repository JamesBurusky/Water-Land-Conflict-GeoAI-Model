import { useMemo } from "react";
import { usePanelSummary, useConflictsForAggregation } from "../hooks/useApiData";
import { useFilterStore } from "../store/useFilterStore";

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="kpi-card">
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
    </div>
  );
}

export function SummaryCards() {
  const { data: summary, isLoading: summaryLoading } = usePanelSummary();
  const { data: conflicts, isLoading: conflictsLoading } = useConflictsForAggregation();
  const { county, subcounty } = useFilterStore();

  // Conflict-record-derived stats respond to EVERY filter (including
  // topic and search) -- the environmental stats below (NDVI, rainfall)
  // come from the panel, which doesn't have a topic/search dimension,
  // so those two only reflect region/date filters. This split is a
  // real, structural distinction worth understanding, not an oversight.
  const conflictStats = useMemo(() => {
    if (!conflicts) return null;
    const compositeCount = conflicts.filter((c) => c.Is_Composite).length;
    const severities = conflicts.map((c) => c.severity_score).filter((s): s is number => typeof s === "number");
    const avgSeverity = severities.length > 0 ? severities.reduce((a, b) => a + b, 0) / severities.length : null;
    return { count: conflicts.length, compositeCount, avgSeverity };
  }, [conflicts]);

  if (summaryLoading || conflictsLoading) {
    return <div className="kpi-row kpi-row-empty">Loading…</div>;
  }

  return (
    <div className="kpi-row">
      <Card label={subcounty ? subcounty : county ? county : "All regions"} value="Selection" />
      <Card label="Matching records" value={conflictStats ? String(conflictStats.count) : "—"} />
      <Card label="Chronic/composite" value={conflictStats ? String(conflictStats.compositeCount) : "—"} />
      <Card
        label="Avg. severity"
        value={conflictStats?.avgSeverity != null ? conflictStats.avgSeverity.toFixed(2) : "—"}
      />
      <Card
        label="Avg. NDVI"
        value={summary?.avg_ndvi != null ? summary.avg_ndvi.toFixed(3) : "—"}
      />
      <Card
        label="Avg. rainfall anomaly"
        value={summary?.avg_rainfall_anomaly_pct != null ? `${summary.avg_rainfall_anomaly_pct.toFixed(1)}%` : "—"}
      />
    </div>
  );
}
