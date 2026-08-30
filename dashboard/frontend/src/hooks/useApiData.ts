// Each hook here reads the relevant slice of useFilterStore (and, for
// horizon-aware endpoints, useHorizonStore) and passes it to the API
// client via React Query. Because the query key includes those values,
// changing a filter OR the selected forecast horizon anywhere in the
// app automatically triggers a re-fetch in every component using these
// hooks -- that's the actual mechanism behind "filter/horizon affects
// all visualizations".
//
// 404 responses (a phase script hasn't been run yet, e.g. no SHAP
// results, or a horizon that hasn't been trained) are treated as "no
// data available" rather than a query error -- retry: false +
// returning null on 404 keeps failed/optional panels from showing an
// error spinner forever.

import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import { useFilterStore } from "../store/useFilterStore";
import { useHorizonStore } from "../store/useHorizonStore";

function useOptionalQuery<T>(key: unknown[], fn: () => Promise<T>) {
  return useQuery({
    queryKey: key,
    queryFn: async () => {
      try {
        return await fn();
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
    retry: false,
  });
}

export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: api.health });
}

/** The full per-step, per-horizon pipeline status -- polled periodically
 * since the user may re-run pipeline scripts while the dashboard is open,
 * and this is what drives the StatusPanel widget and the HorizonSelector's
 * disabled/available state. */
export function usePipelineStatus() {
  return useQuery({
    queryKey: ["pipeline-status"],
    queryFn: api.status,
    refetchInterval: 30_000,
  });
}

/** The risk map layer for the CURRENTLY SELECTED forecast horizon --
 * switching horizons (via useHorizonStore) automatically refetches
 * this and updates the map. */
export function useRiskLayer() {
  const horizonMonths = useHorizonStore((s) => s.horizonMonths);
  return useOptionalQuery(["risk-layer", horizonMonths], () => api.riskLayer(horizonMonths));
}

export function useHotspots() {
  return useOptionalQuery(["hotspots"], api.hotspots);
}

export function usePanelFilters() {
  return useQuery({ queryKey: ["panel-filters"], queryFn: api.panelFilters });
}

/** The filtered panel time series -- powers the trend charts. Re-fetches
 * whenever county/subcounty/year range changes in the shared store. */
export function usePanel() {
  const { county, subcounty, yearMin, yearMax } = useFilterStore();
  return useQuery({
    queryKey: ["panel", county, subcounty, yearMin, yearMax],
    queryFn: () => api.panel({ county, subcounty, year_min: yearMin, year_max: yearMax, limit: 5000 }),
  });
}

export function usePanelSummary() {
  const { county, subcounty, yearMin, yearMax } = useFilterStore();
  return useQuery({
    queryKey: ["panel-summary", county, subcounty, yearMin, yearMax],
    queryFn: () => api.panelSummary({ county, subcounty, year_min: yearMin, year_max: yearMax }),
  });
}

/** Filtered + searched conflict records -- the map's point layer and
 * the incident list both use this, so a search term filters both at once.
 * Capped at 500 -- reasonable for a scrollable list UI. */
export function useConflicts() {
  const { county, subcounty, yearMin, yearMax, topicId, domain, search } = useFilterStore();
  return useQuery({
    queryKey: ["conflicts", county, subcounty, yearMin, yearMax, topicId, domain, search],
    queryFn: () => api.conflicts({
      county, subcounty, year_min: yearMin, year_max: yearMax,
      topic_id: topicId, domain, search: search || undefined, limit: 500,
    }),
  });
}

/** Same filtered query, but at the backend's max limit (5000) -- used
 * for trend/seasonal chart AGGREGATION, where capping at 500 would
 * silently undercount whenever more records match the current filters
 * than fit in a reasonable scrollable list. Filters still apply; only
 * the row limit differs from useConflicts() above. */
export function useConflictsForAggregation() {
  const { county, subcounty, yearMin, yearMax, topicId, domain, search } = useFilterStore();
  return useQuery({
    queryKey: ["conflicts-agg", county, subcounty, yearMin, yearMax, topicId, domain, search],
    queryFn: () => api.conflicts({
      county, subcounty, year_min: yearMin, year_max: yearMax,
      topic_id: topicId, domain, search: search || undefined, limit: 5000,
    }),
  });
}

/** Same as useConflictsForAggregation, but deliberately EXCLUDES the
 * topic filter -- used by the topics breakdown panel so that selecting
 * a theme doesn't collapse the panel's own list down to just itself
 * (respects every other active filter, so the breakdown still reflects
 * the current region/date/search selection). */
export function useConflictsForTopicBreakdown() {
  const { county, subcounty, yearMin, yearMax, domain, search } = useFilterStore();
  return useQuery({
    queryKey: ["conflicts-topic-breakdown", county, subcounty, yearMin, yearMax, domain, search],
    queryFn: () => api.conflicts({
      county, subcounty, year_min: yearMin, year_max: yearMax,
      domain, search: search || undefined, limit: 5000,
    }),
  });
}

// --- Horizon-aware model performance (Phase 6/7) ---

export function useModelMetrics() {
  const horizonMonths = useHorizonStore((s) => s.horizonMonths);
  return useOptionalQuery(["model-metrics", horizonMonths], () => api.modelMetrics(horizonMonths));
}

export function useConfusionMatrices() {
  const horizonMonths = useHorizonStore((s) => s.horizonMonths);
  return useOptionalQuery(["confusion-matrices", horizonMonths], () => api.confusionMatrices(horizonMonths));
}

export function useCalibration() {
  const horizonMonths = useHorizonStore((s) => s.horizonMonths);
  return useOptionalQuery(["calibration", horizonMonths], () => api.calibration(horizonMonths));
}

/** All horizons at once, for the "how does performance degrade as the
 * forecast window lengthens" comparison chart -- NOT horizon-scoped,
 * since the whole point is comparing across horizons. */
export function useHorizonComparison() {
  return useOptionalQuery(["horizon-comparison"], api.horizonComparison);
}

export function useShapImportance() {
  return useOptionalQuery(["shap-importance"], api.shapImportance);
}

export function useTopics() {
  return useOptionalQuery(["topics"], api.topics);
}

export function useYearlyTrend() {
  return useOptionalQuery(["yearly-trend"], api.yearlyTrend);
}

export function useSeasonalPattern() {
  return useOptionalQuery(["seasonal-pattern"], api.seasonalPattern);
}

// --- Land-water relationship analysis ---

export function useLandWaterSeverity() {
  return useOptionalQuery(["land-water-severity"], api.landWaterSeverity);
}

export function useLandWaterByCounty() {
  return useOptionalQuery(["land-water-by-county"], api.landWaterByCounty);
}

export function useLandWaterYearlyTrend() {
  return useOptionalQuery(["land-water-yearly-trend"], api.landWaterYearlyTrend);
}
