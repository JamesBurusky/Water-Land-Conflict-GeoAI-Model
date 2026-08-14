// Thin fetch wrappers around the backend API. Every function here maps
// 1:1 to a route in dashboard/backend/app/routers/*.py -- if a backend
// route changes, this is the one file to update.
//
// VITE_API_BASE_URL is read at build time (see .env.example) -- in dev
// it defaults to the local FastAPI server; in production (Contabo) it's
// set to wherever nginx proxies /api to the backend container.

import type {
  RiskGeoJSON, HotspotRow, PanelRow, PanelSummary, AvailableFilters,
  ConflictRecord, ModelMetricRow, ShapRow, TopicInfoRow, YearlyTrendRow,
  SeasonalRow, HealthResponse, ConfusionMatrixRow, CalibrationRow,
  HorizonComparisonRow, LandWaterSeverityRow, LandWaterCountyRow, LandWaterYearlyRow,
  PipelineStatus,
} from "../types/api";

// In production this is set to "" (empty) at build time so requests
// go to the same origin nginx serves the app from, which proxies
// /api/* to the backend container -- no CORS, no separate domain
// needed. In local dev it's the full backend URL (see .env).
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  // Deliberately NOT using `new URL(path, BASE_URL)` here -- the URL
  // constructor requires an ABSOLUTE base, which throws in production
  // where BASE_URL is "" (same-origin relative requests proxied by
  // nginx). Plain string concatenation + URLSearchParams works for
  // both an absolute BASE_URL (local dev) and an empty one
  // (production), since fetch() itself resolves relative URLs
  // against the current page origin.
  const query = new URLSearchParams();
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== "") {
        query.set(key, String(value));
      }
    }
  }
  const queryString = query.toString();
  const url = `${BASE_URL}${path}${queryString ? `?${queryString}` : ""}`;

  const res = await fetch(url);
  if (!res.ok) {
    // 404 is a normal, expected state here (e.g. SHAP not run yet) --
    // callers check for this via ApiError.status rather than treating
    // every non-200 as a crash-worthy error.
    throw new ApiError(res.status, `${path} returned ${res.status}`);
  }
  return res.json();
}

export { ApiError };

export interface PanelFilterParams {
  county?: string;
  subcounty?: string;
  year_min?: number;
  year_max?: number;
  limit?: number;
  [key: string]: string | number | undefined;
}

export interface ConflictFilterParams extends PanelFilterParams {
  topic_id?: number;
  domain?: string;
  search?: string;
}

export const api = {
  health: () => get<HealthResponse>("/api/health"),
  status: () => get<PipelineStatus>("/api/status"),

  riskLayer: (horizonMonths: number) =>
    get<RiskGeoJSON>("/api/spatial/risk-layer", { horizon_months: horizonMonths }),
  hotspots: () => get<HotspotRow[]>("/api/spatial/hotspots"),

  panel: (params: PanelFilterParams) => get<PanelRow[]>("/api/panel", params),
  panelFilters: () => get<AvailableFilters>("/api/panel/filters"),
  panelSummary: (params: { county?: string; subcounty?: string; year_min?: number; year_max?: number }) =>
    get<PanelSummary>("/api/panel/summary", params),

  conflicts: (params: ConflictFilterParams) => get<ConflictRecord[]>("/api/conflicts", params),

  modelMetrics: (horizonMonths: number) =>
    get<ModelMetricRow[]>("/api/analytics/model-metrics", { horizon_months: horizonMonths }),
  confusionMatrices: (horizonMonths: number) =>
    get<ConfusionMatrixRow[]>("/api/analytics/confusion-matrices", { horizon_months: horizonMonths }),
  calibration: (horizonMonths: number) =>
    get<CalibrationRow[]>("/api/analytics/calibration", { horizon_months: horizonMonths }),
  horizonComparison: () => get<HorizonComparisonRow[]>("/api/analytics/horizon-comparison"),

  shapImportance: () => get<ShapRow[]>("/api/analytics/shap-importance"),
  topics: () => get<TopicInfoRow[]>("/api/analytics/topics"),
  yearlyTrend: () => get<YearlyTrendRow[]>("/api/analytics/yearly-trend"),
  seasonalPattern: () => get<SeasonalRow[]>("/api/analytics/seasonal-pattern"),

  landWaterSeverity: () => get<LandWaterSeverityRow[]>("/api/analytics/land-water-severity"),
  landWaterByCounty: () => get<LandWaterCountyRow[]>("/api/analytics/land-water-by-county"),
  landWaterYearlyTrend: () => get<LandWaterYearlyRow[]>("/api/analytics/land-water-yearly-trend"),
};
