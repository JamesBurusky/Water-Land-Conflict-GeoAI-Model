// Mirrors the shapes returned by the FastAPI backend (dashboard/backend/app/routers/*.py).
// Keeping these in one file makes it obvious when a backend response shape
// changes and the frontend needs updating to match.

import type { Geometry } from "geojson";

export interface RiskFeatureProperties {
  County: string;
  SubCounty: string;
  risk_probability: number | null;
  risk_category: "Low" | "Medium" | "High" | "Lower" | "Higher" | null;
  [key: string]: unknown; // panel columns carried through (population, NDVI, etc.)
}

export interface RiskGeoJSON {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    properties: RiskFeatureProperties;
    geometry: Geometry;
  }>;
}

export interface HotspotRow {
  County: string;
  SubCounty: string;
  gi_zscore: number;
  gi_pvalue: number;
  hotspot_category: string;
}

export interface PanelRow {
  County: string;
  SubCounty: string;
  panel_date: string;
  Year: number;
  Month: number;
  Population_Density_per_SqKm: number | null;
  Total_Population: number | null;
  total_abstraction_m3_per_day: number | null;
  wrua_count: number | null;
  Mean_NDVI: number | null;
  Rainfall_mm: number | null;
  Rainfall_Anomaly_mm: number | null;
  Rainfall_Anomaly_Percent: number | null;
  conflict_onset: boolean;
  conflict_persistence: number | null;
  conflict_severity_weighted: number | null;
  decayed_sentiment?: number | null;
  dominant_topic_id?: number | null;
  topic_diversity?: number | null;
}

export interface PanelSummary {
  total_months?: number;
  total_onset_months?: number | null;
  avg_persistence?: number | null;
  avg_ndvi?: number | null;
  avg_rainfall_anomaly_pct?: number | null;
}

export interface AvailableFilters {
  counties: string[];
  subcounties: string[];
  year_min: number | null;
  year_max: number | null;
  topics: Array<{ Topic: number; Name: string }>;
  domains: string[];
  horizons_months: number[];
  n_unparseable_dates: number;
}

export interface ConflictRecord {
  Record_ID: string;
  County: string;
  Sub_Location?: string;
  Date_Start?: string;
  Date_End?: string;
  Incident_Summary?: string;
  NLP_Keywords?: string;
  Casualties_Reported?: string;
  Displaced_Persons?: string;
  Is_Composite?: boolean;
  severity_score?: number | null;
  geo_county?: string | null;
  geo_subcounty?: string | null;
  topic_id?: number | null;
  topic_label?: string | null;
  sentiment_score?: number | null;
  sentiment_label?: string | null;
  conflict_domain?: string | null;
  matched_land_terms?: string | null;
  matched_water_terms?: string | null;
  [key: string]: unknown;
}

export interface ModelMetricRow {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  pr_auc: number | null;
}

export interface ShapRow {
  feature: string;
  mean_abs_shap: number;
  mean_signed_shap: number;
}

export interface TopicInfoRow {
  Topic: number;
  Count: number;
  Name: string;
}

export interface YearlyTrendRow {
  Year: number;
  conflict_onset: number;
}

export interface SeasonalRow {
  Month: number;
  conflict_onset_total: number;
  conflict_onset_mean: number;
  conflict_onset_std: number;
}

export interface PipelineStatus {
  outputs_dir: string;
  outputs_dir_exists: boolean;
  data_dir: string;
  steps: Record<string, boolean>;
  step_10_ml_modeling_horizons: Record<string, boolean>;
  step_12_conflict_risk_mapping_horizons: Record<string, boolean>;
  horizon_comparison_available: boolean;
  diagnostic?: string;
}

export interface HealthResponse {
  status: string;
  outputs_dir: string;
  forecast_horizons_months: number[];
  horizons_available: Record<string, boolean>;
  available: Record<string, boolean>;
}

export interface ConfusionMatrixRow {
  model: string;
  true_negative: number;
  false_positive: number;
  false_negative: number;
  true_positive: number;
  total: number;
}

export interface CalibrationRow {
  mean_predicted_probability: number;
  fraction_of_positives: number;
  bin_count: number;
  model: string;
}

export interface HorizonComparisonRow extends ModelMetricRow {
  horizon_months: number;
}

export interface LandWaterSeverityRow {
  conflict_domain: string;
  n_records: number;
  mean: number;
  median: number;
  std: number;
}

export interface LandWaterCountyRow {
  County: string;
  [domain: string]: string | number; // one column per domain (Mixed, Water-only, etc.)
}

export interface LandWaterYearlyRow {
  [key: string]: number; // first column is the year index, rest are domain counts
}
