import { usePipelineStatus } from "../hooks/useApiData";

const STEP_LABELS: Record<string, string> = {
  "01_phase1_data_audit": "01 — Data audit",
  "02_conflict_cleaning": "02 — Conflict cleaning",
  "03_deduplication_check": "03 — Deduplication",
  "04_nlp_pipeline": "04 — NLP pipeline",
  "05_topic_modelling": "05 — Topic modelling (needs internet)",
  "06_spatial_feature_engineering": "06 — Spatial feature engineering",
  "07_topic_refit_temporal_safe": "07 — Temporal-safe topic refit (needs internet)",
  "08_nlp_panel_features": "08 — NLP panel features",
  "09_exploratory_spatial_analysis": "09 — Exploratory spatial analysis",
  "11_shap_interpretability": "11 — SHAP interpretability",
  "13_land_water_relationship_analysis": "13 — Land-water relationship analysis",
};

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={ok ? "status-dot status-dot-ok" : "status-dot status-dot-missing"} />;
}

export function StatusPanel() {
  const { data: status, isLoading, error } = usePipelineStatus();

  if (isLoading) return <p>Loading…</p>;
  if (error || !status) {
    return (
      <p className="empty-note">
        Can't reach the backend at all — check it's running and VITE_API_BASE_URL is correct.
      </p>
    );
  }

  if (!status.outputs_dir_exists) {
    return (
      <div>
        <p className="empty-note">{status.diagnostic}</p>
        <p className="chart-note">outputs_dir: <code>{status.outputs_dir}</code></p>
      </div>
    );
  }

  return (
    <div className="status-panel">
      <p className="chart-note">outputs_dir: <code>{status.outputs_dir}</code></p>

      <h4 className="submetric-title">Pipeline steps</h4>
      <ul className="status-list">
        {Object.entries(status.steps).map(([step, ok]) => (
          <li key={step} className="status-list-item">
            <StatusDot ok={ok} />
            {STEP_LABELS[step] ?? step}
          </li>
        ))}
      </ul>

      <h4 className="submetric-title">Model training (step 10) by horizon</h4>
      <ul className="status-list">
        {Object.entries(status.step_10_ml_modeling_horizons).map(([h, ok]) => (
          <li key={h} className="status-list-item">
            <StatusDot ok={ok} />
            {h}-month horizon
          </li>
        ))}
      </ul>

      <h4 className="submetric-title">Risk mapping (step 12) by horizon</h4>
      <ul className="status-list">
        {Object.entries(status.step_12_conflict_risk_mapping_horizons).map(([h, ok]) => (
          <li key={h} className="status-list-item">
            <StatusDot ok={ok} />
            {h}-month horizon
          </li>
        ))}
      </ul>

      <p className="chart-note">
        A red dot means that step's output file wasn't found where expected — run the
        corresponding script (see the pipeline README for exact commands), then reload.
      </p>
    </div>
  );
}
