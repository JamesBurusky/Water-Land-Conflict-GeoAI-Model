import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line,
} from "recharts";
import { useModelMetrics, useConfusionMatrices, useCalibration } from "../hooks/useApiData";
import { useHorizonStore } from "../store/useHorizonStore";

const MODEL_COLORS: Record<string, string> = {
  LogisticRegression: "#264653",
  RandomForest: "#2a9d8f",
  XGBoost: "#e76f51",
};

function ConfusionMatrixTable() {
  const { data, isLoading } = useConfusionMatrices();
  if (isLoading) return null;
  if (!data) return <p className="empty-note">Confusion matrices not available for this horizon.</p>;

  return (
    <table className="confusion-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>True Neg.</th>
          <th>False Pos.</th>
          <th>False Neg.</th>
          <th>True Pos.</th>
        </tr>
      </thead>
      <tbody>
        {data.map((row) => (
          <tr key={row.model}>
            <td style={{ color: MODEL_COLORS[row.model] ?? "#333" }}>{row.model}</td>
            <td>{row.true_negative}</td>
            <td>{row.false_positive}</td>
            <td>{row.false_negative}</td>
            <td>{row.true_positive}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CalibrationChart() {
  const { data, isLoading } = useCalibration();
  if (isLoading) return null;
  if (!data || data.length === 0) return null;

  const models = Array.from(new Set(data.map((d) => d.model)));
  const byX = new Map<number, Record<string, number>>();
  for (const row of data) {
    const key = Math.round(row.mean_predicted_probability * 100) / 100;
    const existing = byX.get(key) ?? { x: key };
    existing[row.model] = row.fraction_of_positives;
    byX.set(key, existing);
  }
  const chartData = Array.from(byX.values()).sort((a, b) => a.x - b.x);

  return (
    <div>
      <h4 className="submetric-title">Calibration — is predicted risk actually observed?</h4>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="x" domain={[0, 1]} label={{ value: "Predicted probability", position: "insideBottom", offset: -5, fontSize: 11 }} />
          <YAxis domain={[0, 1]} label={{ value: "Observed fraction positive", angle: -90, position: "insideLeft", fontSize: 11 }} />
          <Tooltip />
          <Legend />
          {models.map((m) => (
            <Line key={m} type="monotone" dataKey={m} stroke={MODEL_COLORS[m] ?? "#999"} dot strokeWidth={2} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <p className="chart-note">
        Points below the diagonal mean the model is overconfident — predicted probability higher
        than what's actually observed. A "70% risk" figure on the map is only literally accurate
        if this line tracks the diagonal.
      </p>
    </div>
  );
}

export function ModelMetricsPanel() {
  const { data, isLoading } = useModelMetrics();
  const horizonMonths = useHorizonStore((s) => s.horizonMonths);

  if (isLoading) return <p>Loading…</p>;
  if (!data) {
    return (
      <p className="empty-note">
        No model trained yet for the {horizonMonths}-month horizon — run 10_ml_modeling.py.
      </p>
    );
  }

  return (
    <div>
      <p className="chart-note">
        Baseline (Logistic Regression), comparison (Random Forest), and final (XGBoost) models —
        {" "}{horizonMonths}-month-ahead horizon.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="model" tick={{ fontSize: 11 }} />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Legend />
          <Bar dataKey="precision" fill="#264653" />
          <Bar dataKey="recall" fill="#2a9d8f" />
          <Bar dataKey="f1" fill="#e9c46a" />
          <Bar dataKey="roc_auc" name="ROC-AUC" fill="#e76f51" />
        </BarChart>
      </ResponsiveContainer>

      <h4 className="submetric-title">Confusion matrices</h4>
      <ConfusionMatrixTable />

      <CalibrationChart />
    </div>
  );
}
