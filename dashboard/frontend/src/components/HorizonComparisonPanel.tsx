import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useHorizonComparison } from "../hooks/useApiData";

const MODEL_COLORS: Record<string, string> = {
  LogisticRegression: "#264653",
  RandomForest: "#2a9d8f",
  XGBoost: "#e76f51",
};

export function HorizonComparisonPanel() {
  const { data, isLoading } = useHorizonComparison();

  if (isLoading) return <p>Loading…</p>;
  if (!data || data.length === 0) {
    return <p className="empty-note">Not yet available — run 10_ml_modeling.py (produces all horizons at once).</p>;
  }

  const models = Array.from(new Set(data.map((d) => d.model)));
  const byHorizon = new Map<number, Record<string, number>>();
  for (const row of data) {
    if (row.roc_auc == null) continue; // undefined for a horizon whose test set had only one class
    const existing = byHorizon.get(row.horizon_months) ?? { horizon: row.horizon_months };
    existing[row.model] = row.roc_auc;
    byHorizon.set(row.horizon_months, existing);
  }
  const chartData = Array.from(byHorizon.values()).sort((a, b) => a.horizon - b.horizon);

  return (
    <div>
      <p className="chart-note">ROC-AUC per model across every trained forecast horizon.</p>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="horizon" label={{ value: "Forecast horizon (months)", position: "insideBottom", offset: -5, fontSize: 11 }} />
          <YAxis domain={[0, 1]} label={{ value: "ROC-AUC", angle: -90, position: "insideLeft", fontSize: 11 }} />
          <Tooltip />
          <Legend />
          {models.map((m) => (
            <Line key={m} type="monotone" dataKey={m} stroke={MODEL_COLORS[m] ?? "#999"} strokeWidth={2} dot />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <p className="chart-note">
        A falling line as the horizon lengthens is expected — forecasting further ahead is
        genuinely harder. If a longer horizon performs BETTER, that's worth investigating rather
        than reporting at face value.
      </p>
    </div>
  );
}
