import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from "recharts";
import { useShapImportance } from "../hooks/useApiData";

export function ShapPanel() {
  const { data, isLoading } = useShapImportance();

  if (isLoading) return <p>Loading…</p>;
  if (!data) {
    return <p className="empty-note">Not yet available — run 11_shap_interpretability.py in the pipeline.</p>;
  }

  const sorted = [...data].sort((a, b) => a.mean_abs_shap - b.mean_abs_shap);

  return (
    <>
      <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 40)}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="feature" width={160} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(value) => (typeof value === "number" ? value.toFixed(4) : String(value ?? ""))} />
          <Bar dataKey="mean_abs_shap" name="Mean |SHAP|">
            {sorted.map((row, i) => (
              <Cell key={i} fill={row.mean_signed_shap >= 0 ? "#e76f51" : "#2a9d8f"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="chart-note">
        Red = higher feature value increases predicted risk. Teal = higher value decreases it.
      </p>
    </>
  );
}
