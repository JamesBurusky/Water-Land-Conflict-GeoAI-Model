import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useConflictsForAggregation } from "../hooks/useApiData";
import { aggregateByMonth } from "../utils/aggregate";

export function SeasonalChart() {
  const { data: conflicts, isLoading } = useConflictsForAggregation();

  if (isLoading) return <p>Loading…</p>;
  if (!conflicts || conflicts.length === 0) {
    return <p className="empty-note">No records match the current filters.</p>;
  }

  const data = aggregateByMonth(conflicts);

  return (
    <>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="monthLabel" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" name="Records" fill="#2a9d8f" />
        </BarChart>
      </ResponsiveContainer>
      <p className="chart-note">Reflects all active filters, including theme and search.</p>
    </>
  );
}
