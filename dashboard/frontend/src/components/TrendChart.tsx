import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useConflictsForAggregation } from "../hooks/useApiData";
import { aggregateByYear } from "../utils/aggregate";

export function TrendChart() {
  const { data: conflicts, isLoading } = useConflictsForAggregation();

  if (isLoading) return <p>Loading…</p>;
  if (!conflicts || conflicts.length === 0) {
    return <p className="empty-note">No records match the current filters.</p>;
  }

  const data = aggregateByYear(conflicts);

  return (
    <>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="year" />
          <YAxis yAxisId="left" allowDecimals={false} label={{ value: "Record count", angle: -90, position: "insideLeft" }} />
          <YAxis yAxisId="right" orientation="right" label={{ value: "Avg. severity", angle: 90, position: "insideRight" }} />
          <Tooltip />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="count" name="Records" stroke="#e76f51" strokeWidth={2} dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="avgSeverity" name="Avg. severity" stroke="#264653" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <p className="chart-note">Reflects all active filters, including theme and search.</p>
    </>
  );
}
