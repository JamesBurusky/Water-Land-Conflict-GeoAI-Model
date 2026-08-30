import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";
import { useLandWaterSeverity, useLandWaterByCounty } from "../hooks/useApiData";
import { useFilterStore } from "../store/useFilterStore";

const DOMAIN_COLORS: Record<string, string> = {
  "Land-only": "#f4a261",
  "Water-only": "#2a9d8f",
  Mixed: "#264653",
  Neither: "#dddddd",
};

export function LandWaterPanel() {
  const { data: severity, isLoading: severityLoading } = useLandWaterSeverity();
  const { data: byCounty, isLoading: countyLoading } = useLandWaterByCounty();
  const { domain, setDomain } = useFilterStore();

  if (severityLoading || countyLoading) return <p>Loading…</p>;
  if (!severity) {
    return (
      <p className="empty-note">
        Not yet available — run 13_land_water_relationship_analysis.py.
      </p>
    );
  }

  return (
    <div>
      <p className="chart-note">
        Every conflict record classified as Land-only / Water-only / Mixed / Neither, from a
        keyword heuristic (not ground truth — see the pipeline's src/land_water_analysis.py).
        Click a bar to filter the dashboard by that domain.
      </p>

      <h4 className="submetric-title">Mean severity by domain</h4>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={severity}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="conflict_domain" tick={{ fontSize: 11 }} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="mean" name="Mean severity" style={{ cursor: "pointer" }}>
            {severity.map((row) => (
              <Cell
                key={row.conflict_domain}
                fill={DOMAIN_COLORS[row.conflict_domain] ?? "#999"}
                opacity={!domain || domain === row.conflict_domain ? 1 : 0.35}
                onClick={() => setDomain(domain === row.conflict_domain ? undefined : row.conflict_domain)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {byCounty && byCounty.length > 0 && (
        <>
          <h4 className="submetric-title">Domain by county</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={byCounty}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="County" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {Object.keys(byCounty[0])
                .filter((k) => k !== "County")
                .map((d) => (
                  <Bar key={d} dataKey={d} stackId="a" fill={DOMAIN_COLORS[d] ?? "#999"} />
                ))}
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}
