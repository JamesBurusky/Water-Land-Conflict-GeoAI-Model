import { usePanelFilters } from "../hooks/useApiData";
import { useFilterStore } from "../store/useFilterStore";

// Search lives in the always-visible toolbar (App.tsx), not here, to
// avoid a confusing duplicate search box when this widget is open.
export function FilterSidebar() {
  const { data: filters, isLoading } = usePanelFilters();
  const {
    county, subcounty, yearMin, yearMax, topicId, domain,
    setCounty, setSubcounty, setYearRange, setTopicId, setDomain, reset,
  } = useFilterStore();

  if (isLoading) return <p>Loading filters…</p>;

  return (
    <>
      <label className="field">
        <span>County</span>
        <select value={county ?? ""} onChange={(e) => setCounty(e.target.value || undefined)}>
          <option value="">All counties</option>
          {filters?.counties.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Sub-county</span>
        <select value={subcounty ?? ""} onChange={(e) => setSubcounty(e.target.value || undefined)}>
          <option value="">All sub-counties</option>
          {filters?.subcounties.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </label>

      {filters?.year_min != null && filters?.year_max != null && (
        <div className="field">
          <span>Year range</span>
          <div className="year-range-row">
            <input
              type="number"
              min={filters.year_min}
              max={yearMax ?? filters.year_max}
              value={yearMin ?? filters.year_min}
              onChange={(e) => setYearRange(Number(e.target.value), yearMax)}
            />
            <span>to</span>
            <input
              type="number"
              min={yearMin ?? filters.year_min}
              max={filters.year_max}
              value={yearMax ?? filters.year_max}
              onChange={(e) => setYearRange(yearMin, Number(e.target.value))}
            />
          </div>
          {filters.n_unparseable_dates > 0 && (
            <span className="field-note">
              {filters.n_unparseable_dates} record{filters.n_unparseable_dates === 1 ? "" : "s"} in the
              source data {filters.n_unparseable_dates === 1 ? "has" : "have"} a date that couldn't be
              read and {filters.n_unparseable_dates === 1 ? "is" : "are"} excluded whenever a year range
              is set.
            </span>
          )}
        </div>
      )}

      {filters?.topics && filters.topics.length > 0 && (
        <label className="field">
          <span>Conflict theme</span>
          <select
            value={topicId ?? ""}
            onChange={(e) => setTopicId(e.target.value === "" ? undefined : Number(e.target.value))}
          >
            <option value="">All themes</option>
            {filters.topics.map((t) => (
              <option key={t.Topic} value={t.Topic}>{t.Name}</option>
            ))}
          </select>
        </label>
      )}

      {filters?.domains && filters.domains.length > 0 && (
        <label className="field">
          <span>Land/water domain</span>
          <select value={domain ?? ""} onChange={(e) => setDomain(e.target.value || undefined)}>
            <option value="">All domains</option>
            {filters.domains.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
      )}

      <button className="reset-button" onClick={reset}>Reset all filters</button>
    </>
  );
}
