import { Suspense, lazy } from "react";
import { Widget } from "./components/Widget";
import { FilterSidebar } from "./components/FilterSidebar";
import { LayersPanel } from "./components/LayersPanel";
import { SummaryCards } from "./components/SummaryCards";
import { TrendChart } from "./components/TrendChart";
import { SeasonalChart } from "./components/SeasonalChart";
import { ModelMetricsPanel } from "./components/ModelMetricsPanel";
import { HorizonComparisonPanel } from "./components/HorizonComparisonPanel";
import { ShapPanel } from "./components/ShapPanel";
import { TopicsPanel } from "./components/TopicsPanel";
import { ConflictList } from "./components/ConflictList";
import { WordCloud } from "./components/WordCloud";
import { LandWaterPanel } from "./components/LandWaterPanel";
import { StatusPanel } from "./components/StatusPanel";
import { HorizonSelector } from "./components/HorizonSelector";
import { HealthBanner } from "./components/HealthBanner";
import { useWidgetStore, type WidgetId } from "./store/useWidgetStore";
import { useFilterStore } from "./store/useFilterStore";
import { useSelectionStore } from "./store/useSelectionStore";

// MapLibre-sized dependency avoidance -- OpenLayers is much lighter
// than the old MapLibre setup, but the map is still by far the
// largest single chunk, so it stays lazy-loaded.
const MapView = lazy(() => import("./components/MapView").then((m) => ({ default: m.MapView })));

const TOOLBAR_BUTTONS: Array<{ id: WidgetId; label: string }> = [
  { id: "filters", label: "Filters" },
  { id: "layers", label: "Layers" },
  { id: "status", label: "Pipeline Status" },
  { id: "summary", label: "Summary" },
  { id: "trend", label: "Trend" },
  { id: "seasonal", label: "Seasonal" },
  { id: "topics", label: "Themes" },
  { id: "landWater", label: "Land vs Water" },
  { id: "wordCloud", label: "Word Cloud" },
  { id: "conflictList", label: "Records" },
  { id: "modelMetrics", label: "Model" },
  { id: "horizonComparison", label: "Reliability Over Time" },
  { id: "shap", label: "SHAP" },
];

function App() {
  const visible = useWidgetStore((s) => s.visible);
  const toggle = useWidgetStore((s) => s.toggle);
  const { search, setSearch } = useFilterStore();

  const leftDrawerOpen = visible.filters || visible.layers || visible.status;
  const rightDrawerOpen =
    visible.summary || visible.trend || visible.seasonal || visible.topics ||
    visible.landWater || visible.wordCloud || visible.conflictList ||
    visible.modelMetrics || visible.horizonComparison || visible.shap;

  return (
    <div className="app-fixed-shell">
      <Suspense fallback={<div className="map-fullscreen-fallback">Loading map…</div>}>
        <MapView />
      </Suspense>

      <header className="app-toolbar">
        <div className="app-toolbar-title">
          <h1>Kenya Water–Land Conflict Dashboard</h1>
        </div>

        <input
          type="text"
          className="app-toolbar-search"
          placeholder="Search by ID, location, or keyword (e.g. Mathare)…"
          title="Searches: Record ID, Sub-location, Incident Summary, Full Description, and NLP Keywords"
          value={search}
          onChange={(e) => { setSearch(e.target.value); useSelectionStore.getState().clearPin(); }}
        />

        <div className="app-toolbar-buttons">
          {TOOLBAR_BUTTONS.map((b) => (
            <button
              key={b.id}
              className={visible[b.id] ? "toolbar-btn toolbar-btn-active" : "toolbar-btn"}
              onClick={() => toggle(b.id)}
            >
              {b.label}
            </button>
          ))}
        </div>

        <HorizonSelector />
      </header>

      <div className="app-health-overlay">
        <HealthBanner />
      </div>

      {leftDrawerOpen && (
        <div className="drawer drawer-left">
          {visible.filters && (
            <Widget id="filters" title="Filters">
              <FilterSidebar />
            </Widget>
          )}
          {visible.layers && (
            <Widget id="layers" title="Layers & Symbology">
              <LayersPanel />
            </Widget>
          )}
          {visible.status && (
            <Widget id="status" title="Pipeline Status">
              <StatusPanel />
            </Widget>
          )}
        </div>
      )}

      {rightDrawerOpen && (
        <div className="drawer drawer-right">
          {visible.summary && (
            <Widget id="summary" title="Summary">
              <SummaryCards />
            </Widget>
          )}
          {visible.trend && (
            <Widget id="trend" title="Yearly Trend">
              <TrendChart />
            </Widget>
          )}
          {visible.seasonal && (
            <Widget id="seasonal" title="Seasonal Pattern">
              <SeasonalChart />
            </Widget>
          )}
          {visible.topics && (
            <Widget id="topics" title="Conflict Themes">
              <TopicsPanel />
            </Widget>
          )}
          {visible.landWater && (
            <Widget id="landWater" title="Land vs Water Relationship">
              <LandWaterPanel />
            </Widget>
          )}
          {visible.wordCloud && (
            <Widget id="wordCloud" title="Word Cloud">
              <WordCloud />
            </Widget>
          )}
          {visible.conflictList && (
            <Widget id="conflictList" title="Conflict Records">
              <ConflictList />
            </Widget>
          )}
          {visible.modelMetrics && (
            <Widget id="modelMetrics" title="Model Performance">
              <ModelMetricsPanel />
            </Widget>
          )}
          {visible.horizonComparison && (
            <Widget id="horizonComparison" title="Model Reliability Over Time">
              <p className="chart-note">
                How well the model predicts as the forecast window lengthens — 3 to 24 months
                ahead, in 3-month steps. Use the "Forecast horizon" control in the top toolbar to
                switch which of these predictions is currently shown on the map.
              </p>
              <HorizonComparisonPanel />
            </Widget>
          )}
          {visible.shap && (
            <Widget id="shap" title="Feature Importance (SHAP)">
              <ShapPanel />
            </Widget>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
