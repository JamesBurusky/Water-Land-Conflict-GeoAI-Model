import { useMapLayersStore, type PointSymbology, type RiskSymbology } from "../store/useMapLayersStore";

export function LayersPanel() {
  const {
    showRiskLayer, showConflictPoints, showHeatmap, pointSymbology, riskSymbology,
    setShowRiskLayer, setShowConflictPoints, setShowHeatmap, setPointSymbology, setRiskSymbology,
  } = useMapLayersStore();

  return (
    <div className="layers-panel">
      <div className="layers-section">
        <span className="layers-section-title">Layers</span>
        <label className="checkbox-row">
          <input type="checkbox" checked={showRiskLayer} onChange={(e) => setShowRiskLayer(e.target.checked)} />
          Risk prediction (choropleth)
        </label>
        <label className="checkbox-row">
          <input type="checkbox" checked={showConflictPoints} onChange={(e) => setShowConflictPoints(e.target.checked)} />
          Conflict records (points)
        </label>
        <label className="checkbox-row">
          <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />
          Conflict density heatmap
        </label>
      </div>

      <div className="layers-section">
        <span className="layers-section-title">Symbolize risk layer by</span>
        <select value={riskSymbology} onChange={(e) => setRiskSymbology(e.target.value as RiskSymbology)}>
          <option value="category">Risk category (Low/Medium/High)</option>
          <option value="probability">Risk probability (continuous)</option>
        </select>
      </div>

      <div className="layers-section">
        <span className="layers-section-title">Symbolize conflict points by</span>
        <select value={pointSymbology} onChange={(e) => setPointSymbology(e.target.value as PointSymbology)}>
          <option value="none">Uniform color</option>
          <option value="severity">Severity score</option>
          <option value="sentiment">Sentiment</option>
          <option value="topic">Conflict theme</option>
          <option value="composite">Chronic / discrete</option>
        </select>
      </div>
    </div>
  );
}
