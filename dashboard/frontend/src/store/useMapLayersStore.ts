// Controls what the map actually renders: which layers are on, and how
// features are symbolized. Separate from useFilterStore (WHICH data is
// shown) and useWidgetStore (WHICH panels are open) -- this is HOW the
// map data looks.

import { create } from "zustand";

export type PointSymbology = "none" | "severity" | "sentiment" | "topic" | "composite";
export type RiskSymbology = "category" | "probability";
export type RiskCategory = "Low" | "Medium" | "High" | "No data";

interface MapLayersState {
  showRiskLayer: boolean;
  showConflictPoints: boolean;
  showHeatmap: boolean;
  pointSymbology: PointSymbology;
  riskSymbology: RiskSymbology;
  // Which risk categories are hidden via the on-map legend's own toggle
  // checkboxes -- separate from showRiskLayer (that's the whole layer;
  // this is per-category within it, e.g. "hide all Low-risk sub-counties
  // but keep showing Medium/High").
  hiddenRiskCategories: Set<RiskCategory>;

  setShowRiskLayer: (v: boolean) => void;
  setShowConflictPoints: (v: boolean) => void;
  setShowHeatmap: (v: boolean) => void;
  setPointSymbology: (v: PointSymbology) => void;
  setRiskSymbology: (v: RiskSymbology) => void;
  toggleRiskCategory: (category: RiskCategory) => void;
}

export const useMapLayersStore = create<MapLayersState>((set) => ({
  showRiskLayer: true,
  showConflictPoints: true,
  showHeatmap: false,
  pointSymbology: "none",
  riskSymbology: "category",
  hiddenRiskCategories: new Set(),

  setShowRiskLayer: (v) => set({ showRiskLayer: v }),
  setShowConflictPoints: (v) => set({ showConflictPoints: v }),
  setShowHeatmap: (v) => set({ showHeatmap: v }),
  setPointSymbology: (v) => set({ pointSymbology: v }),
  setRiskSymbology: (v) => set({ riskSymbology: v }),
  toggleRiskCategory: (category) => set((state) => {
    const next = new Set(state.hiddenRiskCategories);
    if (next.has(category)) next.delete(category);
    else next.add(category);
    return { hiddenRiskCategories: next };
  }),
}));
