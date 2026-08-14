// Controls which floating widget panels are visible over the map. This
// is deliberately separate from useFilterStore (filter VALUES vs. UI
// panel VISIBILITY are different concerns) -- toggling a widget open/
// closed should never affect what data is being shown, only whether
// its panel is currently rendered on screen.

import { create } from "zustand";

export type WidgetId =
  | "filters" | "layers" | "summary" | "trend" | "seasonal"
  | "topics" | "modelMetrics" | "horizonComparison" | "shap"
  | "wordCloud" | "conflictList" | "landWater" | "status";

interface WidgetState {
  visible: Record<WidgetId, boolean>;
  toggle: (id: WidgetId) => void;
}

// Filters and layers open by default -- everything else the user
// opens deliberately, so the map isn't cluttered on first load.
const DEFAULT_VISIBLE: Record<WidgetId, boolean> = {
  filters: true,
  layers: true,
  summary: true,
  trend: false,
  seasonal: false,
  topics: false,
  modelMetrics: false,
  horizonComparison: false,
  shap: false,
  wordCloud: false,
  conflictList: false,
  landWater: false,
  status: false,
};

export const useWidgetStore = create<WidgetState>((set) => ({
  visible: DEFAULT_VISIBLE,
  toggle: (id) => set((state) => ({ visible: { ...state.visible, [id]: !state.visible[id] } })),
}));
