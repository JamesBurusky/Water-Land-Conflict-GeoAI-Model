// Which forecast horizon (3 through 24 months ahead, in 3-month steps)
// is currently being viewed -- drives the map's risk layer AND the
// model performance panels simultaneously, so switching horizons
// updates predictions and model metrics together, consistently.

import { create } from "zustand";

export const AVAILABLE_HORIZONS = [3, 6, 9, 12, 15, 18, 21, 24] as const;
export type HorizonMonths = (typeof AVAILABLE_HORIZONS)[number];

interface HorizonState {
  horizonMonths: HorizonMonths;
  setHorizonMonths: (h: HorizonMonths) => void;
}

export const useHorizonStore = create<HorizonState>((set) => ({
  horizonMonths: 3,
  setHorizonMonths: (horizonMonths) => set({ horizonMonths }),
}));
