// The cross-filtering state. THIS is what makes "click a sub-county on
// the map -> every chart updates" and "adjust the year range -> the map
// recolors" work: every component reads its filter values from here
// (via useFilterStore) instead of managing its own local state, so
// there's exactly one source of truth. Any component can also WRITE to
// this store (e.g. the map's click handler calls setSubcounty()), and
// every other component subscribed to that field re-renders automatically.

import { create } from "zustand";

interface FilterState {
  county: string | undefined;
  subcounty: string | undefined;
  yearMin: number | undefined;
  yearMax: number | undefined;
  topicId: number | undefined;
  domain: string | undefined; // Land-only / Water-only / Mixed / Neither
  search: string;

  setCounty: (county: string | undefined) => void;
  setSubcounty: (subcounty: string | undefined) => void;
  setYearRange: (min: number | undefined, max: number | undefined) => void;
  setTopicId: (topicId: number | undefined) => void;
  setDomain: (domain: string | undefined) => void;
  setSearch: (search: string) => void;
  reset: () => void;
}

export const useFilterStore = create<FilterState>((set) => ({
  county: undefined,
  subcounty: undefined,
  yearMin: undefined,
  yearMax: undefined,
  topicId: undefined,
  domain: undefined,
  search: "",

  // Selecting a county clears any sub-county selection from a
  // DIFFERENT county -- prevents an impossible filter state like
  // county=Kiambu + subcounty=Turkana South silently returning nothing
  // with no indication why.
  setCounty: (county) => set((state) => ({
    county,
    subcounty: county === state.county ? state.subcounty : undefined,
  })),
  setSubcounty: (subcounty) => set({ subcounty }),
  setYearRange: (yearMin, yearMax) => set({ yearMin, yearMax }),
  setTopicId: (topicId) => set({ topicId }),
  setDomain: (domain) => set({ domain }),
  setSearch: (search) => set({ search }),
  reset: () => set({
    county: undefined, subcounty: undefined, yearMin: undefined,
    yearMax: undefined, topicId: undefined, domain: undefined, search: "",
  }),
}));
