// Separate from useFilterStore (which conflicts MATCH) -- this is a
// one-off "fly the map to this specific point" trigger, e.g. clicking
// a record in the Conflict Records widget. Cleared after the map
// consumes it isn't necessary -- MapView just re-flies whenever this
// changes, including to the same record clicked twice (a fresh click
// on the same record should still re-center/re-zoom, which is the
// behavior a user would expect).
//
// pinnedRecordId is DELIBERATELY separate from flyToRecordId: both get
// set when a MAP POINT is clicked (so the records list narrows to just
// that one record), but clicking a record in the LIST only sets
// flyToRecordId (to fly/highlight on the map) -- NOT pinnedRecordId.
// If clicking a list item also pinned it, the list would immediately
// collapse to just the item you clicked with no easy way back, which
// is a confusing loop, not the "click a map point to filter" behavior
// that was actually asked for.

import { create } from "zustand";

interface SelectionState {
  flyToRecordId: string | null;
  flyToCoordinate: [number, number] | null; // [lon, lat]
  sequence: number; // increments on every selection, even a repeat click of the same record
  pinnedRecordId: string | null; // set only by a map point click; filters the records list to just this one
  selectConflict: (recordId: string, lon: number, lat: number) => void;
  pinFromMap: (recordId: string, lon: number, lat: number) => void;
  clearPin: () => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  flyToRecordId: null,
  flyToCoordinate: null,
  sequence: 0,
  pinnedRecordId: null,
  selectConflict: (recordId, lon, lat) => set((state) => ({
    flyToRecordId: recordId, flyToCoordinate: [lon, lat], sequence: state.sequence + 1,
  })),
  pinFromMap: (recordId, lon, lat) => set((state) => ({
    flyToRecordId: recordId, flyToCoordinate: [lon, lat], sequence: state.sequence + 1,
    pinnedRecordId: recordId,
  })),
  clearPin: () => set({ pinnedRecordId: null }),
}));
