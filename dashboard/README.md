# GeoAI Conflict Dashboard

Interactive dashboard for the water–land conflict prediction pipeline —
zoomable risk map, cross-filtering (map ↔ charts ↔ search), and every
pipeline output (Phases 1–8) in one place.

## Structure

```
dashboard/
├── backend/          FastAPI — serves pipeline outputs as a REST API
│   └── README.md      local dev instructions
├── frontend/          React + TypeScript + Vite — the actual UI
│   └── README.md      local dev instructions
├── docker-compose.yml  brings up both for production (Contabo)
└── DEPLOYMENT.md       step-by-step Contabo deployment guide
```

## Quick start (local development)

Two terminals:

```powershell
# Terminal 1 -- backend
cd dashboard\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
$env:OUTPUTS_DIR = "..\..\outputs"
$env:DATA_DIR = "..\..\data"
uvicorn app.main:app --reload --port 8000
```

```powershell
# Terminal 2 -- frontend
cd dashboard\frontend
npm install
copy .env.example .env
npm run dev
```

Visit `http://localhost:5173`. See each folder's own README for more detail.

## Deploying to Contabo

See `DEPLOYMENT.md` — one command (`docker compose up -d --build`) after
copying the project + a completed pipeline run onto the server.

## Design notes worth knowing before extending this

- **The "Forecast horizon" control lives in the top toolbar**
  (`components/HorizonSelector.tsx`, rendered directly in `App.tsx`
  next to the widget toggle buttons), NOT as a map overlay. It used to
  be positioned absolutely on top of the map, and that was a real,
  repeated source of bugs: the toolbar sits at a higher z-index with a
  near-opaque background and also anchors to the top-left corner, so
  the overlay kept ending up invisible underneath it. A flex sibling
  inside the toolbar's own `display: flex` row can't suffer that
  failure mode — there's no stacking to get wrong. **If you're adding
  another always-visible control, prefer the toolbar over a map
  overlay for exactly this reason**, unless it genuinely needs to sit
  over map content (the legend, basemap switcher, and tooltip do, and
  stay as overlays; the horizon selector didn't need to).
- **Eight forecast horizons, 3 through 24 months in 3-month steps**
  (`store/useHorizonStore.ts`'s `AVAILABLE_HORIZONS`, matching
  `HORIZONS_MONTHS` in the pipeline's `10_ml_modeling.py` and
  `12_conflict_risk_mapping.py`, and `FORECAST_HORIZONS_MONTHS` in
  `backend/app/config.py` — all four must stay in sync). Originally
  1/3/6 months; widened because that range was too narrow to show any
  real reliability trend on the "Model Reliability Over Time" chart —
  a flat line across just 1-6 months turned out to be a genuine
  finding (performance doesn't degrade much that close in), not a bug,
  and the fix was to look further out, not to force the chart to show
  something.
- **The year-range filter used to crash, not misbehave.** Filtering
  `query_conflicts()` by year called `pd.to_datetime(df["Date_Start"])`
  with no `errors="coerce"`. This dataset has a handful of genuinely
  corrupted `Date_Start` values (documented separately — see the CSV
  review this project produced), and pandas *raises* on those rather
  than skipping them. That exception was propagating out of every
  year-filtered request as a 500 error — which is why "filter to
  2004-2004" appeared to show too many records: the filtered request
  was failing, and the UI was left showing its last successful
  (unfiltered) result rather than the filtered one that never arrived.
  Reproduced directly, fixed with `errors="coerce"` plus the pipeline's
  own known date format (`%m/%d/%Y`), and re-verified end-to-end
  through the live API afterward.
- **Root-cause fix: conflict points/heatmap were rendering nothing.**
  Confirmed via a real reproduction test: `backend/app/data_access.py`
  never coerced `Latitude`/`Longitude` to numeric. If even ONE row in
  the source data has a non-numeric value there (confirmed present in
  this dataset — see `find_bad_coordinates.py` in the pipeline root),
  pandas silently infers the WHOLE COLUMN as string/object dtype, so
  every coordinate — including otherwise-valid ones — serializes as a
  JSON string instead of a number. The frontend's numeric check then
  failed for every record, not just the bad one, so nothing rendered on
  either the points layer or the heatmap (same source, same bug). Fixed
  with explicit `pd.to_numeric(..., errors="coerce")` in
  `get_conflicts_geocoded()`, plus a defensive second layer on the
  frontend (`toFiniteNumber()` in `MapView.tsx`) that accepts a numeric
  string too, in case anything upstream ever sends one again.
- **Zoom-to-selection now covers counties and individual records, not
  just sub-counties.** Priority order in `MapView.tsx`: selected
  sub-county (tightest) → selected county (union extent of its
  sub-counties) → any other active filter narrowing the conflict set →
  full data extent. Clicking a record in the Conflict Records widget (or
  clicking a point directly on the map) flies the view to that exact
  point and highlights it — both directions go through the same
  `store/useSelectionStore.ts`, so list-to-map and map-to-list land on
  the same selection state.
- **The on-map legend is interactive, not decorative.** Every row is a
  checkbox: risk categories (Low/Medium/High/No data) can be hidden
  individually via `hiddenRiskCategories` in `useMapLayersStore`, and
  the same store drives both the legend's own checkboxes and the
  separate "Layers" widget — toggling either one stays in sync with the
  other since they share state.
- **Filters now visibly change map color, not just add an outline.**
  `riskFeatureStyle()` computes an opacity tier based on the active
  county/sub-county filter: a selected sub-county is fully opaque with a
  thick outline, a selected county's sub-counties are emphasized, and
  everything outside the current filter is heavily dimmed (~15% opacity)
  rather than shown at full strength alongside the selection.
- **Fixed-viewport layout, not a scrolling page.** `html`/`body`/`#root`
  have `overflow: hidden` (see `index.css`) — the map fills the entire
  screen as a fixed background, and every other view is a floating
  "widget" (`components/Widget.tsx`) toggled on/off from the top toolbar.
  Only a widget's own body scrolls internally if its content overflows
  (`max-height: 60vh; overflow-y: auto` on `.widget-body`) — the page
  itself never does. Widget open/closed state lives in
  `store/useWidgetStore.ts`, separate from filter state.
- **`backend/app/data_access.py` is the ONLY file that knows data comes
  from CSV/GeoJSON files.** Migrating to Postgres later means changing
  only that file — routers and the entire frontend stay untouched.
- **`frontend/src/store/useFilterStore.ts` is the ONLY source of truth
  for filters.** Every chart, the map, and the search box read from and
  write to it — that's the whole mechanism behind cross-filtering. Add a
  new filterable view by reading from this store, not by inventing a new
  local filter state.
- **Every panel is dynamic on every filter** (region, date range, theme,
  search) except where that's structurally impossible: `ModelMetricsPanel`
  and `ShapPanel` show properties of the trained model itself (not the
  data), so they can't sensibly be filtered by region/date — that's a
  deliberate design decision, not a gap. Trend/seasonal/themes/word-cloud
  widgets aggregate from the FILTERED CONFLICT RECORDS list (not the
  monthly panel), specifically because the panel endpoint doesn't support
  theme/search filtering — see `frontend/src/utils/aggregate.ts` and
  `useConflictsForAggregation`/`useConflictsForTopicBreakdown` in
  `useApiData.ts`. The topics widget deliberately excludes the topic
  filter from its own query, so selecting a theme doesn't collapse its
  own breakdown list down to just itself.
- **Map: OpenLayers, not MapLibre.** MapLibre's demo vector style
  silently failed to resolve (blank blue background, no basemap visible)
  during development — OpenLayers with plain raster XYZ tiles (OSM, Esri
  satellite) sidesteps that failure mode entirely. Three basemaps are
  wired up (switcher control, bottom-right of the map): OpenStreetMap,
  Esri World Imagery (satellite), and Google Roads. **The Google option
  uses an undocumented tile endpoint** (`mt1.google.com/vt/...`) that
  Google's terms don't sanction for production use without a paid Maps
  Platform license — included because it was requested and is common in
  personal projects, but worth revisiting with a real API key before
  relying on it for a long-lived public deployment. See the comment at
  the top of `MapView.tsx` for the fully-licensed alternative path.
- **Zoom-to-selection, not just zoom-to-full-extent-once.** The map view
  refits automatically: to a selected sub-county's polygon (tightest
  zoom), or to the extent of whatever conflict points currently match
  the active filters, or to the full data extent when nothing is
  filtered — see the priority-ordered effect in `MapView.tsx`.
- **Symbology and layer visibility** are controlled from the "Layers"
  widget (`LayersPanel.tsx`) via `store/useMapLayersStore.ts` — conflict
  points can be colored by severity, sentiment, theme, or
  chronic/discrete status; the risk layer can show categorical
  (Low/Medium/High) or continuous probability coloring; a heatmap layer
  (density-weighted by severity) can be toggled independently of the
  point layer. Color scales live in `utils/colors.ts` so choices (e.g.
  which color means which topic) stay consistent everywhere they appear.
- **Word cloud** (`WordCloud.tsx`) is a lightweight CSS flex-wrap tag
  cloud (font size scaled by frequency), not a spiral-packed layout via
  a library like `d3-cloud` — deliberately, since a packing-layout
  library is much harder to verify without being able to visually test
  it, and the simpler approach is still a genuine, useful word cloud. It
  prefers `NLP_Keywords` (Phase 2's extracted terms) and falls back to
  tokenizing `Incident_Summary` — see `utils/wordFrequency.ts`.
- **Fullscreen**: OpenLayers' built-in `ol/control/FullScreen`, restyled
  to match the dashboard (top-right of the map).
- **This was built and verified without a browser available** (sandboxed
  dev environment). TypeScript compiles cleanly and the production build
  succeeds — several real compile errors were caught and fixed this way —
  but actually clicking through the map, widgets, and filters has not
  been visually confirmed. Do that locally before trusting it fully; if
  something looks or behaves wrong, it's very plausibly a real bug rather
  than something already checked. The floating-widget layout in
  particular (drawer positioning, overlap with the toolbar, widget stack
  overflow) is the area most likely to need visual polish once you can
  actually see it.
