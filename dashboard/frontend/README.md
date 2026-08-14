# GeoAI Conflict Dashboard — Frontend

React + TypeScript + Vite dashboard. Talks to the FastAPI backend
(`../backend`) — start that first.

## Architecture

- **`src/store/useFilterStore.ts`** — the single shared filter state
  (county/subcounty/year range/topic/search). Every component reads from
  and writes to this store — that's the entire mechanism behind
  "clicking the map filters every chart" and "the search box filters the
  map's points too". No component manages its own separate copy of a filter.
- **`src/hooks/useApiData.ts`** — React Query hooks that read the filter
  store and call the API client. Changing a filter anywhere automatically
  triggers refetches everywhere that filter matters.
- **`src/api/client.ts`** — typed fetch wrappers, one per backend endpoint.
- **`src/components/MapView.tsx`** — MapLibre GL choropleth. Lazy-loaded
  from `App.tsx` since it's by far the largest dependency (~1MB) — the
  rest of the dashboard becomes interactive without waiting on it.

## Local development

**Windows (PowerShell)**:
```powershell
cd dashboard\frontend
npm install
copy .env.example .env
npm run dev
```

**macOS / Linux (bash)**:
```bash
cd dashboard/frontend
npm install
cp .env.example .env
npm run dev
```

Visit `http://localhost:5173`. Make sure the backend is running on
`http://localhost:8000` first (see `../backend/README.md`) — `.env`'s
`VITE_API_BASE_URL` points there by default.

## Type-checking and building

```bash
npx tsc --noEmit   # type-check only, fast
npm run build      # full production build -- stricter than the above,
                    # catches things plain tsc --noEmit misses (this
                    # project hit 6 real compile errors this way during
                    # development that --noEmit alone didn't catch)
```

**Important — this project could not be visually tested in a browser during
development** (built in a sandboxed environment without one). Everything
here compiles cleanly and the logic was reasoned through carefully, but
**click through it yourself locally before trusting it fully** — map
interactions, chart rendering, and the overall layout genuinely need human
eyes on them.

## Production build

Don't run `npm run build` directly for deployment — Docker handles this
(see `Dockerfile` + `../docker-compose.yml`), since `VITE_API_BASE_URL`
needs to be set correctly at build time for production (empty string, for
same-origin requests through nginx — see the comment in `src/api/client.ts`
for why). See `../DEPLOYMENT.md` for the full Contabo deployment steps.
