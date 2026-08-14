# GeoAI Conflict Dashboard — Backend

FastAPI service that serves the pipeline's outputs (`outputs/*.csv`,
`*.geojson`, `*.joblib`) as a clean REST API for the React frontend.

## Architecture — read this before touching anything

**`app/data_access.py` is the only file that knows where data comes from.**
Every route handler calls a function in there — never reads a file directly.
Right now every function reads from the pipeline's `outputs/` folder. When
you migrate to Postgres later, only `data_access.py` changes (e.g.
`get_panel()` becomes `pd.read_sql(...)` instead of `pd.read_csv(...)`) —
the routers, the API contract, and the entire React frontend stay untouched.

## Local development

**Windows (PowerShell)** — what you've been using for the rest of this project:
```powershell
cd dashboard\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Point at your pipeline's outputs/data folders (defaults assume this
# backend folder sits at geoai_conflict\dashboard\backend\, two levels
# below geoai_conflict\ itself):
$env:OUTPUTS_DIR = "..\..\outputs"
$env:DATA_DIR = "..\..\data"

uvicorn app.main:app --reload --port 8000
```

**macOS / Linux (bash)**:
```bash
cd dashboard/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export OUTPUTS_DIR=../../outputs
export DATA_DIR=../../data

uvicorn app.main:app --reload --port 8000
```

Note: `$env:VAR = "value"` is PowerShell's equivalent of bash's `export VAR=value`
— they don't mix, so use the block matching your actual shell.

Visit `http://localhost:8000/docs` for interactive API docs (auto-generated
from the route definitions — the fastest way to see exactly what's
available while building the frontend).

Visit `http://localhost:8000/api/health` first — it reports which pipeline
outputs are actually present, so you know which frontend panels have real
data to show.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness + which pipeline outputs are available |
| `GET /api/spatial/risk-layer` | GeoJSON: sub-county risk map (Phase 8) |
| `GET /api/spatial/hotspots` | Getis-Ord Gi* hotspot classification (Phase 4) |
| `GET /api/panel` | Filtered ML panel rows (county/subcounty/year range) |
| `GET /api/panel/filters` | Distinct counties/sub-counties/years/topics for dropdowns |
| `GET /api/panel/summary` | Aggregated KPIs for the current filter selection |
| `GET /api/conflicts` | Individual conflict records, filter + free-text search |
| `GET /api/analytics/model-metrics` | Phase 6 model comparison table |
| `GET /api/analytics/shap-importance` | Phase 7 SHAP feature importance |
| `GET /api/analytics/topics` | Phase 2 topic modelling results |
| `GET /api/analytics/yearly-trend` | Phase 4 yearly trend |
| `GET /api/analytics/seasonal-pattern` | Phase 4 seasonal pattern |

All list-returning endpoints are NaN/Timestamp-safe (see
`data_access.df_to_json_records` / `geodf_to_json_safe_geojson`) — a real
bug class we hit and fixed during development: pandas NaN in a numeric
column can't be replaced with Python `None` in place (numpy has no such
value for a float array), so naive `.where(df.notna(), None)` silently
fails and the bad value survives to break strict JSON encoding. Fixed by
converting to plain Python dicts first, then cleaning values.

## Deploying on Contabo

See `../docker-compose.yml` at the project root — builds this backend,
the React frontend, and an nginx reverse proxy as three containers. Full
steps in `../DEPLOYMENT.md`.
