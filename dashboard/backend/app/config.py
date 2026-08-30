"""
config.py

All configurable paths/settings in one place, overridable via
environment variables -- this is what makes the same code work
unchanged in local dev, in a Docker container on Contabo, and later
when OUTPUTS_DIR points at a checked-out pipeline run on a mounted
volume instead of a local folder.
"""
import os
from pathlib import Path

# Points at the geoai_conflict pipeline's outputs/ and data/ folders.
# In Docker, this gets set to wherever the pipeline output volume is
# mounted -- see docker-compose.yml.
#
# IMPORTANT: the default is resolved relative to THIS FILE's location
# (dashboard/backend/app/config.py -> geoai_conflict/), NOT the
# process's current working directory. A relative default like "../.."
# would silently break every single endpoint if uvicorn is ever
# launched from a directory other than dashboard/backend/ exactly --
# this bit a real user during development (a 404 on every endpoint
# that looked like a missing-pipeline-output problem was actually this).
# Explicit PIPELINE_ROOT/OUTPUTS_DIR/DATA_DIR env vars (e.g. in Docker)
# still take priority over this file-relative default.
_THIS_FILE = Path(__file__).resolve()


def _resolve_pipeline_root() -> Path:
    env_val = os.environ.get("PIPELINE_ROOT")
    if env_val:
        return Path(env_val)
    # Only reached if PIPELINE_ROOT itself isn't set via environment --
    # in this project's actual Docker deployment, OUTPUTS_DIR and
    # DATA_DIR are BOTH set directly (see docker-compose.yml), so this
    # fallback value is computed but never actually used for anything.
    # It must still not crash on computation, though: confirmed as a
    # real production failure. A Docker image built with
    # `context: ./backend` copies this file to roughly /app/app/config.py
    # inside the container, only 2 real parent directories deep, not the
    # 4 a local checkout (dashboard/backend/app/config.py ->
    # geoai_conflict/) has -- .parents[3] doesn't exist there, and
    # raises IndexError, which crashed the app on import before it ever
    # reached the OUTPUTS_DIR/DATA_DIR environment variables below that
    # would have made this fallback irrelevant in the first place.
    if len(_THIS_FILE.parents) > 3:
        return _THIS_FILE.parents[3]
    return Path.cwd()


PIPELINE_ROOT = _resolve_pipeline_root()
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(PIPELINE_ROOT / "outputs")))
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PIPELINE_ROOT / "data")))

# The pipeline now writes each step into its own subfolder under
# outputs/ (e.g. outputs/06_spatial_feature_engineering/) -- see
# src/output_paths.py in the pipeline itself. This mirrors that
# convention so data_access.py can find each step's files without
# hardcoding outputs/ as a flat namespace.
def step_dir(step_name: str) -> Path:
    return OUTPUTS_DIR / step_name


def horizon_dir(step_name: str, horizon_months: int) -> Path:
    return step_dir(step_name) / f"horizon_{horizon_months}month"


# Forecast horizons the multi-horizon modelling/risk-mapping scripts
# produce -- MUST match HORIZONS_MONTHS in 10_ml_modeling.py and
# 12_conflict_risk_mapping.py.
FORECAST_HORIZONS_MONTHS = [3, 6, 9, 12, 15, 18, 21, 24]

# CORS origins allowed to call this API -- the React dev server locally,
# and whatever your Contabo domain/IP ends up being in production.
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

# Simple in-process cache TTL (seconds) for file reads -- files change
# only when the pipeline re-runs (infrequent), so re-reading a CSV on
# every single API request is wasted work. See data_access.py.
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "300"))