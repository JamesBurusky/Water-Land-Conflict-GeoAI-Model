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
PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "../.."))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", PIPELINE_ROOT / "outputs"))
DATA_DIR = Path(os.environ.get("DATA_DIR", PIPELINE_ROOT / "data"))

# CORS origins allowed to call this API -- the React dev server locally,
# and whatever your Contabo domain/IP ends up being in production.
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

# Simple in-process cache TTL (seconds) for file reads -- files change
# only when the pipeline re-runs (infrequent), so re-reading a CSV on
# every single API request is wasted work. See data_access.py.
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
