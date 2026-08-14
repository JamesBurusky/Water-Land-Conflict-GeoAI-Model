# Deploying to Contabo

## Prerequisites on the Contabo VPS

1. SSH into your VPS.
2. Install Docker and Docker Compose (Ubuntu example):
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo apt-get install -y docker-compose-plugin
   ```
3. Open port 80 in the firewall, if one is active:
   ```bash
   sudo ufw allow 80/tcp
   ```
   (Contabo's control panel may also have its own firewall — check there too if `ufw` isn't the blocker.)

## Get the project onto the server

Two options:

**Option A — copy the whole project via `scp`/`rsync`** (simplest, since you're
running the pipeline locally on Windows right now):
```bash
# From your Windows machine, in a terminal with scp available (e.g. Git Bash):
scp -r geoai_conflict user@YOUR_CONTABO_IP:/opt/
```

**Option B — put it in a git repo and `git clone` it on the server.** Cleaner
for ongoing updates, but needs a repo set up first. Either works — the
pipeline's `outputs/` and `data/` folders need to actually be present on the
server either way, since the dashboard reads real files from them.

## Run it

```bash
ssh user@YOUR_CONTABO_IP
cd /opt/geoai_conflict/dashboard
docker compose up -d --build
```

First build takes a few minutes (installing GDAL/GeoPandas system
dependencies + npm packages). Subsequent rebuilds are much faster thanks to
Docker's layer caching.

**Visit `http://YOUR_CONTABO_IP/`** — that's the whole dashboard, on port 80,
no further configuration needed.

## Updating after re-running the pipeline

If you re-run any pipeline script (`01_phase1_data_audit.py` through
`12_conflict_risk_mapping.py`) on the server and it produces new/updated
files in `outputs/`, restart just the backend to pick them up immediately
(it caches file reads for 5 minutes by default — see `CACHE_TTL_SECONDS` in
`backend/app/config.py` — but a restart guarantees it right away):
```bash
cd /opt/geoai_conflict/dashboard
docker compose restart backend
```

## Updating the dashboard code itself

```bash
cd /opt/geoai_conflict/dashboard
# pull/copy your updated code here first, then:
docker compose up -d --build
```

## Adding HTTPS later (optional, once you have a domain)

Right now this serves plain HTTP on the raw IP, which is fine for an
internal/thesis-demo tool. If you later point a domain at this server and
want HTTPS, the simplest path is adding [Caddy](https://caddyserver.com/) or
[nginx + certbot](https://certbot.eff.org/) in front of this stack — happy
to help set that up when you're at that point; it's a small addition, not a
redesign.

## Migrating to Postgres later

When you're ready: only `backend/app/data_access.py` needs to change. Each
function's signature and return type (a pandas DataFrame) stays the same —
`pd.read_csv(path)` becomes `pd.read_sql(query, conn)`, and filter arguments
become a `WHERE` clause instead of pandas boolean indexing. Nothing in
`routers/`, the React frontend, or this deployment setup needs to change.
You'd add a `postgres` service to `docker-compose.yml` and point the
backend's connection string at it via a new environment variable, similar
to how `OUTPUTS_DIR`/`DATA_DIR` work now.
