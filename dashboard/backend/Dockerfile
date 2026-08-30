FROM python:3.12-slim

# geopandas needs GDAL/GEOS/PROJ system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# OUTPUTS_DIR / DATA_DIR are set by docker-compose.yml to point at the
# mounted pipeline output volume -- see ../docker-compose.yml
ENV OUTPUTS_DIR=/pipeline/outputs
ENV DATA_DIR=/pipeline/data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
