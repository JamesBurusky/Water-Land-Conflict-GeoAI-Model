// The dashboard's map (OpenLayers). See dashboard/README.md for the
// full design rationale (MapLibre -> OpenLayers switch, symbology,
// heatmap, zoom-to-selection). This version fixes several reported
// issues: conflict points/heatmap not rendering (root cause was in the
// BACKEND -- see data_access.py's numeric coercion fix; this file adds
// a defensive frontend-side coercion too), zoom-to-selection only
// working for sub-counties (now also works for counties and for a
// specific conflict clicked in the records list), no way to toggle
// features on/off from a legend (the on-map legend is now interactive,
// not decorative), and filters not visibly affecting map colors (now
// non-matching risk-layer features are dimmed, not just outlined).

import { useEffect, useRef, useState } from "react";
import "ol/ol.css";
import Map from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import HeatmapLayer from "ol/layer/Heatmap";
import OSM from "ol/source/OSM";
import XYZ from "ol/source/XYZ";
import VectorSource from "ol/source/Vector";
import GeoJSON from "ol/format/GeoJSON";
import { Style, Fill, Stroke, Circle as CircleStyle } from "ol/style";
import { defaults as defaultControls } from "ol/control";
import { fromLonLat, toLonLat } from "ol/proj";
import { createEmpty, extend as extendExtent } from "ol/extent";
import type Point from "ol/geom/Point";
import type { FeatureLike } from "ol/Feature";
import type { Coordinate } from "ol/coordinate";
import type { Extent } from "ol/extent";
import { useRiskLayer, useConflictsForAggregation } from "../hooks/useApiData";
import { useFilterStore } from "../store/useFilterStore";
import { useMapLayersStore, type RiskCategory } from "../store/useMapLayersStore";
import { useSelectionStore } from "../store/useSelectionStore";
import { useHorizonStore } from "../store/useHorizonStore";
import {
  RISK_COLORS, NO_DATA_COLOR, SENTIMENT_COLORS,
  colorForTopic, colorForRatio, colorForComposite,
} from "../utils/colors";

type BasemapId = "osm" | "satellite" | "google";

const BASEMAP_LABELS: Record<BasemapId, string> = {
  osm: "OpenStreetMap",
  satellite: "Satellite",
  google: "Google Roads",
};

function makeBasemapLayer(id: BasemapId, onTileError?: () => void): TileLayer {
  let source: OSM | XYZ;
  switch (id) {
    case "osm":
      source = new OSM();
      break;
    case "satellite":
      source = new XYZ({
        url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attributions: "Tiles © Esri — Esri, Maxar, Earthstar Geographics",
        maxZoom: 19,
      });
      break;
    case "google":
      source = new XYZ({
        // Unofficial endpoint -- see the design note in dashboard/README.md.
        url: "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attributions: "Map data © Google (unofficial endpoint — see README)",
        maxZoom: 20,
      });
      break;
  }
  // Surfaces a load failure explicitly instead of leaving a silently
  // blank/grey map -- added specifically because "the basemap button
  // doesn't do anything" and "the tiles failed to load after
  // switching" look IDENTICAL to a user with no error message to tell
  // them apart, and they need genuinely different fixes (a UI bug vs.
  // a network/firewall block on that specific tile domain).
  if (onTileError) {
    source.on("tileloaderror", onTileError);
  }
  return new TileLayer({ source });
}

function riskCategoryOf(feature: FeatureLike): RiskCategory {
  const category = feature.get("risk_category") as string | null;
  if (!category) return "No data";
  if (category === "Low" || category === "Lower") return "Low";
  if (category === "Medium") return "Medium";
  if (category === "High" || category === "Higher") return "High";
  return "No data";
}

function riskFeatureStyle(
  feature: FeatureLike,
  filterCounty: string | undefined,
  filterSubcounty: string | undefined,
  symbology: "category" | "probability",
  hiddenCategories: Set<RiskCategory>
): Style | undefined {
  const category = riskCategoryOf(feature);
  if (hiddenCategories.has(category)) return undefined; // legend toggle: hide this category entirely

  const probability = feature.get("risk_probability") as number | null;
  const featureCounty = feature.get("County") as string | undefined;
  const featureSubcounty = feature.get("SubCounty") as string | undefined;

  let baseColor: string;
  if (symbology === "probability" && typeof probability === "number") {
    baseColor = colorForRatio(probability);
  } else {
    baseColor = category === "No data" ? NO_DATA_COLOR : RISK_COLORS[category];
  }

  // Filters visibly affect map color, not just an outline highlight:
  // a selected sub-county is fully opaque and outlined; features
  // outside the active county/subcounty filter are heavily dimmed
  // rather than shown at full strength alongside the selection.
  let alphaHex = "B3"; // ~70% -- default, no filter active
  let strokeWidth = 0.75;
  if (filterSubcounty) {
    if (featureSubcounty === filterSubcounty) {
      alphaHex = "F0"; strokeWidth = 2.5;
    } else {
      alphaHex = "26"; // ~15%
    }
  } else if (filterCounty) {
    if (featureCounty === filterCounty) {
      alphaHex = "D9"; strokeWidth = 1.25; // ~85%, clearly emphasized but not as tight as a single sub-county pick
    } else {
      alphaHex = "26";
    }
  }

  return new Style({
    fill: new Fill({ color: baseColor + alphaHex }),
    stroke: new Stroke({ color: "#333333", width: strokeWidth }),
  });
}

function conflictPointStyle(feature: FeatureLike, symbology: string, isSelected: boolean): Style {
  let color = "#264653";
  switch (symbology) {
    case "severity": {
      const s = feature.get("severity_score") as number | null;
      color = typeof s === "number" ? colorForRatio(Math.min(s, 10) / 10) : NO_DATA_COLOR;
      break;
    }
    case "sentiment": {
      const label = feature.get("sentiment_label") as string | null;
      color = label ? SENTIMENT_COLORS[label] ?? NO_DATA_COLOR : NO_DATA_COLOR;
      break;
    }
    case "topic":
      color = colorForTopic(feature.get("topic_id") as number | null);
      break;
    case "composite":
      color = colorForComposite(feature.get("Is_Composite") as boolean | null);
      break;
    default:
      color = "#264653";
  }
  return new Style({
    image: new CircleStyle({
      radius: isSelected ? 9 : 5,
      fill: new Fill({ color }),
      stroke: new Stroke({ color: isSelected ? "#e76f51" : "#ffffff", width: isSelected ? 2.5 : 1 }),
    }),
    zIndex: isSelected ? 10 : 1,
  });
}

function isValidExtent(extent: Extent | null): extent is Extent {
  return extent !== null && extent.every((v) => Number.isFinite(v));
}

/** Defensive numeric parse -- the backend now coerces Latitude/
 * Longitude explicitly (see data_access.py), but this stays as a
 * second line of defense: Number.isFinite is used instead of a bare
 * `typeof === "number"` check so a numeric string that slips through
 * for any reason still gets parsed correctly rather than silently
 * dropping the point. */
function toFiniteNumber(v: unknown): number | null {
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
}

export function MapView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapPanelRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const riskSourceRef = useRef<VectorSource | null>(null);
  const conflictSourceRef = useRef<VectorSource | null>(null);
  const riskLayerObjRef = useRef<VectorLayer | null>(null);
  const conflictLayerObjRef = useRef<VectorLayer | null>(null);
  const heatmapLayerObjRef = useRef<HeatmapLayer | null>(null);
  const hasFitFullExtentRef = useRef(false);

  const [basemap, setBasemap] = useState<BasemapId>("osm");
  const [basemapTileError, setBasemapTileError] = useState(false);
  const tileErrorCountRef = useRef(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Custom fullscreen implementation (not OL's built-in FullScreen
  // control) specifically because the toggle button needs to keep
  // working WHILE fullscreened, and OL's control only fullscreens its
  // own map viewport -- our own overlays (legend, basemap switcher,
  // this very button) are siblings of that viewport, not descendants
  // of it, so they'd vanish the moment OL's version went fullscreen.
  // Targeting our own outer .map-panel div keeps everything, including
  // this button, visible and usable throughout.
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      mapPanelRef.current?.requestFullscreen().catch(() => {
        // Some browsers/embeds (e.g. within a cross-origin iframe
        // without the allowfullscreen attribute) reject this silently
        // -- nothing more useful to do here than let the click be a
        // no-op rather than throw an unhandled rejection.
      });
    } else {
      document.exitFullscreen();
    }
  };

  useEffect(() => {
    const handleChange = () => setIsFullscreen(document.fullscreenElement === mapPanelRef.current);
    document.addEventListener("fullscreenchange", handleChange);
    return () => document.removeEventListener("fullscreenchange", handleChange);
  }, []);

  const [hovered, setHovered] = useState<{ props: Record<string, unknown>; pixel: Coordinate } | null>(null);

  const { data: riskLayer, isLoading: riskLoading, error: riskError } = useRiskLayer();
  const { data: conflicts } = useConflictsForAggregation();
  const { county, subcounty, yearMin, yearMax, topicId, search, setCounty, setSubcounty } = useFilterStore();
  const {
    showRiskLayer, showConflictPoints, showHeatmap, pointSymbology, riskSymbology, hiddenRiskCategories,
    toggleRiskCategory,
  } = useMapLayersStore();
  const { flyToCoordinate, flyToRecordId, sequence } = useSelectionStore();
  const horizonMonths = useHorizonStore((s) => s.horizonMonths);

  // --- Map initialization (once) ---
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const riskSource = new VectorSource();
    const conflictSource = new VectorSource();
    riskSourceRef.current = riskSource;
    conflictSourceRef.current = conflictSource;

    const riskLayerObj = new VectorLayer({
      source: riskSource,
      style: (f) => {
        const fs = useFilterStore.getState();
        const ms = useMapLayersStore.getState();
        return riskFeatureStyle(f, fs.county, fs.subcounty, ms.riskSymbology, ms.hiddenRiskCategories);
      },
    });
    const conflictLayerObj = new VectorLayer({
      source: conflictSource,
      style: (f) => {
        const ms = useMapLayersStore.getState();
        const sel = useSelectionStore.getState();
        return conflictPointStyle(f, ms.pointSymbology, f.get("Record_ID") === sel.flyToRecordId);
      },
    });
    const heatmapLayerObj = new HeatmapLayer({
      source: conflictSource,
      blur: 18,
      radius: 10,
      weight: (feature) => {
        const s = feature.get("severity_score") as number | null;
        return typeof s === "number" ? Math.min(s, 10) / 10 : 0.4;
      },
    });
    heatmapLayerObj.setVisible(false);

    riskLayerObjRef.current = riskLayerObj;
    conflictLayerObjRef.current = conflictLayerObj;
    heatmapLayerObjRef.current = heatmapLayerObj;

    const map = new Map({
      target: containerRef.current,
      layers: [
        makeBasemapLayer("osm", () => {
          tileErrorCountRef.current += 1;
          if (tileErrorCountRef.current >= 3) setBasemapTileError(true);
        }),
        riskLayerObj, heatmapLayerObj, conflictLayerObj,
      ],
      view: new View({ center: [0, 0], zoom: 2 }),
      controls: defaultControls(),
    });

    map.on("click", (e) => {
      // Conflict points sit on top -- check those first. Clicking a
      // point selects/highlights that specific record (matches the
      // "click a conflict in the records list" behavior symmetrically:
      // either direction -- list-to-map or map-to-list -- lands on the
      // same selection state) rather than also triggering a county/
      // sub-county filter change underneath it.
      const conflictFeature = map.forEachFeatureAtPixel(e.pixel, (f) => f, {
        layerFilter: (l) => l === conflictLayerObj,
      });
      if (conflictFeature) {
        const recordId = conflictFeature.get("Record_ID") as string | undefined;
        const geometry = conflictFeature.getGeometry() as Point | undefined;
        if (recordId && geometry) {
          const [lon, lat] = toLonLat(geometry.getCoordinates());
          useSelectionStore.getState().pinFromMap(recordId, lon, lat);
        }
        return;
      }

      const riskFeature = map.forEachFeatureAtPixel(e.pixel, (f) => f, { layerFilter: (l) => l === riskLayerObj });
      if (riskFeature) {
        const c = riskFeature.get("County") as string | undefined;
        const sc = riskFeature.get("SubCounty") as string | undefined;
        if (c) setCounty(c);
        if (sc) setSubcounty(sc);
        useSelectionStore.getState().clearPin();
      }
    });

    map.on("pointermove", (e) => {
      const feature = map.forEachFeatureAtPixel(e.pixel, (f) => f, {
        layerFilter: (l) => l === riskLayerObj || l === conflictLayerObj,
      });
      map.getViewport().style.cursor = feature ? "pointer" : "";
      setHovered(feature ? { props: feature.getProperties(), pixel: e.pixel } : null);
    });

    mapRef.current = map;
    return () => {
      map.setTarget(undefined);
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Basemap switching ---
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    setBasemapTileError(false);
    tileErrorCountRef.current = 0;
    map.getLayers().setAt(0, makeBasemapLayer(basemap, () => {
      // A single failed tile is normal even on a healthy connection
      // (a transient blip, one tile at the edge of coverage) -- only
      // surface the error banner once several fail in a row, which
      // indicates a systemic problem (the whole domain being blocked)
      // rather than one flaky request.
      tileErrorCountRef.current += 1;
      if (tileErrorCountRef.current >= 3) setBasemapTileError(true);
    }));
  }, [basemap]);

  // --- Layer visibility toggles ---
  useEffect(() => { riskLayerObjRef.current?.setVisible(showRiskLayer); }, [showRiskLayer]);
  useEffect(() => { conflictLayerObjRef.current?.setVisible(showConflictPoints); }, [showConflictPoints]);
  useEffect(() => { heatmapLayerObjRef.current?.setVisible(showHeatmap); }, [showHeatmap]);

  // --- Re-style on symbology/filter/legend-toggle change ---
  useEffect(() => {
    riskLayerObjRef.current?.setStyle((f) =>
      riskFeatureStyle(f, county, subcounty, riskSymbology, hiddenRiskCategories)
    );
  }, [county, subcounty, riskSymbology, hiddenRiskCategories]);
  useEffect(() => {
    conflictLayerObjRef.current?.setStyle((f) =>
      conflictPointStyle(f, pointSymbology, f.get("Record_ID") === flyToRecordId)
    );
  }, [pointSymbology, flyToRecordId]);

  // --- Risk layer data: parse GeoJSON, add to source ---
  useEffect(() => {
    const source = riskSourceRef.current;
    if (!source || !riskLayer) return;
    source.clear();
    const features = new GeoJSON().readFeatures(riskLayer, {
      dataProjection: "EPSG:4326",
      featureProjection: "EPSG:3857",
    });
    source.addFeatures(features);
  }, [riskLayer]);

  // --- Conflict points: rebuilt whenever the filtered list changes ---
  useEffect(() => {
    const source = conflictSourceRef.current;
    if (!source) return;
    source.clear();
    const withCoords = (conflicts ?? [])
      .map((c) => ({ ...c, _lat: toFiniteNumber(c.Latitude), _lon: toFiniteNumber(c.Longitude) }))
      .filter((c) => c._lat !== null && c._lon !== null);
    if (withCoords.length === 0) return;
    const geojson = {
      type: "FeatureCollection" as const,
      features: withCoords.map((c) => ({
        type: "Feature" as const,
        properties: {
          Record_ID: c.Record_ID,
          summary: c.Incident_Summary ?? "",
          severity_score: c.severity_score ?? null,
          sentiment_label: c.sentiment_label ?? null,
          topic_id: c.topic_id ?? null,
          topic_label: c.topic_label ?? null,
          Is_Composite: c.Is_Composite ?? null,
        },
        geometry: { type: "Point" as const, coordinates: [c._lon as number, c._lat as number] },
      })),
    };
    const features = new GeoJSON().readFeatures(geojson, {
      dataProjection: "EPSG:4326",
      featureProjection: "EPSG:3857",
    });
    source.addFeatures(features);
  }, [conflicts]);

  // --- Fly to a specific conflict record (selected from the records
  // list, OR clicked directly on the map -- both go through the same
  // store, so either direction lands in the same place) ---
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !flyToCoordinate) return;
    const center = fromLonLat(flyToCoordinate);
    map.getView().animate({ center, zoom: Math.max(map.getView().getZoom() ?? 8, 12), duration: 500 });
    // sequence, not just flyToCoordinate, is the dependency -- clicking
    // the SAME record twice produces the same coordinate value but a
    // new sequence number, so the map still re-centers/re-animates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sequence]);

  // --- Zoom-to-selection (region-level; the effect above handles a
  // single clicked/selected conflict record separately) ---
  // Priority: (1) a selected sub-county -> zoom tight to its polygon;
  // (2) a selected county (no sub-county) -> zoom to the union extent
  // of all its sub-county polygons; (3) any other active filter
  // narrowing the conflict set -> zoom to the filtered points' extent;
  // (4) no filters -> fit the full risk-layer extent (once).
  useEffect(() => {
    const map = mapRef.current;
    const riskSource = riskSourceRef.current;
    const conflictSource = conflictSourceRef.current;
    if (!map || !riskSource) return;

    if (subcounty) {
      const match = riskSource.getFeatures().find((f) => f.get("SubCounty") === subcounty);
      const geomExtent = match?.getGeometry()?.getExtent() ?? null;
      if (isValidExtent(geomExtent)) {
        map.getView().fit(geomExtent, { padding: [60, 60, 60, 60], maxZoom: 13, duration: 400 });
        return;
      }
    }

    if (county) {
      const matches = riskSource.getFeatures().filter((f) => f.get("County") === county);
      if (matches.length > 0) {
        const combined = createEmpty();
        for (const f of matches) {
          const ext = f.getGeometry()?.getExtent();
          if (ext) extendExtent(combined, ext);
        }
        if (isValidExtent(combined)) {
          map.getView().fit(combined, { padding: [50, 50, 50, 50], maxZoom: 11, duration: 400 });
          return;
        }
      }
    }

    const hasActiveFilter = Boolean(yearMin || yearMax || topicId !== undefined || search);
    if (hasActiveFilter && conflictSource) {
      const extent = conflictSource.getExtent();
      if (isValidExtent(extent) && conflictSource.getFeatures().length > 0) {
        map.getView().fit(extent, { padding: [60, 60, 60, 60], maxZoom: 12, duration: 400 });
        return;
      }
    }

    if (!hasFitFullExtentRef.current || !(county || hasActiveFilter)) {
      const fullExtent = riskSource.getExtent();
      if (isValidExtent(fullExtent) && riskSource.getFeatures().length > 0) {
        map.getView().fit(fullExtent, { padding: [30, 30, 30, 30], maxZoom: 10, duration: 400 });
        hasFitFullExtentRef.current = true;
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subcounty, county, yearMin, yearMax, topicId, search, conflicts, riskLayer]);

  const legendCategories: RiskCategory[] = ["Low", "Medium", "High", "No data"];
  const legendColors: Record<RiskCategory, string> = {
    Low: RISK_COLORS.Low, Medium: RISK_COLORS.Medium, High: RISK_COLORS.High, "No data": NO_DATA_COLOR,
  };

  return (
    <div ref={mapPanelRef} className="map-panel map-panel-fullbleed">
      <div ref={containerRef} className="map-container" />

      <button
        className="map-fullscreen-btn"
        onClick={toggleFullscreen}
        title={isFullscreen ? "Exit full screen" : "View map full screen"}
      >
        {isFullscreen ? "⤡ Exit full screen" : "⤢ Full screen"}
      </button>

      {riskLoading && <div className="map-overlay-message">Loading risk map…</div>}
      {riskError && (
        <div className="map-overlay-message map-overlay-error">
          Risk layer unavailable — run 12_conflict_risk_mapping.py in the pipeline first.
        </div>
      )}
      {!riskLoading && !riskError && !riskLayer && (
        <div className="map-overlay-message map-overlay-error">
          No prediction for the {horizonMonths}-month horizon yet — run 10_ml_modeling.py and
          12_conflict_risk_mapping.py, or pick a different horizon above.
        </div>
      )}

      <div className="map-basemap-switcher">
        {(Object.keys(BASEMAP_LABELS) as BasemapId[]).map((id) => (
          <button
            key={id}
            className={basemap === id ? "basemap-btn basemap-btn-active" : "basemap-btn"}
            onClick={() => setBasemap(id)}
          >
            {BASEMAP_LABELS[id]}
          </button>
        ))}
      </div>

      {basemapTileError && (
        <div className="map-basemap-error">
          "{BASEMAP_LABELS[basemap]}" tiles failed to load — this is a network issue reaching{" "}
          {basemap === "satellite" ? "Esri's" : basemap === "google" ? "Google's" : "the tile"} servers
          (a firewall or ad-blocker on this network is the most common cause), not a broken button.
          Try OpenStreetMap, or check your network's outbound access to this basemap's domain.
        </div>
      )}

      {hovered && (
        <div className="map-tooltip" style={{ left: hovered.pixel[0] + 12, top: hovered.pixel[1] + 12 }}>
          {hovered.props.SubCounty ? (
            <>
              <strong>{String(hovered.props.SubCounty)}</strong>, {String(hovered.props.County)}
              <br />
              Risk: {String(hovered.props.risk_category ?? "No data")}
              {typeof hovered.props.risk_probability === "number" && (
                <> ({(hovered.props.risk_probability * 100).toFixed(1)}%)</>
              )}
            </>
          ) : (
            <>
              <strong>Record: {String(hovered.props.Record_ID ?? "")}</strong>
              {hovered.props.summary ? (
                <><br />{String(hovered.props.summary).slice(0, 140)}
                  {String(hovered.props.summary).length > 140 ? "…" : ""}</>
              ) : null}
              {hovered.props.topic_label ? <><br />Theme: {String(hovered.props.topic_label)}</> : null}
              {typeof hovered.props.severity_score === "number" && (
                <><br />Severity score: {(hovered.props.severity_score as number).toFixed(1)}</>
              )}
              {hovered.props.sentiment_label ? (
                <><br />Report tone: {String(hovered.props.sentiment_label)} <span className="tooltip-hint">(tone of the write-up, not a severity rating)</span></>
              ) : null}
              <br /><span className="tooltip-hint">Click to filter the records list to this incident</span>
            </>
          )}
        </div>
      )}

      {/* Interactive legend -- every row is a toggle, not decorative.
          Risk categories hide/show via hiddenRiskCategories; points and
          heatmap use the same booleans the Layers widget controls, so
          both stay in sync regardless of which control the user uses. */}
      <div className="map-legend">
        <div className="map-legend-title">Risk level</div>
        {legendCategories.map((cat) => (
          <label key={cat} className="map-legend-row map-legend-row-toggle">
            <input
              type="checkbox"
              checked={!hiddenRiskCategories.has(cat)}
              onChange={() => toggleRiskCategory(cat)}
            />
            <span className="map-legend-swatch" style={{ background: legendColors[cat] }} />
            {cat}
          </label>
        ))}
        <div className="map-legend-divider" />
        <label className="map-legend-row map-legend-row-toggle">
          <input
            type="checkbox"
            checked={showConflictPoints}
            onChange={(e) => useMapLayersStore.getState().setShowConflictPoints(e.target.checked)}
          />
          <span className="map-legend-swatch map-legend-swatch-dot" style={{ background: "#264653" }} />
          Conflict records
        </label>
        <label className="map-legend-row map-legend-row-toggle">
          <input
            type="checkbox"
            checked={showHeatmap}
            onChange={(e) => useMapLayersStore.getState().setShowHeatmap(e.target.checked)}
          />
          <span className="map-legend-swatch" style={{ background: "linear-gradient(90deg, #2a9d8f, #e76f51)" }} />
          Density heatmap
        </label>
      </div>

      {(county || subcounty) && (
        <div className="map-active-filter">
          Filtered to: {subcounty ? `${subcounty}, ` : ""}{county}
        </div>
      )}
    </div>
  );
}
