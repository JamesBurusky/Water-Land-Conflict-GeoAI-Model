import { useMemo } from "react";
import { useConflicts } from "../hooks/useApiData";
import { useSelectionStore } from "../store/useSelectionStore";

export function ConflictList() {
  const { data, isLoading } = useConflicts();
  const selectConflict = useSelectionStore((s) => s.selectConflict);
  const flyToRecordId = useSelectionStore((s) => s.flyToRecordId);
  const pinnedRecordId = useSelectionStore((s) => s.pinnedRecordId);
  const clearPin = useSelectionStore((s) => s.clearPin);

  // A map point click PINS this list to just that one record,
  // overriding every other active filter -- this is deliberate: the
  // point was clicked specifically to inspect that one incident, not
  // to additionally intersect it with whatever filters happen to be
  // set elsewhere.
  const displayed = useMemo(() => {
    if (!data) return data;
    if (!pinnedRecordId) return data;
    return data.filter((c) => c.Record_ID === pinnedRecordId);
  }, [data, pinnedRecordId]);

  return (
    <>
      <p className="chart-note">
        Search checks: Record ID, sub-location, incident summary, full description, and keywords.
        Click a point on the map to show just that record here.
      </p>
      {pinnedRecordId ? (
        <div className="pin-banner">
          Showing only {pinnedRecordId} (clicked on map).
          <button onClick={clearPin}>Show all matching records</button>
        </div>
      ) : (
        <p className="chart-note">{data ? `${data.length} matching record${data.length === 1 ? "" : "s"}` : ""}</p>
      )}
      {isLoading && <p>Loading…</p>}
      {!isLoading && (!displayed || displayed.length === 0) && (
        <p className="empty-note">No conflict records match the current filters.</p>
      )}
      <div className="conflict-list">
        {displayed?.map((c) => {
          const hasCoords = typeof c.Latitude === "number" && typeof c.Longitude === "number";
          return (
            <div
              key={c.Record_ID}
              className={
                c.Record_ID === flyToRecordId
                  ? "conflict-list-item conflict-list-item-selected"
                  : "conflict-list-item"
              }
              onClick={() => hasCoords && selectConflict(c.Record_ID, c.Longitude as number, c.Latitude as number)}
              style={{ cursor: hasCoords ? "pointer" : "default" }}
              title={hasCoords ? "Click to zoom the map to this record" : "No coordinates available for this record"}
            >
              <div className="conflict-list-header">
                <span className="conflict-id">{c.Record_ID}</span>
                <span className="conflict-date">{c.Date_Start}</span>
              </div>
              <p className="conflict-summary">{c.Incident_Summary}</p>
              <div className="conflict-tags">
                {c.geo_subcounty && <span className="tag">{c.geo_subcounty as string}</span>}
                {c.topic_label && <span className="tag tag-topic">Theme: {String(c.topic_label)}</span>}
                {c.sentiment_label && (
                  <span className={`tag tag-sentiment tag-sentiment-${String(c.sentiment_label)}`}>
                    Sentiment: {String(c.sentiment_label)}
                  </span>
                )}
                {c.Is_Composite && <span className="tag tag-composite">chronic</span>}
                {!hasCoords && <span className="tag tag-nodata">no location</span>}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
