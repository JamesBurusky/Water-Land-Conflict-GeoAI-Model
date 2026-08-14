import { AVAILABLE_HORIZONS, useHorizonStore } from "../store/useHorizonStore";
import { usePipelineStatus } from "../hooks/useApiData";

// Lives in the app toolbar (a plain flex row, proven reliable) rather
// than floating on top of the map -- an earlier version positioned
// this as a map overlay and it kept ending up hidden underneath the
// toolbar itself (the toolbar sits at a higher z-index with a near-
// opaque background, and both were anchored to the same top-left
// corner). A flex sibling of the other toolbar buttons can't suffer
// that failure mode: there's no stacking to get wrong.
export function HorizonSelector() {
  const horizonMonths = useHorizonStore((s) => s.horizonMonths);
  const setHorizonMonths = useHorizonStore((s) => s.setHorizonMonths);
  const { data: status } = usePipelineStatus();

  const isAvailable = (h: number) => {
    if (!status) return true;
    return status.step_12_conflict_risk_mapping_horizons[String(h)] ?? true;
  };

  const index = AVAILABLE_HORIZONS.indexOf(horizonMonths);
  const goPrev = () => index > 0 && setHorizonMonths(AVAILABLE_HORIZONS[index - 1]);
  const goNext = () => index < AVAILABLE_HORIZONS.length - 1 && setHorizonMonths(AVAILABLE_HORIZONS[index + 1]);
  const available = isAvailable(horizonMonths);

  return (
    <div className="toolbar-horizon" title="How many months ahead this prediction is forecasting">
      <span className="toolbar-horizon-label">Forecast horizon:</span>

      <button className="toolbar-horizon-arrow" onClick={goPrev} disabled={index === 0} aria-label="Shorter horizon">
        ‹
      </button>

      <span className="toolbar-horizon-value">
        {horizonMonths} months ahead
        {!available && <span className="toolbar-horizon-unavailable"> (not run)</span>}
      </span>

      <button
        className="toolbar-horizon-arrow"
        onClick={goNext}
        disabled={index === AVAILABLE_HORIZONS.length - 1}
        aria-label="Longer horizon"
      >
        ›
      </button>

      <div className="toolbar-horizon-dots">
        {AVAILABLE_HORIZONS.map((h) => (
          <button
            key={h}
            className={h === horizonMonths ? "toolbar-horizon-dot toolbar-horizon-dot-active" : "toolbar-horizon-dot"}
            onClick={() => setHorizonMonths(h)}
            title={`${h} months ahead${isAvailable(h) ? "" : " (not run)"}`}
            aria-label={`${h} months ahead`}
          />
        ))}
      </div>
    </div>
  );
}
