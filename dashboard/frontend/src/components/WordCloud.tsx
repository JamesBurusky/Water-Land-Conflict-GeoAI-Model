import { useMemo } from "react";
import { useConflictsForAggregation } from "../hooks/useApiData";
import { computeWordFrequencies } from "../utils/wordFrequency";
import { colorForRatio } from "../utils/colors";

// A lightweight tag-cloud rendering (font size scaled by frequency, CSS
// flex-wrap layout) rather than a spiral-packed word cloud via a
// library like d3-cloud -- deliberately, to avoid pulling in a
// heavier, layout-fragile dependency that's much harder to verify
// without being able to visually test it. This is simpler and more
// predictable to reason about, while still being a genuine word cloud.
export function WordCloud() {
  const { data: conflicts, isLoading } = useConflictsForAggregation();

  const words = useMemo(() => computeWordFrequencies(conflicts ?? []), [conflicts]);

  if (isLoading) return <p>Loading…</p>;
  if (words.length === 0) return <p className="empty-note">No text data for the current selection.</p>;

  const maxCount = words[0].count;
  const minCount = words[words.length - 1].count;
  const range = Math.max(maxCount - minCount, 1);

  return (
    <div className="word-cloud">
      {words.map((w) => {
        const ratio = (w.count - minCount) / range;
        const fontSize = 12 + ratio * 26; // 12px .. 38px
        return (
          <span
            key={w.text}
            className="word-cloud-word"
            style={{ fontSize: `${fontSize}px`, color: colorForRatio(ratio) }}
            title={`${w.text}: ${w.count} occurrence${w.count === 1 ? "" : "s"}`}
          >
            {w.text}
          </span>
        );
      })}
    </div>
  );
}
