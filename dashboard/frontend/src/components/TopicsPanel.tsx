import { useMemo } from "react";
import { useTopics, useConflictsForTopicBreakdown } from "../hooks/useApiData";
import { useFilterStore } from "../store/useFilterStore";

export function TopicsPanel() {
  const { data: topicInfo, isLoading: topicsLoading } = useTopics();
  const { data: conflicts, isLoading: conflictsLoading } = useConflictsForTopicBreakdown();
  const { topicId, setTopicId } = useFilterStore();

  // Counts recomputed from the CURRENTLY FILTERED conflict set (region/
  // date/search applied, topic itself excluded -- see the hook) rather
  // than the static global topic_info.csv counts, so this breakdown
  // reflects whatever selection is currently active elsewhere in the
  // dashboard, matching "dynamic on filters/searches/selections".
  const counts = useMemo(() => {
    const map = new Map<number, number>();
    for (const c of conflicts ?? []) {
      if (typeof c.topic_id === "number") {
        map.set(c.topic_id, (map.get(c.topic_id) ?? 0) + 1);
      }
    }
    return map;
  }, [conflicts]);

  if (topicsLoading || conflictsLoading) return <p>Loading…</p>;
  if (!topicInfo) {
    return <p className="empty-note">Not yet available — run 05_topic_modelling.py in the pipeline.</p>;
  }

  const topics = topicInfo
    .filter((t) => t.Topic !== -1)
    .map((t) => ({ ...t, currentCount: counts.get(t.Topic) ?? 0 }))
    .sort((a, b) => b.currentCount - a.currentCount);

  return (
    <>
      <ul className="topic-list">
        {topics.map((t) => (
          <li
            key={t.Topic}
            className={topicId === t.Topic ? "topic-item topic-item-active" : "topic-item"}
            onClick={() => setTopicId(topicId === t.Topic ? undefined : t.Topic)}
          >
            <span className="topic-name">{t.Name}</span>
            <span className="topic-count">{t.currentCount}</span>
          </li>
        ))}
      </ul>
      <p className="chart-note">
        Counts reflect current region/date/search filters. Click a theme to filter the whole dashboard by it.
      </p>
    </>
  );
}
