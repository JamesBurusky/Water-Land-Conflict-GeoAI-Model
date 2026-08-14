// Word/keyword frequency for the NLP word cloud widget. Prefers
// NLP_Keywords (Phase 2's cleaner, pipeline-extracted terms) when
// present; falls back to a simple tokenization of Incident_Summary
// for records where it isn't (e.g. if 04_nlp_pipeline.py wasn't run
// on this record for some reason).

import type { ConflictRecord } from "../types/api";

const STOPWORDS = new Set([
  "the", "and", "for", "that", "this", "with", "from", "were", "have",
  "has", "had", "was", "are", "been", "being", "their", "they", "them",
  "which", "when", "while", "into", "over", "under", "after", "before",
  "county", "kenya", "kenyan", "area", "region", "report", "reported",
  "according", "said", "stated", "also", "would", "could", "should",
  "documented", "confirmed", "between", "during", "since", "such",
]);

export interface WordCount {
  text: string;
  count: number;
}

export function computeWordFrequencies(records: ConflictRecord[], maxWords = 60): WordCount[] {
  const freq = new Map<string, number>();

  for (const record of records) {
    let terms: string[] = [];
    if (record.NLP_Keywords) {
      terms = String(record.NLP_Keywords)
        .split(",")
        .map((t) => t.trim().toLowerCase())
        .filter((t) => t.length > 2);
    } else if (record.Incident_Summary) {
      terms = String(record.Incident_Summary)
        .toLowerCase()
        .replace(/[^a-z\s]/g, " ")
        .split(/\s+/)
        .filter((t) => t.length > 3 && !STOPWORDS.has(t));
    }
    for (const term of terms) {
      freq.set(term, (freq.get(term) ?? 0) + 1);
    }
  }

  return Array.from(freq.entries())
    .map(([text, count]) => ({ text, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, maxWords);
}
