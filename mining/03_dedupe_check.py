"""
03_dedupe_check.py

Stage 3 of the data collection pipeline. Checks every row in the
review queue (stage 2's output) against:
  (a) every other row in the same review queue, and
  (b) your existing master dataset, if config.EXISTING_DATASET_PATH is set,
for likely near-duplicates: word-overlap (Jaccard) text similarity
above config.DEDUPE_SIMILARITY_THRESHOLD (0.30 by default -- see the
comment beside that setting in config.py for why this differs from,
and is not simply reusing, the 80% figure documented for the main
pipeline's own near-duplicate check), within the same county, within
a config.DEDUPE_DATE_WINDOW_DAYS date window (30 days by default).

This does not delete anything. It splits the review queue into two
files: rows flagged as likely duplicates (for you to inspect and
probably discard) and rows that appear to be new (still requiring the
same manual field-by-field review as everything else -- deduplication
only checks for repeat coverage of the same incident, it says nothing
about whether the suggested fields are accurate).

Usage:
    python 03_dedupe_check.py
"""

from datetime import datetime

import pandas as pd

import config
import utils


def text_similarity(a: str, b: str) -> float:
    """Word-overlap (Jaccard) similarity, not character-level sequence
    matching. Testing during development found character-level
    similarity (difflib.SequenceMatcher) badly under-scores the most
    common real duplicate scenario in a multi-source news compilation:
    two outlets covering the identical real event in different
    wording (e.g. 'three people died' vs 'three people were killed')
    scored only 55% character-similarity despite being the same
    incident, well under the intended threshold. Jaccard word overlap
    scored that same genuine pair at 45%, against 12% for two
    genuinely different incidents sharing only a topic and county --
    a much cleaner separation, which is why the threshold below is
    calibrated for this metric specifically."""
    words_a, words_b = set((a or "").lower().split()), set((b or "").lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def parse_date_safe(s: str):
    if not s:
        return None
    # suggested dates may be a ';'-joined list of NER guesses (e.g.
    # "14 March 2024; Tuesday") -- try each fragment and take the
    # first one that parses cleanly.
    for fragment in str(s).split(";"):
        try:
            return pd.to_datetime(fragment.strip(), errors="raise")
        except Exception:
            continue
    return None


def find_duplicates(new_rows: list[dict], existing_rows: list[dict]) -> dict:
    """Returns {index_in_new_rows: [match description, ...]}."""
    flags = {}
    all_comparisons = [(i, "existing", j, ex) for i in range(len(new_rows)) for j, ex in enumerate(existing_rows)]
    all_comparisons += [(i, "new", j, new_rows[j]) for i in range(len(new_rows)) for j in range(len(new_rows)) if j != i]

    for i, other_kind, j, other in all_comparisons:
        row = new_rows[i]
        if row.get("County") and other.get("County") and row["County"] != other["County"]:
            continue

        row_date = parse_date_safe(row.get("Date_Start", ""))
        other_date = parse_date_safe(other.get("Date_Start", ""))
        if row_date is not None and other_date is not None:
            if abs((row_date - other_date).days) > config.DEDUPE_DATE_WINDOW_DAYS:
                continue

        text_a = f"{row.get('Incident_Summary', '')} {row.get('Full_Text_Description', '')}"
        text_b = f"{other.get('Incident_Summary', '')} {other.get('Full_Text_Description', '')}"
        sim = text_similarity(text_a, text_b)
        if sim >= config.DEDUPE_SIMILARITY_THRESHOLD:
            flags.setdefault(i, []).append(
                f"{sim:.0%} similar to {other_kind} record "
                f"({other.get('Record_ID') or other.get('Source_URL', 'unknown')})"
            )
    return flags


def main():
    print("Stage 3: Checking for near-duplicates\n")
    if not config.REVIEW_QUEUE_PATH.exists():
        print(f"ERROR: {config.REVIEW_QUEUE_PATH} not found -- run 02_extract_and_suggest.py first.")
        return

    new_df = utils.read_csv_robust(config.REVIEW_QUEUE_PATH).fillna("")
    new_rows = new_df.to_dict("records")
    print(f"Loaded {len(new_rows)} rows from the review queue.")

    existing_rows = []
    if config.EXISTING_DATASET_PATH and config.EXISTING_DATASET_PATH.exists():
        existing_df = utils.read_csv_robust(config.EXISTING_DATASET_PATH).fillna("")
        existing_rows = existing_df.to_dict("records")
        print(f"Loaded {len(existing_rows)} rows from your existing dataset to check against.")
    else:
        print("No existing dataset configured (config.EXISTING_DATASET_PATH) -- "
              "checking new rows only against each other.")

    flags = find_duplicates(new_rows, existing_rows)
    print(f"\n{len(flags)} of {len(new_rows)} new rows flagged as likely duplicates.")

    flagged_rows, clear_rows = [], []
    for i, row in enumerate(new_rows):
        if i in flags:
            row = {**row, "duplicate_reasons": " | ".join(flags[i])}
            flagged_rows.append(row)
        else:
            clear_rows.append(row)

    pd.DataFrame(flagged_rows).to_csv(config.DEDUPE_FLAGGED_PATH, index=False)
    pd.DataFrame(clear_rows).to_csv(config.DEDUPE_CLEAR_PATH, index=False)

    print(f"\nSaved {len(flagged_rows)} likely-duplicate rows to {config.DEDUPE_FLAGGED_PATH} (inspect and probably discard)")
    print(f"Saved {len(clear_rows)} rows with no detected duplicate to {config.DEDUPE_CLEAR_PATH}")
    print("\nNeither file is verified yet -- deduplication only checks for repeat coverage "
          "of the same incident. Every row in both files still needs the same manual "
          "field-by-field review described in 02_extract_and_suggest.py before merging "
          "anything into your master dataset.")


if __name__ == "__main__":
    main()
