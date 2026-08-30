"""
dedup_check.py

Two different duplicate problems, checked separately because they need
different logic:

1. EXACT duplicates (any dataset): identical rows, or repeated ID
   values, from re-exports/re-merges. Cheap, unambiguous, safe to
   report and usually safe to drop.

2. NEAR-duplicate conflict records: the SAME real-world incident
   documented independently by two different sources (GDELT, ACLED,
   ReliefWeb, Kenya Law, NGO/media all covering the same event) with
   different Record_IDs and slightly different wording. These are
   NEVER auto-dropped -- text similarity alone is unreliable (two
   genuinely different conflicts in the same county can use similar
   generic phrasing), so candidates are flagged for manual review only,
   requiring similarity AND spatial/temporal proximity to agree before
   a pair is even surfaced.
"""

from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz


def check_exact_duplicates(df: pd.DataFrame, id_col: str | None = None,
                            label: str = "dataset") -> pd.DataFrame:
    """
    Reports (does not drop) exact full-row duplicates, and separately,
    duplicate values in id_col if given (e.g. Record_ID / WRA_ID) --
    a repeated ID with DIFFERENT row contents is a different, often
    more serious problem (conflicting records under one ID) than a
    fully identical repeated row, so the two are reported separately.
    """
    full_dupes = df[df.duplicated(keep=False)]
    print(f"  {label}: {len(full_dupes)} row(s) involved in exact full-row duplicates")

    if id_col and id_col in df.columns:
        id_dupes = df[df.duplicated(subset=[id_col], keep=False)]
        n_id_dupes = len(id_dupes)
        n_full_among_id = len(id_dupes[id_dupes.index.isin(full_dupes.index)])
        n_conflicting = n_id_dupes - n_full_among_id
        print(f"  {label}: {n_id_dupes} row(s) share a repeated '{id_col}' value "
              f"({n_conflicting} of these have DIFFERING content under the same "
              f"ID -- needs manual resolution, not a simple drop-duplicate)")
        return id_dupes

    return full_dupes


def inspect_id_collisions(df: pd.DataFrame, id_col: str = "Record_ID",
                           display_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Returns every row involved in a repeated id_col value, sorted so
    each colliding group sits together -- for eyeballing whether the
    collisions are a genuine ID-generation bug (differing content under
    one ID, most likely explanation if exact-row-duplicate count is 0)
    versus true duplicate rows that just weren't caught by the exact
    check (e.g. differ only in a trailing whitespace or NaN vs "").
    """
    cols = display_cols or [c for c in df.columns if not c.startswith("_")]
    dupes = df[df.duplicated(subset=[id_col], keep=False)].sort_values(id_col)
    return dupes[cols]


def resolve_id_collisions(df: pd.DataFrame, id_col: str = "Record_ID") -> pd.DataFrame:
    """
    Mints a guaranteed-unique ID for every row, WITHOUT discarding any
    data or guessing which record is "correct" -- that judgment call
    belongs to a human who can read the Full_Text_Description /
    Source_URL for each colliding pair.

    The ORIGINAL id_col value is preserved as f"Original_{id_col}" for
    traceability. For the first occurrence of any id_col value, the ID
    is left unchanged; subsequent occurrences get a letter suffix
    (e.g. KEN-NBI-001, KEN-NBI-001-B, KEN-NBI-001-C, ...) so the
    relationship to the original grouping stays visible rather than
    being hidden behind an opaque new ID.

    This does NOT decide whether the collided records are genuinely
    different conflicts (keep both, now safely unique) or accidental
    re-entries of the same one (should be merged/dropped) -- run
    inspect_id_collisions() first and make that call manually; this
    function only guarantees uniqueness so downstream joins stop being
    silently wrong regardless of which resolution you choose.
    """
    out = df.copy()
    out[f"Original_{id_col}"] = out[id_col]
    suffix_letters = "BCDEFGHIJKLMNOPQRSTUVWXYZ"

    occurrence = out.groupby(id_col).cumcount()
    needs_suffix = occurrence > 0
    # occurrence=1 -> 'B', occurrence=2 -> 'C', ... (occurrence=0 unchanged)
    suffixes = occurrence[needs_suffix].map(lambda n: suffix_letters[n - 1])
    out.loc[needs_suffix, id_col] = (
        out.loc[needs_suffix, id_col].astype(str) + "-" + suffixes
    )
    n_renamed = needs_suffix.sum()
    print(f"  Resolved {n_renamed} colliding ID(s): original values preserved "
          f"in 'Original_{id_col}', new unique values in '{id_col}'")
    return out


def find_near_duplicate_conflicts(
    df: pd.DataFrame,
    text_col: str = "Incident_Summary",
    county_col: str = "County",
    start_col: str = "Date_Start_parsed",
    end_col: str = "Date_End_parsed",
    similarity_threshold: float = 80.0,
    max_gap_days: int = 30,
) -> pd.DataFrame:
    """
    Flags CANDIDATE near-duplicate conflict record pairs for manual
    review. A pair is only surfaced if ALL of these agree:
      - same County
      - text similarity (token_sort_ratio, robust to word reordering)
        >= similarity_threshold
      - date ranges overlap, or are within max_gap_days of each other

    Requires parse_dates() to have already been run on df (needs
    start_col/end_col as real datetimes).

    Returns a dataframe of candidate pairs with their similarity score
    and both records' key fields side by side, sorted by similarity
    descending -- review from the top down. NOTHING is dropped
    automatically; that decision needs a human looking at both
    Full_Text_Description / Source_URL fields to confirm it's really
    the same incident and not two similar-sounding but distinct ones.
    """
    candidates = []
    counties = df[county_col].dropna().unique()
    for county in counties:
        group = df[df[county_col] == county].reset_index()
        n = len(group)
        n_pairs = n * (n - 1) // 2
        print(f"    scanning {county}: {n} records, {n_pairs:,} pairs to compare...")
        for i in range(n):
            for j in range(i + 1, n):
                a, b = group.loc[i], group.loc[j]

                text_a, text_b = str(a[text_col]), str(b[text_col])
                score = fuzz.token_sort_ratio(text_a, text_b)
                if score < similarity_threshold:
                    continue

                # Date proximity: overlapping ranges, or starts within max_gap_days
                a_start, a_end = a[start_col], a[end_col]
                b_start, b_end = b[start_col], b[end_col]
                if pd.isna(a_start) or pd.isna(b_start):
                    continue
                overlap = (a_start <= b_end) and (b_start <= a_end)
                gap = abs((a_start - b_start).days)
                if not (overlap or gap <= max_gap_days):
                    continue

                candidates.append({
                    "County": county,
                    "similarity_score": score,
                    "record_id_a": a.get("Record_ID"),
                    "record_id_b": b.get("Record_ID"),
                    "text_a": text_a,
                    "text_b": text_b,
                    "date_start_a": a_start, "date_start_b": b_start,
                    "source_a": a.get("Source_Name"),
                    "source_b": b.get("Source_Name"),
                })

    result = pd.DataFrame(candidates)
    if len(result) > 0:
        result = result.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    return result
