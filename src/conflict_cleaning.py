"""
conflict_cleaning.py

Cleans the raw conflict dataset (the dependent variable) and implements
the agreed temporal methodology:

    "Because conflict records often represent prolonged socio-
    environmental disputes rather than discrete events, each record is
    decomposed into onset, persistence, and severity. Onset uses the
    recorded start date. Persistence is represented via an exponential
    decay function anchored on Date_Start (never Date_End, to avoid
    leaking future information when later training on earlier years
    and testing on later years). Composite/chronic records are given a
    longer decay half-life rather than a hard end-date cutoff, so their
    sustained nature is reflected without referencing a specific future
    end point."

Three cleaning problems solved here, each because the raw data is
genuinely inconsistent in a way that would silently break downstream
steps if left alone:

1. Date_Start / Date_End are strings ("3/1/2016") -> parsed to real
   datetimes so they can be compared, subtracted, and joined against
   the Year/Month columns used elsewhere.
2. Casualties_Reported / Displaced_Persons mix plain integers with
   "69+", "5000+", and "Unknown" -> split into a numeric estimate, a
   is_lower_bound flag (the "+" cases), and a is_unknown flag, so
   "Unknown" is never silently coded as 0 (which would understate
   severity in exactly the chronic/composite cases where severity is
   highest).
3. No explicit "is this a discrete event or a chronic/composite
   situation" flag exists -> derived from two independent signals
   (long recorded duration, and text mentions of "composite"/
   "chronic"/"long-running" etc. in Notes), kept as separate columns so
   you can see which signal fired.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import ftfy

TEXT_COLUMNS_TO_REPAIR = [
    "Full_Text_Description", "Incident_Summary", "Notes",
    "Parties_Involved", "NLP_Keywords", "Source_Name",
]


def repair_mojibake(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Fixes double-encoding corruption (mojibake) in free-text fields --
    e.g. an em-dash or arrow that was UTF-8 encoded, decoded once as
    Latin-1/cp1252, then re-saved as UTF-8, producing garbage sequences
    like 'Ã¢â‚¬â€' in place of the original character. Different from
    the single-encoding UnicodeDecodeError already fixed in the phase
    scripts' read_csv_robust() -- that crashes loudly on load; this
    corrupts text silently, so it's easy to miss until you're reading
    NER/topic model output and see garbled characters inside real words.

    Uses ftfy ("fixes text for you"), the standard library for this.
    Run BEFORE any other text processing (NER, sentiment, topic
    modelling) -- corrupted bytes inside words can silently break
    tokenization for all of them.
    """
    out = df.copy()
    cols = columns or [c for c in TEXT_COLUMNS_TO_REPAIR if c in out.columns]
    n_changed_total = 0
    for col in cols:
        original = out[col].fillna("").astype(str)
        fixed = original.apply(ftfy.fix_text)
        n_changed = (fixed != original).sum()
        n_changed_total += n_changed
        if n_changed > 0:
            print(f"  repair_mojibake: fixed {n_changed} value(s) in '{col}'")
        out[col] = out[col].where(out[col].isna(), fixed)
    if n_changed_total == 0:
        print("  repair_mojibake: no mojibake detected -- nothing to fix")
    return out

# Records spanning more than this many days are treated as
# chronic/composite by duration alone, regardless of wording in Notes.
# 180 days (~6 months) was chosen as a starting assumption: this is a
# HYPERPARAMETER, not a fact -- test sensitivity to this threshold the
# same way you're already planning to test the decay half-life.
LONG_DURATION_THRESHOLD_DAYS = 180

# Text signals that a record documents a sustained/chronic situation
# rather than a discrete event, even when its duration alone doesn't
# clear the threshold above (e.g. a short-dated record whose Notes
# still describe it as part of an ongoing pattern).
COMPOSITE_KEYWORDS = re.compile(
    r"composite|chronic|long[- ]running|long[- ]term|multiple sources|"
    r"ongoing|sustained",
    flags=re.IGNORECASE,
)


def parse_dates(df: pd.DataFrame, start_col: str = "Date_Start",
                 end_col: str = "Date_End") -> pd.DataFrame:
    """
    Parses Date_Start/Date_End from strings to real datetimes.
    Unparseable dates become NaT (not silently dropped) and are
    reported, so you can see exactly which records need a manual look.

    Also flags a second, subtler problem found during dissertation
    review: 39.1% of ALL records in this dataset are dated exactly
    January 1st of some year -- against a baseline of 12-24% for
    day-equals-1 in other months (checked directly: February 12.5%,
    March 11.8%, June 23.9%, September 20.3%). That gap is the
    signature of source records where only the YEAR was known at
    compilation time, defaulted to January 1 rather than left blank --
    not a genuine concentration of incidents on that specific date.
    This matters beyond the seasonal-pattern chart: every monthly-
    granularity feature this project builds downstream (conflict
    persistence, severity-weighted persistence, decayed sentiment,
    topic diversity -- see panel_builder.py) is computed at sub-county-
    MONTH resolution, so a mis-dated record pollutes two months at
    once: false signal in January, missing signal in whatever month
    the incident actually happened in. `date_precision_suspect` flags
    this pattern (day==1 AND month==1) so panel_builder.py can exclude
    these records from monthly dynamic features specifically, while
    still counting them in year-level analysis, where being off by a
    day or two within the right year is a far smaller problem.
    """
    out = df.copy()
    out[f"{start_col}_parsed"] = pd.to_datetime(
        out[start_col], format="%m/%d/%Y", errors="coerce"
    )
    out[f"{end_col}_parsed"] = pd.to_datetime(
        out[end_col], format="%m/%d/%Y", errors="coerce"
    )

    for col, parsed_col in [(start_col, f"{start_col}_parsed"),
                             (end_col, f"{end_col}_parsed")]:
        failed = out[out[parsed_col].isna() & out[col].notna()]
        if len(failed) > 0:
            print(f"  WARNING: {len(failed)} value(s) in '{col}' could not "
                  f"be parsed as M/D/YYYY and are now NaT: "
                  f"{failed[col].tolist()}")

    out["duration_days"] = (
        out[f"{end_col}_parsed"] - out[f"{start_col}_parsed"]
    ).dt.days

    parsed = out[f"{start_col}_parsed"]
    out["date_precision_suspect"] = (
        parsed.dt.month.eq(1) & parsed.dt.day.eq(1) & parsed.notna()
    )
    n_suspect = out["date_precision_suspect"].sum()
    if n_suspect > 0:
        print(f"  date_precision_suspect: {n_suspect:,} / {len(out):,} records "
              f"({n_suspect/len(out):.1%}) dated exactly Jan 1 -- flagged as "
              f"likely year-only precision, not necessarily a genuine Jan 1 event. "
              f"panel_builder.py excludes these from monthly dynamic features by default.")

    return out


def parse_count_field(series: pd.Series) -> pd.DataFrame:
    """
    Splits a mixed-type count column (e.g. "0", "69+", "Unknown") into:
      - <name>_numeric: best-available numeric estimate (NaN if unknown)
      - <name>_is_lower_bound: True for "N+" values (the true count may
        be higher than recorded)
      - <name>_is_unknown: True for "Unknown" (kept as NaN, NOT coded
        as 0 -- coding it as 0 would understate severity precisely in
        the composite/chronic records where severity tends to be
        highest and least precisely counted)
    """
    s = series.astype(str).str.strip()
    is_unknown = s.str.lower().eq("unknown")
    is_lower_bound = s.str.endswith("+") & ~is_unknown
    numeric = pd.to_numeric(s.str.rstrip("+"), errors="coerce")
    numeric = numeric.where(~is_unknown, np.nan)
    return pd.DataFrame({
        "numeric": numeric,
        "is_lower_bound": is_lower_bound,
        "is_unknown": is_unknown,
    })


def clean_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Applies parse_count_field to Casualties_Reported and Displaced_Persons."""
    out = df.copy()
    for col in ["Casualties_Reported", "Displaced_Persons"]:
        parsed = parse_count_field(out[col])
        out[f"{col}_numeric"] = parsed["numeric"]
        out[f"{col}_is_lower_bound"] = parsed["is_lower_bound"]
        out[f"{col}_is_unknown"] = parsed["is_unknown"]
    return out


def flag_composite_records(df: pd.DataFrame,
                            notes_col: str = "Notes") -> pd.DataFrame:
    """
    Adds:
      - is_long_duration: duration_days > LONG_DURATION_THRESHOLD_DAYS
      - notes_mentions_chronic: keyword match in Notes text
      - Is_Composite: either signal fires (kept alongside the two
        components above so you can see which one drove the flag for
        any given record -- useful for a methodology footnote on how
        many records were flagged by duration alone vs. by wording)
    Requires parse_dates() to have already been run (needs duration_days).
    """
    out = df.copy()
    if "duration_days" not in out.columns:
        raise ValueError("Run parse_dates() first -- duration_days is required.")

    out["is_long_duration"] = out["duration_days"] > LONG_DURATION_THRESHOLD_DAYS
    out["notes_mentions_chronic"] = out[notes_col].fillna("").str.contains(
        COMPOSITE_KEYWORDS
    )
    out["Is_Composite"] = out["is_long_duration"] | out["notes_mentions_chronic"]
    return out


def compute_severity_score(df: pd.DataFrame) -> pd.Series:
    """
    A single severity score per record from casualties + displacement,
    log-scaled since both are extremely right-skewed (most records are
    0, a few are in the thousands) and a raw sum would let the largest
    few records dominate everything. Confidence_Score is NOT folded in
    here -- it measures how sure we are the record is accurate, not how
    severe the conflict was; use it separately as a data-quality filter
    if needed, not as part of severity itself.

    Records with BOTH counts unknown get NaN severity (not 0) -- a
    record with completely unknown impact should not be treated as
    equivalent to a documented zero-casualty, zero-displacement one.
    """
    casualties = df["Casualties_Reported_numeric"]
    displaced = df["Displaced_Persons_numeric"]
    both_unknown = df["Casualties_Reported_is_unknown"] & df["Displaced_Persons_is_unknown"]

    score = np.log1p(casualties.fillna(0)) + 0.5 * np.log1p(displaced.fillna(0))
    score = score.where(~both_unknown, np.nan)
    return score


def decayed_persistence(
    events: pd.DataFrame,
    eval_date: pd.Timestamp,
    half_life_days: float,
    composite_half_life_multiplier: float = 3.0,
    start_col: str = "Date_Start_parsed",
    composite_col: str = "Is_Composite",
    severity_col: str | None = None,
) -> float:
    """
    Core building block for the Phase 5 monthly panel: the decayed
    "how much conflict risk is still active here" score at eval_date,
    from a set of past events (already filtered to one sub-county).

    Only events with Date_Start <= eval_date contribute (never events
    starting after eval_date) -- this is what makes the feature safe
    for the train-on-earlier-years / test-on-later-years design in
    Phase 6: at no point does computing this value require knowing
    anything that happens after eval_date.

    Composite/chronic events decay more slowly (longer half-life) to
    reflect their sustained nature, WITHOUT referencing Date_End
    directly -- avoiding the leakage risk of using a documented future
    end point as a hard cutoff.

    If severity_col is given, returns the severity-WEIGHTED persistence
    (each event's decayed contribution multiplied by its severity
    score) instead of a plain occurrence-weighted sum.
    """
    past = events[events[start_col] <= eval_date]
    if len(past) == 0:
        return 0.0

    days_elapsed = (eval_date - past[start_col]).dt.days.clip(lower=0)
    effective_half_life = np.where(
        past[composite_col], half_life_days * composite_half_life_multiplier,
        half_life_days,
    )
    decay = np.exp(-np.log(2) * days_elapsed / effective_half_life)

    if severity_col is not None:
        weights = past[severity_col].fillna(0)
        return float((decay * weights).sum())
    return float(decay.sum())


def onset_flag(events: pd.DataFrame, eval_year: int, eval_month: int,
                start_col: str = "Date_Start_parsed") -> bool:
    """True if at least one event's Date_Start falls in this exact
    (year, month) -- the discrete 'a new conflict began here' signal,
    kept separate from the continuous persistence score above."""
    starts = events[start_col].dropna()
    return bool(((starts.dt.year == eval_year) & (starts.dt.month == eval_month)).any())
