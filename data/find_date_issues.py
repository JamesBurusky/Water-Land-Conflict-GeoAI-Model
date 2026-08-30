"""
find_date_issues.py

Standalone helper -- NOT part of the numbered pipeline, meant to be
run directly from the data/ folder whenever you want to manually
review and correct the year-only-precision dates flagged during
dissertation review.

What it does
------------
Reads Final_Main_kenya_land_water_conflicts.csv, flags every record
dated exactly January 1st (the same day==1 AND month==1 pattern
02_conflict_cleaning.py's parse_dates() flags as date_precision_suspect,
confirmed to disproportionately reflect year-only source precision
defaulted to Jan 1 rather than a genuine date), and writes two CSVs:

  - conflicts_DATE_SUSPECT.csv  -- the flagged records, with the
    columns most useful for manually tracking down the real date
    (Source_Name, Source_URL, Incident_Summary, Full_Text_Description)
    sorted by source so records from the same article/ruling/report
    are grouped together for faster batch-checking.
  - conflicts_DATE_OK.csv       -- everything else, unchanged.

This script does NOT modify Final_Main_kenya_land_water_conflicts.csv.
Once you've corrected dates in conflicts_DATE_SUSPECT.csv, update the
Date_Start values in your master CSV directly (or ask for a follow-up
script to merge corrections back in) and re-run the pipeline from
02_conflict_cleaning.py onward.

Usage: place in the same folder as Final_Main_kenya_land_water_conflicts.csv
and run:  python find_date_issues.py
"""

from pathlib import Path
import pandas as pd

SOURCE_FILE = Path(__file__).parent / "Final_Main_kenya_land_water_conflicts.csv"
OUT_SUSPECT = Path(__file__).parent / "conflicts_DATE_SUSPECT.csv"
OUT_OK = Path(__file__).parent / "conflicts_DATE_OK.csv"

# Columns expected to exist, used only as a sanity check below (a
# warning if your CSV's structure differs from what the rest of this
# project assumes) -- NOT used to filter output columns. Both output
# files keep every column from the source CSV.
REVIEW_COLUMNS = [
    "Record_ID", "County", "Sub_Location", "Date_Start", "Date_End",
    "Incident_Summary", "Full_Text_Description",
    "Source_Name", "Source_URL", "Data_Source_Type", "Notes",
]


def read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    """Same fallback chain used throughout this pipeline -- this
    file's own encoding was confirmed to fail on plain UTF-8."""
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not read {path} with utf-8, cp1252, or latin-1.")


def main():
    if not SOURCE_FILE.exists():
        print(f"ERROR: {SOURCE_FILE} not found. Place this script in the same "
              f"folder as Final_Main_kenya_land_water_conflicts.csv and re-run.")
        return

    df = read_csv_robust(SOURCE_FILE)
    print(f"Loaded {len(df):,} records from {SOURCE_FILE.name}")

    parsed = pd.to_datetime(df["Date_Start"], format="%m/%d/%Y", errors="coerce")
    n_unparseable = parsed.isna().sum() - df["Date_Start"].isna().sum()
    if n_unparseable > 0:
        print(f"  NOTE: {n_unparseable} record(s) have a Date_Start that couldn't be "
              f"parsed as M/D/YYYY at all -- these are treated as NOT date-suspect "
              f"here (they're a separate, already-known issue) and land in the "
              f"'OK' file; review them separately if needed.")

    date_suspect = parsed.dt.month.eq(1) & parsed.dt.day.eq(1) & parsed.notna()

    # Every column is kept in BOTH files -- only which ROWS go where
    # differs. Earlier version of this script subset the suspect file
    # down to a "most useful for review" column list, which silently
    # dropped Latitude/Longitude/Casualties_Reported/etc. and left the
    # two files with different schemas. That breaks the actual next
    # step this script exists for: correct dates in the suspect file,
    # then merge those corrections back into the master CSV by
    # Record_ID. Keeping every column makes that a straightforward
    # Record_ID-keyed update instead of a partial-column merge.
    sort_cols = [c for c in ["Source_Name", "Source_URL"] if c in df.columns]
    suspect_df = df.loc[date_suspect].sort_values(by=sort_cols) if sort_cols else df.loc[date_suspect]
    ok_df = df.loc[~date_suspect]

    missing_cols = [c for c in REVIEW_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  NOTE: expected column(s) not found in this CSV: {missing_cols} "
              f"-- your file's structure may differ from what this script assumes.")

    suspect_df.to_csv(OUT_SUSPECT, index=False)
    ok_df.to_csv(OUT_OK, index=False)

    print(f"\n{len(suspect_df):,} / {len(df):,} records ({len(suspect_df)/len(df):.1%}) "
          f"flagged as date-suspect (dated exactly Jan 1) -> {OUT_SUSPECT.name}")
    print(f"{len(ok_df):,} / {len(df):,} records unaffected -> {OUT_OK.name}")
    print(f"\nSuspect file is sorted by Source_Name/Source_URL so records citing the "
          f"same article, ruling, or report are grouped together -- check one source, "
          f"fix every record that cites it, move to the next.")


if __name__ == "__main__":
    main()
