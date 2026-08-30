"""
run_pipeline.py

Runs every pipeline script in the correct dependency order, one after
another, stopping immediately if any step fails (rather than plowing
ahead and producing confusing downstream errors from missing input
files). Each step runs as its own subprocess -- this matters because
several scripts (topic modelling, temporal-safe refit) load heavy ML
libraries; running them in-process one after another in a single
Python session risks memory building up and one step's global state
leaking into the next. Subprocesses start clean every time.

Usage:
    python run_pipeline.py                  # run everything
    python run_pipeline.py --from 06        # resume from step 06 onward
    python run_pipeline.py --only 10 12     # run just these steps
    python run_pipeline.py --skip 05 07     # run everything except these
                                             # (05/07 need internet -- see below)

Steps 05 (topic modelling) and 07 (temporal-safe topic refit) download
an embedding model on first run and need internet access -- if you're
offline, use --skip 05 07 and re-run them later; every other step is
fully offline.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# (step number, script filename, human label, needs_internet)
PIPELINE_STEPS = [
    ("01", "01_phase1_data_audit.py", "Phase 1 - data audit", False),
    ("02", "02_conflict_cleaning.py", "Phase 1b - conflict cleaning", False),
    ("03", "03_deduplication_check.py", "Phase 1c - deduplication", False),
    ("04", "04_nlp_pipeline.py", "Phase 2 - NLP pipeline (text/sentiment/NER)", False),
    ("05", "05_topic_modelling.py", "Phase 2 - topic modelling (BERTopic)", True),
    ("06", "06_spatial_feature_engineering.py", "Phase 3 - spatial feature engineering", False),
    ("07", "07_topic_refit_temporal_safe.py", "Phase 6 fix - temporal-safe topic refit", True),
    ("08", "08_nlp_panel_features.py", "Phase 3 - NLP panel features", False),
    ("09", "09_exploratory_spatial_analysis.py", "Phase 4 - exploratory spatial analysis", False),
    ("10", "10_ml_modeling.py", "Phase 6 - multi-horizon ML modelling", False),
    ("11", "11_shap_interpretability.py", "Phase 7 - SHAP interpretability", False),
    ("12", "12_conflict_risk_mapping.py", "Phase 8 - multi-horizon risk mapping", False),
    ("13", "13_land_water_relationship_analysis.py", "Land-water relationship analysis", False),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GeoAI conflict pipeline end-to-end.")
    parser.add_argument("--from", dest="from_step", default=None,
                         help="Resume from this step number onward (e.g. 06)")
    parser.add_argument("--only", nargs="+", default=None,
                         help="Run only these step numbers (e.g. --only 10 12)")
    parser.add_argument("--skip", nargs="+", default=[],
                         help="Skip these step numbers (e.g. --skip 05 07 if offline)")
    parser.add_argument("--continue-on-error", action="store_true",
                         help="Don't stop on the first failure -- run every remaining "
                              "step regardless (useful for a first exploratory pass; "
                              "NOT recommended for a real run, since a failed early step "
                              "usually means every later step will fail too on missing input)")
    return parser.parse_args()


def select_steps(args) -> list[tuple[str, str, str, bool]]:
    steps = PIPELINE_STEPS
    if args.only:
        wanted = set(args.only)
        steps = [s for s in steps if s[0] in wanted]
    elif args.from_step:
        started = False
        filtered = []
        for s in steps:
            if s[0] == args.from_step:
                started = True
            if started:
                filtered.append(s)
        steps = filtered
    if args.skip:
        skip_set = set(args.skip)
        steps = [s for s in steps if s[0] not in skip_set]
    return steps


def main():
    args = parse_args()
    steps = select_steps(args)

    if not steps:
        print("No steps selected -- check your --from/--only/--skip arguments.")
        sys.exit(1)

    print("=" * 70)
    print("GEOAI CONFLICT PIPELINE -- FULL RUN")
    print("=" * 70)
    print(f"Steps to run: {', '.join(s[0] for s in steps)}")
    internet_steps = [s[0] for s in steps if s[3]]
    if internet_steps:
        print(f"NOTE: step(s) {', '.join(internet_steps)} need internet access "
              f"(download an embedding model on first run).")
    print()

    results = []
    overall_start = time.time()

    for number, filename, label, needs_internet in steps:
        script_path = Path(filename)
        if not script_path.exists():
            print(f"[{number}] SKIPPED -- {filename} not found in the current directory")
            results.append((number, label, "SKIPPED (file not found)", 0.0))
            continue

        print(f"\n{'=' * 70}")
        print(f"[{number}] {label}")
        print(f"    Running: python {filename}")
        print("=" * 70)

        step_start = time.time()
        result = subprocess.run([sys.executable, filename])
        elapsed = time.time() - step_start

        if result.returncode == 0:
            print(f"\n[{number}] DONE in {elapsed:.1f}s")
            results.append((number, label, "OK", elapsed))
        else:
            print(f"\n[{number}] FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
            results.append((number, label, f"FAILED (exit {result.returncode})", elapsed))
            if not args.continue_on_error:
                print("\nStopping here -- a failed step usually means later steps will "
                      "fail too on missing input files. Fix this step and re-run with "
                      f"'--from {number}' to resume from here, or pass --continue-on-error "
                      "to run everything regardless.")
                break

    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print("PIPELINE RUN SUMMARY")
    print("=" * 70)
    for number, label, status, elapsed in results:
        status_marker = "OK" if status == "OK" else "XX"
        print(f"  [{status_marker}] {number}  {label:<45s} {status:<25s} {elapsed:6.1f}s")
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    any_failed = any(status.startswith("FAILED") for _, _, status, _ in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
