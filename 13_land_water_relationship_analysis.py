# %% [markdown]
# # Land–Water Conflict Relationship Analysis
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# Addresses the explicit research question: this study covers both
# land and water conflicts, and part of its purpose is understanding
# whether/how they relate. Classifies every conflict record as
# Land-only / Water-only / Mixed / Neither (keyword heuristic -- see
# src/land_water_analysis.py for the exact terms and the documented
# limitations of this approach), then analyzes how that classification
# relates to severity, chronicity, county, and time.
#
# Run this after 03_deduplication_check.py (needs conflict_cleaned.csv)
# — ideally also after 04_nlp_pipeline.py for severity_score and
# Is_Composite, which come from the NLP-enriched data.

# %%
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, file-saving backend -- set BEFORE pyplot
    # is imported, and before pipeline_utils or anything else can import it first.
    # Without this, matplotlib auto-selects an interactive GUI backend (TkAgg on
    # Windows, since Tkinter ships with the standard installer), which creates
    # Tkinter-backed figure objects even though this pipeline only ever calls
    # plt.savefig(), never plt.show(). Those Tk objects are not thread-safe, and
    # this project's model training uses joblib parallelism (n_jobs=-1 in
    # GridSearchCV/RandomizedSearchCV) -- when Python garbage-collects a leftover
    # Tk figure from a worker thread instead of the main thread, Tkinter raises
    # "RuntimeError: main thread is not in main loop" during its own cleanup.
    # Confirmed as the cause of exactly that error during a real Windows run of
    # this pipeline. Agg has no GUI/threading dependency at all, so this class of
    # error becomes structurally impossible, not just less likely.
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from conflict_cleaning import parse_dates, clean_counts, flag_composite_records, compute_severity_score
from land_water_analysis import (
    add_domain_classification, domain_severity_summary,
    domain_composite_crosstab, domain_by_county, domain_yearly_trend,
)
from output_paths import step_dir

pd.set_option("display.max_columns", None)

STEP_NAME = "13_land_water_relationship_analysis"
OUT_DIR = step_dir(STEP_NAME)

NLP_ENRICHED_PATH = step_dir("04_nlp_pipeline") / "conflict_nlp_enriched.csv"
CONFLICT_CLEANED_PATH = step_dir("03_deduplication_check") / "conflict_cleaned.csv"

DOMAIN_COLORS = {"Land-only": "#f4a261", "Water-only": "#2a9d8f",
                  "Mixed": "#264653", "Neither": "#dddddd"}

timer = Timer()
timer.__enter__()
log("=== LAND-WATER RELATIONSHIP ANALYSIS STARTED ===")

# %% [markdown]
# ## 1. Load and classify

# %%
log("STAGE 1/3: Loading conflict data and classifying land/water domain...")
if NLP_ENRICHED_PATH.exists():
    conflict = pd.read_csv(NLP_ENRICHED_PATH)
    print(f"  Using {NLP_ENRICHED_PATH} (has severity_score, Is_Composite already)")
elif CONFLICT_CLEANED_PATH.exists():
    print(f"  {NLP_ENRICHED_PATH} not found -- using {CONFLICT_CLEANED_PATH} and "
          f"computing severity/composite flags here instead.")
    conflict = pd.read_csv(CONFLICT_CLEANED_PATH)
    conflict = parse_dates(conflict) if "Date_Start_parsed" not in conflict.columns else conflict
    conflict = clean_counts(conflict)
    conflict = flag_composite_records(conflict)
    conflict["severity_score"] = compute_severity_score(conflict)
else:
    print(f"  Neither {NLP_ENRICHED_PATH} nor {CONFLICT_CLEANED_PATH} found -- "
          f"run 01/02/03 (and ideally 04) first.")
    raise SystemExit(1)

classified = add_domain_classification(conflict)
print("\n  Domain distribution:")
print(classified["conflict_domain"].value_counts().to_string())
print(f"  ({classified['conflict_domain'].value_counts(normalize=True).mul(100).round(1).to_dict()})")

classified.to_csv(OUT_DIR / "conflict_with_domain.csv", index=False)
log(f"STAGE 1/3: DONE. Saved -> {OUT_DIR / 'conflict_with_domain.csv'}\n")

# %% [markdown]
# ## 2. Relationship analysis
#
# Severity by domain, chronicity by domain (with a chi-square test of
# independence), domain by county, and the domain balance over time.

# %%
log("STAGE 2/3: Running relationship analysis...")

severity_summary = domain_severity_summary(classified)
print("\n  Severity by domain:")
print(severity_summary.to_string(index=False))
severity_summary.to_csv(OUT_DIR / "severity_by_domain.csv", index=False)

composite_crosstab, chi2_result = domain_composite_crosstab(classified)
print("\n  Chronic/composite status by domain:")
print(composite_crosstab.to_string())
if chi2_result:
    print(f"\n  Chi-square test of independence: chi2={chi2_result['chi2']:.3f}, "
          f"p={chi2_result['p_value']:.4f}, dof={chi2_result['dof']}")
    if "warning" in chi2_result:
        print(f"  WARNING: {chi2_result['warning']}")
    elif chi2_result["p_value"] < 0.05:
        print("  -> Statistically significant association between domain and "
              "chronic/composite status (p < 0.05)")
    else:
        print("  -> No statistically significant association detected (p >= 0.05) "
              "-- don't claim domain predicts chronicity without noting this")
else:
    print("  (Not enough categories/data for a chi-square test)")
composite_crosstab.to_csv(OUT_DIR / "domain_composite_crosstab.csv")

county_crosstab = domain_by_county(classified)
print("\n  Domain by county:")
print(county_crosstab.to_string())
county_crosstab.to_csv(OUT_DIR / "domain_by_county.csv")

if "Date_Start_parsed" in classified.columns:
    yearly_trend = domain_yearly_trend(classified)
    yearly_trend.to_csv(OUT_DIR / "domain_yearly_trend.csv")
    print(f"\n  Yearly domain trend saved -> {OUT_DIR / 'domain_yearly_trend.csv'}")
else:
    yearly_trend = None

log("STAGE 2/3: DONE.\n")

# %% [markdown]
# ## 3. Visualizations

# %%
log("STAGE 3/3: Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

domain_counts = classified["conflict_domain"].value_counts()
axes[0, 0].bar(domain_counts.index, domain_counts.values,
               color=[DOMAIN_COLORS.get(d, "#999") for d in domain_counts.index])
axes[0, 0].set_title("Conflict domain distribution")
axes[0, 0].set_ylabel("Record count")

if len(severity_summary) > 0:
    axes[0, 1].bar(severity_summary["conflict_domain"], severity_summary["mean"],
                   yerr=severity_summary["std"].fillna(0),
                   color=[DOMAIN_COLORS.get(d, "#999") for d in severity_summary["conflict_domain"]])
    axes[0, 1].set_title("Mean severity by domain (± std)")
    axes[0, 1].set_ylabel("Severity score")
else:
    axes[0, 1].axis("off")

county_crosstab.plot(kind="bar", stacked=True, ax=axes[1, 0],
                      color=[DOMAIN_COLORS.get(c, "#999") for c in county_crosstab.columns])
axes[1, 0].set_title("Domain by county")
axes[1, 0].set_ylabel("Record count")
axes[1, 0].legend(fontsize=8)

if yearly_trend is not None and len(yearly_trend) > 0:
    for domain in yearly_trend.columns:
        axes[1, 1].plot(yearly_trend.index, yearly_trend[domain], marker="o",
                         label=domain, color=DOMAIN_COLORS.get(domain, "#999"))
    axes[1, 1].set_title("Domain balance over time")
    axes[1, 1].set_ylabel("Record count")
    axes[1, 1].legend(fontsize=8)
else:
    axes[1, 1].axis("off")

plt.tight_layout()
save_and_display(fig, OUT_DIR / "land_water_relationship.png")

log("STAGE 3/3: DONE.\n")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("LAND-WATER RELATIONSHIP ANALYSIS COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
Files written to {OUT_DIR}/:
  1. conflict_with_domain.csv       - every record + conflict_domain +
                                       matched_land_terms/matched_water_terms
                                       (for auditing WHY a record was classified
                                       a given way)
  2. severity_by_domain.csv         - mean/median severity per domain
  3. domain_composite_crosstab.csv  - chronic status x domain + chi-square test
  4. domain_by_county.csv           - domain distribution per county
  5. domain_yearly_trend.csv        - domain balance over time
  6. land_water_relationship.png    - all four charts above

REMEMBER: conflict_domain is a keyword heuristic (see
src/land_water_analysis.py for the exact term lists), not a ground-truth
label. Report it as such in your methodology -- e.g. spot-check a sample
of "matched_land_terms"/"matched_water_terms" against the original
Incident_Summary to confirm the classification looks reasonable before
treating the resulting statistics as authoritative.
""")
