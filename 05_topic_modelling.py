# %% [markdown]
# # Phase 2 (step 4) — Topic Modelling (BERTopic)
#
# GeoAI + NLP for Spatio-Temporal Prediction of Water-Land Conflicts in Kenya
#
# **Requires internet access** — downloads a sentence-transformer
# embedding model (~90MB) from HuggingFace on first run. This is the
# one Phase 2 script that could NOT be validated end-to-end in the
# sandbox this project was built in (no HuggingFace access there,
# confirmed by direct test) — the plumbing (clustering, topic
# extraction, labeling) was validated using an offline TF-IDF
# substitute, but the REAL semantic topic quality depends on this
# model, which only your local run can confirm.
#
# Run this after `04_nlp_pipeline.py`.

# %%
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from pipeline_utils import log, save_and_display, Timer
from topic_modeling import fit_topics

pd.set_option("display.max_columns", None)

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

NLP_ENRICHED_PATH = OUT_DIR / "conflict_nlp_enriched.csv"

# Tunable -- see the note in src/topic_modeling.py:build_topic_model.
# 15 is a reasonable starting point for hundreds-to-low-thousands of
# records; if you get too few/many topics on the full run, adjust and
# re-run (re-fitting is fast once you're past the one-time model
# download).
MIN_TOPIC_SIZE = 15

timer = Timer()
timer.__enter__()
log("=== PHASE 2 STEP 4: TOPIC MODELLING STARTED ===")

# %% [markdown]
# ## 1. Load NLP-enriched conflict data

# %%
log("STAGE 1/2: Loading NLP-enriched conflict data...")
if not NLP_ENRICHED_PATH.exists():
    print(f"  {NLP_ENRICHED_PATH} not found -- run 04_nlp_pipeline.py first.")
    raise SystemExit(1)
conflict = pd.read_csv(NLP_ENRICHED_PATH)
print(f"  Loaded {len(conflict):,} records")
log("STAGE 1/2: DONE.\n")

# %% [markdown]
# ## 2. Fit BERTopic
#
# Runs on the RAW Full_Text_Description (not the lemmatized
# clean_tokens) — the embedding model needs natural sentence structure
# to understand context; see the design note in src/topic_modeling.py
# for why pre-lemmatizing would hurt this step.
#
# This cell downloads the embedding model on first run — if it hangs
# here for a long time with no output, check your internet connection
# (this is the ONE step in the whole pipeline that genuinely needs it).

# %%
log("STAGE 2/2: Fitting BERTopic (downloading embedding model on first "
    "run if needed -- requires internet)...")
result, topic_model, topic_info = fit_topics(
    conflict, text_col="Full_Text_Description",
    embedding_backend="sentence_transformer",
    min_topic_size=MIN_TOPIC_SIZE,
)

n_outliers = (result["topic_id"] == -1).sum()
print(f"\n  {len(topic_info) - 1} topic(s) found (excluding the outlier group)")
print(f"  Outlier rate: {n_outliers}/{len(result)} ({n_outliers/len(result):.1%}) -- "
      f"records too dissimilar from any dominant theme to group confidently. "
      f"This is normal; a VERY high rate (>40-50%) would suggest "
      f"MIN_TOPIC_SIZE is too large for your data and should be lowered.")
print("\n  Topic overview:")
print(topic_info[["Topic", "Count", "Name"]].to_string())

result.to_csv(OUT_DIR / "conflict_topics.csv", index=False)
topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False)
topic_model.save(str(OUT_DIR / "bertopic_model"), serialization="safetensors",
                  save_ctfidf=True)
log(f"STAGE 2/2: DONE. Saved -> {OUT_DIR / 'conflict_topics.csv'}, "
    f"{OUT_DIR / 'topic_info.csv'}, {OUT_DIR / 'bertopic_model'}/\n")

# %% [markdown]
# ## 2a. Reduce outliers
#
# Records land in the -1 "outlier" group when they're too dissimilar
# from any cluster to group confidently -- but a high outlier rate
# (yours came back around 24%) usually means many of those records DO
# belong somewhere, just not tightly enough for the original
# clustering to commit. `reduce_outliers` reassigns each outlier to
# its nearest real topic using the c-TF-IDF strategy -- comparing each
# outlier document's word-frequency profile against each topic's, no
# re-embedding needed (fast, and works even without internet access).

# %%
log("STAGE 2a: Reassigning outlier records to their nearest topic...")
docs = conflict["Full_Text_Description"].fillna("").astype(str).tolist()
new_topics = topic_model.reduce_outliers(docs, result["topic_id"].tolist(), strategy="c-tf-idf")
topic_model.update_topics(docs, topics=new_topics, vectorizer_model=topic_model.vectorizer_model)
result["topic_id"] = new_topics
topic_info = topic_model.get_topic_info()
id_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))
result["topic_label"] = result["topic_id"].map(id_to_name)

n_outliers_after = (result["topic_id"] == -1).sum()
print(f"  Outliers before: {n_outliers}/{len(result)} ({n_outliers/len(result):.1%})")
print(f"  Outliers after:  {n_outliers_after}/{len(result)} ({n_outliers_after/len(result):.1%})")
log("STAGE 2a: DONE.\n")

# %% [markdown]
# ## 2b. Hierarchical topic reduction
#
# 20 raw topics is usually too granular for a thesis "dominant conflict
# themes" section -- several will be near-duplicates (variations on
# "water company disputes") or single-incident micro-clusters. Rather
# than merging by eyeballing keyword lists, this uses BERTopic's own
# hierarchical clustering of the topics' embeddings to show which ones
# are ACTUALLY closest together, and reduces to TARGET_N_TOPICS broader
# themes. Both the fine-grained and coarse topic assignments are kept
# as separate columns -- use the coarse one for the thesis narrative,
# keep the fine-grained one available if you want more granular
# features later.

# %%
TARGET_N_TOPICS = 10  # tune based on the hierarchy plot below

docs = conflict["Full_Text_Description"].fillna("").astype(str).tolist()
hierarchical_topics = topic_model.hierarchical_topics(docs)
hierarchical_topics.to_csv(OUT_DIR / "topic_hierarchy.csv", index=False)
print(f"  Saved topic merge hierarchy -> {OUT_DIR / 'topic_hierarchy.csv'} "
      f"-- read this top-to-bottom: each row shows two topics/clusters "
      f"merging at a given distance (smaller distance = more similar). "
      f"This tells you WHICH topics BERTopic itself considers closest, "
      f"rather than guessing from keyword lists.")

topic_model.reduce_topics(docs, nr_topics=TARGET_N_TOPICS)
result["topic_id_coarse"] = topic_model.topics_
coarse_info = topic_model.get_topic_info()
id_to_coarse_name = dict(zip(coarse_info["Topic"], coarse_info["Name"]))
result["topic_label_coarse"] = result["topic_id_coarse"].map(id_to_coarse_name)

result.to_csv(OUT_DIR / "conflict_topics.csv", index=False)
coarse_info.to_csv(OUT_DIR / "topic_info_coarse.csv", index=False)
print(f"\n  Reduced to {len(coarse_info) - 1} coarse topics:")
print(coarse_info[["Topic", "Count", "Name"]].to_string())
print(f"\n  Re-saved {OUT_DIR / 'conflict_topics.csv'} with topic_id_coarse / "
      f"topic_label_coarse added alongside the original fine-grained columns.")

# %% [markdown]
# ## 3. Visualizations

# %%
log("Generating topic visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

topic_counts = topic_info[topic_info["Topic"] != -1].sort_values("Count", ascending=True)
axes[0].barh(topic_counts["Name"], topic_counts["Count"], color="#2a9d8f")
axes[0].set_title("Records per topic")
axes[0].set_xlabel("Count")

county_topic = pd.crosstab(result["County"], result["topic_label"])
county_topic.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
axes[1].set_title("Topic distribution by county")
axes[1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
save_and_display(fig, OUT_DIR / "topic_distribution.png")

# %% [markdown]
# ## Summary

# %%
timer.__exit__()
print("=" * 60)
print("PHASE 2 STEP 4 (TOPIC MODELLING) COMPLETE")
print(f"Total runtime: {timer.elapsed:.1f} seconds")
print("=" * 60)
print(f"""
Files written to {OUT_DIR}/:
  1. conflict_topics.csv  - original data + topic_id, topic_probability, topic_label
  2. topic_info.csv       - one row per topic: size, name, representative words
  3. topic_distribution.png - records-per-topic + topic-by-county charts

REVIEW: read through topic_info.csv's representative words per topic and
give each a short human-readable name (e.g. "riparian encroachment",
"pastoral/cross-border conflict", "irrigation/water abstraction disputes")
for use in Phase 4/10 write-ups -- BERTopic's auto-generated names
(the underscored word lists) are functional but not thesis-quality
prose labels.
""")
