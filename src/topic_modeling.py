"""
topic_modeling.py

Phase 2 step 4: topic modelling via BERTopic, to identify dominant
conflict themes (riparian encroachment, drought, illegal abstraction,
land disputes, pastoral conflicts, etc.) per the project spec.

IMPORTANT design choice: BERTopic runs on the RAW (or lightly cleaned)
Full_Text_Description, NOT the heavily lemmatized clean_tokens from
text_cleaning.py. This is deliberate: the embedding step needs natural
sentence structure to understand context (a sentence-transformer model
was trained on real sentences, not bags of lemmas), while the
topic-word extraction step (c-TF-IDF) uses its own internal vectorizer
that strips stopwords separately. Feeding it pre-lemmatized text would
hurt embedding quality without meaningfully improving topic-word
output.

Embedding backend
------------------
Two options, matching the same pattern as src/sentiment.py:

  - "sentence_transformer" (default, RECOMMENDED for your actual run):
    downloads a real embedding model (all-MiniLM-L6-v2 by default) from
    HuggingFace on first use. Needs internet access. NOT reachable from
    the sandbox this was built in (confirmed by direct test: HuggingFace
    is not on the sandbox's allowed domain list) -- you must run this on
    your local machine, which does have internet access.

  - "tfidf_svd": a TF-IDF + dimensionality reduction substitute that
    needs no internet or model download. Used ONLY to validate this
    module's plumbing (does clustering/topic extraction work at all)
    in the sandbox. This is NOT a real semantic embedding and will
    produce noticeably worse topic quality -- do not use it for your
    actual thesis results, only as a fallback if you ever need to
    test changes to this code without internet access.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS, ENGLISH_STOP_WORDS


def build_place_name_stopwords(
    canonical_subcounties: list[str] | None = None,
    nlp_enriched_df: pd.DataFrame | None = None,
    entity_cols: tuple[str, ...] = ("entities_Place", "entities_River"),
) -> list[str]:
    """
    Place names dominating a topic's word list is a known BERTopic
    behavior, not a bug: the topic REPRESENTATION (the words shown per
    topic) comes from c-TF-IDF over a CountVectorizer, which by default
    only strips generic English stopwords -- "Nairobi", "Kiambu",
    "Turkana", ward names, etc. are frequent, statistically distinctive
    terms in a geospatial conflict dataset, so they naturally score
    high and show up in the topic's name/keywords.

    This builds a comprehensive place-name exclusion list from TWO
    complementary sources:
      1. A static gazetteer: the 4 target counties and rivers from
         ner_extraction.py's gazetteer, the canonical sub-county list
         from name_cleaning.py, non-target-county place names
         confirmed present in this dataset (Pokot, Baringo, Samburu,
         etc.), and generic geography words ("county", "kenya", ...).
      2. OPTIONAL, and stronger: pass nlp_enriched_df (the output of
         04_nlp_pipeline.py, with entities_Place/entities_River
         columns already computed) to ALSO exclude every specific
         place spaCy's NER actually found in your real text --
         critically, this catches WARD/NEIGHBORHOOD-level names the
         static gazetteer can't anticipate (Kangemi, Mathare, Mukuru,
         Lavington, Dagoretti, ...), since those sit below the
         sub-county level covered by canonical_subcounties. This is
         the "stronger, entity-masking fix" referenced elsewhere in
         this module -- pass nlp_enriched_df to get it; the function
         still works with just the static gazetteer if you don't.

    This ONLY affects the topic-word REPRESENTATION (what's shown as
    each topic's name/keywords) -- NOT the embeddings themselves, which
    still see the full raw text for context (see the module docstring
    for why that separation matters). If place names still dominate
    topics after this, that's a sign the embedding step itself is
    clustering partly by region rather than theme, which this
    word-list fix can't address (a genuinely different problem).
    """
    from ner_extraction import RIVERS, TARGET_COUNTY_NAMES

    other_kenyan_places = [
        "pokot", "west pokot", "baringo", "samburu", "marakwet",
        "elgeyo marakwet", "laikipia", "marsabit", "isiolo", "nakuru",
        "uganda", "ethiopia", "tiaty", "kerio", "kerio valley",
        "north rift", "kajiado", "kitui", "makueni",
    ]
    generic_geo_terms = [
        "county", "counties", "sub-county", "subcounty", "sub county",
        "kenya", "kenyan", "ward", "village", "area", "region", "district",
    ]

    river_terms = [r.lower() for r in RIVERS] + [r.replace(" River", "").lower() for r in RIVERS]
    county_terms = [c.lower() for c in TARGET_COUNTY_NAMES]
    subcounty_terms = [s.lower() for s in (canonical_subcounties or [])]

    words = set(river_terms + county_terms + subcounty_terms + other_kenyan_places + generic_geo_terms)

    if nlp_enriched_df is not None:
        # Reuses the exact same term lists as src/land_water_analysis.py
        # so "protected" and "thematically meaningful" mean the same
        # thing everywhere in this codebase, not two lists that could
        # silently drift apart.
        from land_water_analysis import LAND_KEYWORDS, WATER_KEYWORDS
        protected_words = set()
        for phrase in LAND_KEYWORDS + WATER_KEYWORDS:
            protected_words.update(phrase.lower().split())

        def _parse_list_cell(cell) -> list[str]:
            # After a CSV round-trip, a list column becomes a string
            # like "['Nairobi', 'Kangemi']" -- parse it back; if it's
            # already a real list (same-session, no round-trip), use
            # it as-is.
            if isinstance(cell, list):
                return cell
            if isinstance(cell, str) and cell.startswith("["):
                try:
                    return ast.literal_eval(cell)
                except (ValueError, SyntaxError):
                    return []
            return []

        # Multi-word names ("Athi River") are split into individual
        # words ("athi", "river") because CountVectorizer's stop_words
        # filtering happens on unigram tokens BEFORE n-grams are
        # built -- removing "athi" from the vocabulary also prevents
        # the "athi river" bigram from ever forming. BUT a component
        # word that's independently thematically meaningful ("river"
        # appearing inside "Nairobi River") is protected and
        # never added, even as part of a multi-word entity -- confirmed
        # necessary by testing: naive splitting was silently excluding
        # "river" itself, which is exactly the kind of word a water-
        # conflict topic model needs to keep (it's what distinguishes
        # a river conflict from a borehole or dam one).
        for col in entity_cols:
            if col not in nlp_enriched_df.columns:
                continue
            for cell in nlp_enriched_df[col].dropna():
                for place in _parse_list_cell(cell):
                    for token in str(place).lower().replace("-", " ").split():
                        if len(token) > 2 and token not in protected_words:
                            words.add(token)

    return sorted(words)


def get_embeddings(docs: list[str], backend: str = "sentence_transformer",
                    model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if backend == "sentence_transformer":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(docs, show_progress_bar=True)
    elif backend == "tfidf_svd":
        from sklearn.decomposition import TruncatedSVD
        tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
        X = tfidf.fit_transform(docs)
        n_components = min(50, X.shape[1] - 1, X.shape[0] - 1)
        n_components = max(n_components, 2)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        return svd.fit_transform(X)
    else:
        raise ValueError(f"Unknown backend '{backend}'")


def build_topic_model(min_topic_size: int = 15, nr_topics: int | str | None = None,
                       ngram_range: tuple = (1, 2),
                       extra_stopwords: list[str] | None = None) -> BERTopic:
    """
    Configures BERTopic with a domain-appropriate vectorizer for the
    topic-word (c-TF-IDF) representation -- English stopwords removed,
    1-2 word phrases allowed (so it can surface phrases like "riparian
    encroachment" as a unit, not just single words), plus any
    extra_stopwords (e.g. from build_place_name_stopwords()) to keep
    place names out of the topic word lists specifically.

    min_topic_size=15 is a reasonable starting point for a dataset in
    the hundreds-to-low-thousands range -- too low and you get dozens
    of noisy micro-topics, too high and distinct themes get merged.
    Treat it as tunable: if the full run produces an unhelpfully small
    or large number of topics, adjust this and re-run (it's fast to
    re-fit once embeddings are cached).

    nr_topics defaults to None (no auto-reduction at fit time) --
    deliberately, since 05_topic_modelling.py does its OWN explicit,
    inspectable two-stage reduction afterward (reduce_outliers, then
    reduce_topics to a chosen target count). Letting BERTopic
    auto-reduce during fit_transform as well would double up on that
    and make the final topic count harder to reason about.
    """
    stop_words = list(ENGLISH_STOP_WORDS) + (extra_stopwords or [])
    vectorizer_model = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)
    return BERTopic(
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )


def fit_topics_temporal_safe(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    text_col: str = "Full_Text_Description",
    date_col: str = "Date_Start_parsed",
    embedding_backend: str = "sentence_transformer",
    min_topic_size: int = 15,
    extra_stopwords: list[str] | None = None,
    reduce_outliers_after_fit: bool = True,
    target_n_topics: int | None = 10,
) -> tuple[pd.DataFrame, BERTopic, pd.DataFrame]:
    """
    Leakage-safe version of fit_topics: fits BERTopic ONLY on records
    with date_col < cutoff_date (the training period), then applies
    topic_model.transform() -- NOT fit_transform() -- to records at or
    after cutoff_date. transform() assigns each document to the
    CLOSEST already-discovered topic without changing the topic
    definitions themselves, so test-period text can never influence
    what a topic means or where its boundaries sit.

    This directly fixes the leakage flagged during Phase 2/6: the
    original fit_topics() in this module was run on the full corpus at
    once, meaning topic boundaries were partly shaped by documents from
    the test period before any train/test split even happened.
    cutoff_date should match the CUTOFF_YEAR used in
    10_ml_modeling.py's temporal_train_test_split, so the topic
    features and the model's own train/test split agree on what
    counts as "the future" -- using a different cutoff here would
    reintroduce a subtler version of the same leakage problem.

    Test-period documents that transform() cannot confidently assign
    to any existing topic get -1 (outlier), same convention as
    fit_transform -- expect a HIGHER outlier rate in the test period
    than the training period, since BERTopic can only recognize themes
    it already saw during fitting; a new theme that only emerges in
    the test period will correctly show up as -1, not be smuggled into
    an existing topic.

    reduce_outliers_after_fit / target_n_topics: mirrors the same two
    consolidation steps 05_topic_modelling.py applies (outlier
    reassignment, then hierarchical reduction to ~target_n_topics
    coherent themes) -- added here because THIS function had neither,
    which was a real, confirmed gap: without it, this path produces a
    raw, unconsolidated topic set (often 15-25 topics, several tiny)
    while 05's path produces a clean ~10-topic set, and a dashboard
    reading whichever ran most recently would look wildly inconsistent
    run to run for no reason related to data quality. CRITICALLY, both
    steps are fit using ONLY train_docs (the fitted model's own topic
    definitions), and test-period documents are then RE-TRANSFORMED
    against the updated model -- never re-fit -- so this consolidation
    adds no new leakage; it's the exact same train-only-shapes-topics
    guarantee as the initial fit above, just applied one more time.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    train_mask = df[date_col] < cutoff_date

    train_docs = df.loc[train_mask, text_col].fillna("").astype(str).tolist()
    test_docs = df.loc[~train_mask, text_col].fillna("").astype(str).tolist()
    print(f"  Fitting on {len(train_docs)} training-period document(s) "
          f"(before {cutoff_date.date()}); transforming {len(test_docs)} "
          f"test-period document(s) against the fitted model")

    train_embeddings = get_embeddings(train_docs, backend=embedding_backend)
    topic_model = build_topic_model(min_topic_size=min_topic_size, extra_stopwords=extra_stopwords)
    train_topics, _ = topic_model.fit_transform(train_docs, embeddings=train_embeddings)

    if reduce_outliers_after_fit:
        print("  Reassigning training-period outliers to their nearest topic (c-TF-IDF strategy)...")
        train_topics = topic_model.reduce_outliers(train_docs, train_topics, strategy="c-tf-idf")
        topic_model.update_topics(train_docs, topics=train_topics, vectorizer_model=topic_model.vectorizer_model)

    if target_n_topics is not None:
        current_n = len(topic_model.get_topic_info()) - 1  # exclude the -1 outlier row
        if current_n > target_n_topics:
            print(f"  Reducing {current_n} raw topics to ~{target_n_topics} coherent themes "
                  f"(hierarchical merge, train-period documents only)...")
            topic_model.reduce_topics(train_docs, nr_topics=target_n_topics)
            train_topics = topic_model.topics_
        else:
            print(f"  Only {current_n} topic(s) found -- already at or below target_n_topics "
                  f"({target_n_topics}), skipping reduction.")

    if len(test_docs) > 0:
        # Re-transform against the FINAL (post-outlier-reassignment,
        # post-reduction) model -- test documents were never used to
        # shape any of the above, only to be classified against it.
        test_embeddings = get_embeddings(test_docs, backend=embedding_backend)
        test_topics, _ = topic_model.transform(test_docs, embeddings=test_embeddings)
    else:
        test_topics = []

    out = df.copy()
    out["topic_id"] = -99  # sentinel, overwritten below -- makes any row
                            # that somehow missed both branches obvious
    out.loc[train_mask, "topic_id"] = train_topics
    out.loc[~train_mask, "topic_id"] = test_topics

    topic_info = topic_model.get_topic_info()
    id_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))
    out["topic_label"] = out["topic_id"].map(id_to_name)

    # Defensive final check: topic_info should never list a topic with
    # zero documents actually assigned in `out` -- if it somehow does
    # (e.g. a topic only test-period docs would have matched before
    # reduction, now merged away), drop it here so nothing downstream
    # (the dashboard) can ever display a theme that filters to nothing.
    live_counts = out["topic_id"].value_counts()
    topic_info = topic_info[topic_info["Topic"].isin(live_counts.index)].reset_index(drop=True)

    train_outlier_rate = (out.loc[train_mask, "topic_id"] == -1).mean()
    test_outlier_rate = (out.loc[~train_mask, "topic_id"] == -1).mean() if len(test_docs) > 0 else float("nan")
    print(f"  Outlier rate -- train: {train_outlier_rate:.1%}, test: {test_outlier_rate:.1%} "
          f"(test rate being noticeably higher than train is normal and expected here)")

    return out, topic_model, topic_info


def fit_topics(df: pd.DataFrame, text_col: str = "Full_Text_Description",
                embedding_backend: str = "sentence_transformer",
                min_topic_size: int = 15,
                extra_stopwords: list[str] | None = None) -> tuple[pd.DataFrame, BERTopic, pd.DataFrame]:
    """
    Fits BERTopic on df[text_col] and returns:
      - df with new columns: topic_id, topic_probability, topic_label
      - the fitted BERTopic model (for later use: topic_model.get_topic_info(),
        visualizations, transforming new documents, etc.)
      - the topic info table (topic_id, count, name, representative words)

    Records get topic_id == -1 for BERTopic's "outlier" cluster --
    text too dissimilar from any dominant theme to group confidently.
    This is normal and expected, NOT an error; report the outlier
    rate in your methodology (a very high outlier rate would suggest
    min_topic_size is too high for the data).
    """
    docs = df[text_col].fillna("").astype(str).tolist()
    embeddings = get_embeddings(docs, backend=embedding_backend)

    topic_model = build_topic_model(min_topic_size=min_topic_size, extra_stopwords=extra_stopwords)
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    out = df.copy()
    out["topic_id"] = topics
    # calculate_probabilities=True gives a probability matrix; take
    # each doc's probability of belonging to ITS assigned topic
    if probs is not None and hasattr(probs, "ndim") and probs.ndim == 2:
        out["topic_probability"] = [
            probs[i, t] if t != -1 and t < probs.shape[1] else np.nan
            for i, t in enumerate(topics)
        ]
    else:
        out["topic_probability"] = probs

    topic_info = topic_model.get_topic_info()
    id_to_name = dict(zip(topic_info["Topic"], topic_info["Name"]))
    out["topic_label"] = out["topic_id"].map(id_to_name)

    return out, topic_model, topic_info
