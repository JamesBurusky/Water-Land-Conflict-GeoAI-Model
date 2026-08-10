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

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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
                       ngram_range: tuple = (1, 2)) -> BERTopic:
    """
    Configures BERTopic with a domain-appropriate vectorizer for the
    topic-word (c-TF-IDF) representation -- English stopwords removed,
    1-2 word phrases allowed (so it can surface phrases like "riparian
    encroachment" as a unit, not just single words).

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
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=ngram_range)
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
    topic_model = build_topic_model(min_topic_size=min_topic_size)
    train_topics, _ = topic_model.fit_transform(train_docs, embeddings=train_embeddings)

    if len(test_docs) > 0:
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

    train_outlier_rate = (out.loc[train_mask, "topic_id"] == -1).mean()
    test_outlier_rate = (out.loc[~train_mask, "topic_id"] == -1).mean() if len(test_docs) > 0 else float("nan")
    print(f"  Outlier rate -- train: {train_outlier_rate:.1%}, test: {test_outlier_rate:.1%} "
          f"(test rate being noticeably higher than train is normal and expected here)")

    return out, topic_model, topic_info


def fit_topics(df: pd.DataFrame, text_col: str = "Full_Text_Description",
                embedding_backend: str = "sentence_transformer",
                min_topic_size: int = 15) -> tuple[pd.DataFrame, BERTopic, pd.DataFrame]:
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

    topic_model = build_topic_model(min_topic_size=min_topic_size)
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
