"""
sentiment.py

Phase 2 step 2: sentiment analysis on conflict text.

Two backends, chosen explicitly rather than one hardcoded choice:

  - "vader": a lexicon/rule-based sentiment scorer (VADER). Fast, fully
    offline, no model download -- this is what's used and validated in
    this sandbox (which cannot reach huggingface.co to download
    transformer weights). Reasonably accurate for clearly emotive
    language but weaker on the dry, bureaucratic tone common in this
    dataset ("County Government issued a notice...").

  - "transformer": a proper pretrained sentiment model (e.g.
    distilbert-base-uncased-finetuned-sst-2-english) via
    HuggingFace `transformers`. More accurate on nuanced/bureaucratic
    text, but needs internet access to download the model weights on
    first run. NOT validated inside this sandbox (no huggingface.co
    access here) -- run the self-test at the bottom of this file on
    your local machine to confirm it works before trusting it on the
    full dataset.

Recommendation: run BOTH once on your local machine and compare on a
sample of ~50 records by eye. If they mostly agree, VADER is fine
(much faster on 1,200+ records) and worth using for the full run,
with the transformer backend as a documented alternative in your
methodology if a reviewer asks why a "real" NLP project used a
lexicon method for sentiment.
"""

from __future__ import annotations

import pandas as pd


def compute_sentiment_vader(series: pd.Series) -> pd.DataFrame:
    """
    Returns sentiment_score (compound, -1 to +1) and
    sentiment_label (negative / neutral / positive) using VADER's
    standard compound-score thresholds (>=0.05 positive, <=-0.05
    negative, else neutral -- these are VADER's own published
    defaults, not something we invented).
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    texts = series.fillna("").astype(str)
    scores = texts.apply(lambda t: analyzer.polarity_scores(t)["compound"])

    def label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        return "neutral"

    return pd.DataFrame({
        "sentiment_score": scores,
        "sentiment_label": scores.apply(label),
    })


def compute_sentiment_transformer(
    series: pd.Series,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Transformer-based sentiment via HuggingFace `transformers`.
    Requires: pip install transformers torch
    Requires internet access to download model weights on first call
    (NOT available in the sandbox this was built in -- run this
    function's self-test locally before relying on it).

    sentiment_score here is the model's confidence in its predicted
    label (0 to 1), signed negative for the NEGATIVE class so it's on
    a comparable -1..+1-ish scale to the VADER compound score (though
    the two are not numerically equivalent -- don't average them
    together, pick one backend for the actual analysis).
    """
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=model_name, batch_size=batch_size)

    texts = series.fillna("").astype(str).tolist()
    # Transformer models have a max token length; truncate defensively
    # since Full_Text_Description can run long.
    results = clf(texts, truncation=True, max_length=512)

    scores = [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]
    labels = [r["label"].lower() for r in results]
    return pd.DataFrame({"sentiment_score": scores, "sentiment_label": labels})


def compute_sentiment(series: pd.Series, backend: str = "vader", **kwargs) -> pd.DataFrame:
    """Dispatch to the chosen backend -- see module docstring for the
    tradeoffs between the two."""
    if backend == "vader":
        return compute_sentiment_vader(series)
    elif backend == "transformer":
        return compute_sentiment_transformer(series, **kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}' -- use 'vader' or 'transformer'")


if __name__ == "__main__":
    # Local self-test for the transformer backend -- run this file
    # directly on your machine (with internet access) to confirm the
    # transformer path works before using it on the full dataset:
    #   python src/sentiment.py
    import pandas as pd
    sample = pd.Series([
        "Violent clashes erupted between pastoralist communities over grazing rights, leaving several dead.",
        "The county water office issued a routine permit renewal notice.",
        "Residents welcomed the new borehole, praising the improved access to clean water.",
    ])
    print("VADER results:")
    print(compute_sentiment_vader(sample))
    print("\nTransformer results (requires internet + `pip install transformers torch`):")
    try:
        print(compute_sentiment_transformer(sample))
    except Exception as e:
        print(f"  Skipped: {e}")
