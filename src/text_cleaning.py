"""
text_cleaning.py

Phase 2 step 1: text cleaning for the NLP pipeline.

Deliberately produces TWO parallel outputs per text field, not one:
  - a cleaned/lemmatized "bag of words" version, for anything that
    wants normalized tokens (topic modelling input, keyword frequency)
  - the ORIGINAL raw text, always kept alongside it

This matters because lowercasing/stripping/lemmatizing text is
DESTRUCTIVE for two of the steps still to come:
  - Named Entity Recognition needs capitalization ("Kangemi" vs
    "kangemi") -- spaCy's NER accuracy drops noticeably on lowercased
    text, since capitalization is one of its strongest signals for
    proper nouns.
  - Sentiment models (especially transformer-based ones) are trained
    on natural, un-lemmatized text and perform worse on a bag of
    stemmed/lemmatized tokens.

So: clean text is used for topic modelling / keyword frequency only;
NER and sentiment run on the raw Full_Text_Description /
Incident_Summary fields directly.
"""

from __future__ import annotations

import re

import pandas as pd
import spacy

# Domain-specific stopwords worth adding beyond spaCy's generic English
# list -- words that appear in almost every conflict record regardless
# of topic (e.g. "county", "kenya") and would otherwise dominate topic
# modelling output without actually distinguishing one conflict theme
# from another.
DOMAIN_STOPWORDS = {
    "county", "kenya", "kenyan", "area", "region", "report", "reported",
    "according", "said", "stated",
}


def load_nlp_model(model: str = "en_core_web_sm"):
    """
    Loads spaCy once, disabling pipeline components not needed for
    cleaning (parser, ner) for speed -- NER runs separately later with
    the full pipeline, since disabling it here is purely a speed
    optimization for the cleaning step, not a statement that NER isn't
    needed at all.
    """
    return spacy.load(model, disable=["parser", "ner"])


def clean_text_series(series: pd.Series, nlp) -> pd.DataFrame:
    """
    For each text value, returns:
      - clean_tokens: lowercased, lemmatized, stopwords/punctuation/
        numbers removed -- a list of tokens, ready for topic modelling
      - clean_text: the same tokens joined back into a string (some
        tools, like scikit-learn's vectorizers, want a string not a list)
      - n_tokens_removed: how many tokens were dropped as noise --
        surfaced so unusually high removal counts can be spot-checked
        (e.g. a record that's mostly boilerplate/numbers)
    """
    texts = series.fillna("").astype(str)
    # Basic noise removal before tokenization: URLs, and stray control
    # characters that occasionally survive encoding fixes.
    texts = texts.str.replace(r"https?://\S+", " ", regex=True)
    texts = texts.str.replace(r"[\r\n\t]+", " ", regex=True)

    docs = nlp.pipe(texts.tolist(), batch_size=64)

    clean_tokens_list = []
    n_removed_list = []
    for doc in docs:
        n_total = len(doc)
        tokens = [
            tok.lemma_.lower() for tok in doc
            if not tok.is_stop
            and not tok.is_punct
            and not tok.is_space
            and not tok.like_num
            and len(tok.lemma_) > 2
            and tok.lemma_.lower() not in DOMAIN_STOPWORDS
        ]
        clean_tokens_list.append(tokens)
        n_removed_list.append(n_total - len(tokens))

    return pd.DataFrame({
        "clean_tokens": clean_tokens_list,
        "clean_text": [" ".join(t) for t in clean_tokens_list],
        "n_tokens_removed": n_removed_list,
    })


def split_delimited_field(series: pd.Series, delimiter: str = ",") -> pd.Series:
    """
    Splits a delimited string field (e.g. NLP_Keywords: "riparian
    encroachment, river bank, ...") into a list of trimmed values.
    Used for NLP_Keywords and Parties_Involved, which arrive as
    human-written delimited strings rather than proper list columns.
    """
    return series.fillna("").apply(
        lambda s: [v.strip() for v in re.split(delimiter, s) if v.strip()]
    )
