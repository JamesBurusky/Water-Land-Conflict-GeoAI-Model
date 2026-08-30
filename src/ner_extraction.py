"""
ner_extraction.py

Phase 2 step 3: Named Entity Recognition, extracting Places,
Communities, Institutions, Rivers, and Water Infrastructure per the
project spec.

Why plain spaCy isn't enough on its own
-----------------------------------------
Tested directly against the real sample data, spaCy's general-purpose
en_core_web_sm model mislabels "Turkana" as PERSON (it's a county/lake/
community name) and "Ethiopian Dassanech" (a community name) as
PERSON too -- unsurprising, since the model was trained on general
English news text, not Kenyan geography or ethnonyms.

Fix: a gazetteer-based `EntityRuler`, added to the pipeline BEFORE the
statistical NER component with `overwrite_ents=True`, so known terms
get a deterministic, correct label regardless of what the statistical
model would have guessed. This is a starter gazetteer built from what
appears in your samples plus common Kenyan geography for the 4 target
counties -- EXTEND THE LISTS BELOW as you review NER output on the
full dataset and spot terms it's still getting wrong or missing.

The final category mapping deliberately keeps spaCy's original label
alongside the project's 5 target categories (Place / Community /
Institution / River / Water_Infrastructure), so you can always trace
why something was categorized the way it was.
"""

from __future__ import annotations

import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

# --- Starter gazetteer -- EXTEND as you review real NER output ---

RIVERS = [
    "Nairobi River", "Ngong River", "Mathare River", "Athi River",
    "Tana River", "Turkwel River", "Kerio River", "Nzoia River",
    "Omo River", "Ruiru River", "Chania River", "Thiba River",
    "Gatharaini River", "Riara River",
]

WATER_INFRASTRUCTURE = [
    "borehole", "boreholes", "dam", "reservoir", "water pan",
    "irrigation scheme", "water pipeline", "abstraction point",
    "water tower", "canal", "sewerage", "treatment plant",
]

COMMUNITIES = [
    "Dassanech", "Turkana community", "Maasai", "Kamba", "Kikuyu",
    "Pokot", "Samburu", "Merille", "pastoralist", "pastoralists",
    "agropastoralist", "smallholder farmers",
]

INSTITUTIONS = [
    "Water Resources Authority", "WRA", "WRUA", "NEMA",
    "National Environment Management Authority",
    "Water Resource Management Authority", "WRMA",
    "Kenya Alliance of Resident Associations", "KARA",
    "National Land Commission", "NLC", "Law Society of Kenya",
    "Githunguri Water and Sanitation Company", "GIWASCO",
]

# Target-county-derived places -- populated at runtime from the
# canonical sub-county lookup (see build_gazetteer_from_lookup) rather
# than hardcoded here, since that list already exists and is
# authoritative (KNBS-sourced).


TARGET_COUNTY_NAMES = ["Nairobi", "Kiambu", "Machakos", "Turkana"]


def build_gazetteer_patterns(canonical_subcounties: list[str] | None = None) -> list[dict]:
    """
    Builds the EntityRuler pattern list. canonical_subcounties, if
    given (pass lookup.table['SubCounty'].tolist() from
    name_cleaning.py), adds every known sub-county name as a Place --
    reusing the Phase 1 canonical list rather than duplicating it.

    The 4 target county names themselves are always included (not just
    their sub-counties) -- confirmed necessary by testing: bare
    "Turkana" was mislabeled PERSON by the statistical model on real
    sample text, and county names appear standalone in the conflict
    narratives often enough that this needs to be unconditional.
    """
    patterns = []
    for term in TARGET_COUNTY_NAMES:
        patterns.append({"label": "PLACE", "pattern": term})
    for term in RIVERS:
        patterns.append({"label": "RIVER", "pattern": term})
    for term in WATER_INFRASTRUCTURE:
        patterns.append({"label": "WATER_INFRASTRUCTURE", "pattern": term})
    for term in COMMUNITIES:
        patterns.append({"label": "COMMUNITY", "pattern": term})
    for term in INSTITUTIONS:
        patterns.append({"label": "INSTITUTION", "pattern": term})
    if canonical_subcounties:
        for term in canonical_subcounties:
            patterns.append({"label": "PLACE", "pattern": term})
    return patterns


def load_ner_model(canonical_subcounties: list[str] | None = None,
                    model: str = "en_core_web_sm"):
    """
    Loads spaCy with the gazetteer EntityRuler inserted before the
    statistical `ner` component, with overwrite_ents=True so gazetteer
    matches take priority over the statistical model's label for the
    same span.
    """
    nlp = spacy.load(model)
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    ruler.add_patterns(build_gazetteer_patterns(canonical_subcounties))
    return nlp


# Maps spaCy's default statistical labels to the project's target
# categories. Gazetteer-matched entities already arrive with the right
# label (RIVER, PLACE, etc.) directly from the ruler above.
SPACY_LABEL_TO_CATEGORY = {
    "GPE": "Place",
    "LOC": "Place",
    "FAC": "Place",
    "ORG": "Institution",
    "NORP": "Community",   # nationalities/religious/political groups -- closest fit to "Community" for ethnonyms the gazetteer doesn't catch yet
    "PERSON": "Person",    # kept, but NOT one of the 5 target categories -- names of officials quoted in reports, not conflict-relevant entities themselves
}


def extract_entities(series: pd.Series, nlp) -> pd.DataFrame:
    """
    Returns one row per input record with entities grouped into the
    project's target categories as lists, plus the raw spaCy label for
    anything that fell outside the 5 target categories (mostly PERSON
    and DATE/CARDINAL/etc., kept in `other_entities` for visibility
    rather than silently discarded).
    """
    texts = series.fillna("").astype(str).tolist()
    docs = nlp.pipe(texts, batch_size=32)

    rows = []
    for doc in docs:
        by_category = {"Place": [], "River": [], "Water_Infrastructure": [],
                        "Community": [], "Institution": []}
        other = []
        for ent in doc.ents:
            if ent.label_ == "RIVER":
                by_category["River"].append(ent.text)
            elif ent.label_ == "WATER_INFRASTRUCTURE":
                by_category["Water_Infrastructure"].append(ent.text)
            elif ent.label_ == "COMMUNITY":
                by_category["Community"].append(ent.text)
            elif ent.label_ == "INSTITUTION":
                by_category["Institution"].append(ent.text)
            elif ent.label_ == "PLACE":
                by_category["Place"].append(ent.text)
            elif ent.label_ in SPACY_LABEL_TO_CATEGORY:
                mapped = SPACY_LABEL_TO_CATEGORY[ent.label_]
                if mapped in by_category:
                    by_category[mapped].append(ent.text)
                else:
                    other.append((ent.text, ent.label_))
            else:
                other.append((ent.text, ent.label_))

        row = {f"entities_{k}": list(dict.fromkeys(v)) for k, v in by_category.items()}
        row["other_entities"] = other
        rows.append(row)

    return pd.DataFrame(rows)
