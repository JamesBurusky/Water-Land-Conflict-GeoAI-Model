"""
02_extract_and_suggest.py

Stage 2 of the data collection pipeline. For every candidate URL from
stage 1: fetches the page (respecting robots.txt and rate limits),
extracts the main article text, and runs lightweight NLP to SUGGEST
values for the target schema's fields -- dates, locations, party
names, and casualty/displacement mentions.

IMPORTANT -- read this before trusting anything this stage produces:
Every suggested field is a starting point for a human reviewer, not a
verified fact. This deliberately mirrors how the existing compiled
dataset was actually built (Chapter Three): automated retrieval,
followed by manual review of every record. In particular:
  - Casualty and displacement numbers are notoriously easy to get
    wrong automatically (a number near the word "died" might refer to
    something else entirely in the same paragraph); the suggestion
    includes the surrounding sentence specifically so you can check
    it against the source before accepting it.
  - Conflict_Type/Subtype, Legal_Status, and Outcome_Status are left
    BLANK for you to fill in -- these require reading the article, not
    guessing from keywords.
  - Verified is always written as False. Do not change this to True
    until you have actually read the source and confirmed every field.

Usage:
    python 02_extract_and_suggest.py
"""

import csv
import re
from datetime import datetime

import spacy
import trafilatura

import config
import utils

print("Loading spaCy model (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm")

CASUALTY_PATTERN = re.compile(
    r"([^.]*\b(\d+|dozens?|scores?|several|many)\b[^.]*\b(died|killed|dead|casualt\w+|injured|wounded)\b[^.]*\.)",
    re.IGNORECASE,
)
DISPLACEMENT_PATTERN = re.compile(
    r"([^.]*\b(\d+|dozens?|scores?|several|many|hundreds?|thousands?)\b[^.]*\b(displaced|fled|evacuat\w+|homeless)\b[^.]*\.)",
    re.IGNORECASE,
)
LOWER_BOUND_PATTERN = re.compile(r"\bat least\b|\bmore than\b|\bover\b", re.IGNORECASE)


def extract_article_text(url: str) -> str | None:
    resp = utils.polite_get(url)
    if resp is None:
        return None
    text = trafilatura.extract(resp.text, url=url)
    return text


def suggest_fields(text: str, candidate_county_hint: str) -> dict:
    doc = nlp(text[:100_000])  # cap to keep NER runtime reasonable on very long pages

    def clean(s: str) -> str:
        return " ".join(s.split())  # collapses embedded newlines/multiple spaces from source line breaks

    dates = sorted(set(clean(ent.text) for ent in doc.ents if ent.label_ == "DATE"))
    locations = sorted(set(clean(ent.text) for ent in doc.ents if ent.label_ in ("GPE", "LOC")))
    people_orgs = sorted(set(clean(ent.text) for ent in doc.ents if ent.label_ in ("PERSON", "ORG")))[:15]

    casualty_matches = CASUALTY_PATTERN.findall(text)
    displacement_matches = DISPLACEMENT_PATTERN.findall(text)
    casualty_context = clean(casualty_matches[0][0]) if casualty_matches else ""
    displacement_context = clean(displacement_matches[0][0]) if displacement_matches else ""

    keywords = sorted(set(list(locations)[:5] + list(people_orgs)[:5]))

    return {
        "suggested_dates": "; ".join(dates[:5]),
        "suggested_locations": "; ".join(locations[:8]),
        "suggested_parties": "; ".join(people_orgs),
        "casualty_context_SENTENCE_TO_VERIFY": casualty_context,
        "casualty_is_lower_bound_phrasing": bool(LOWER_BOUND_PATTERN.search(casualty_context)),
        "displacement_context_SENTENCE_TO_VERIFY": displacement_context,
        "displacement_is_lower_bound_phrasing": bool(LOWER_BOUND_PATTERN.search(displacement_context)),
        "nlp_keywords_suggested": ", ".join(keywords),
    }


def build_schema_row(candidate: dict, article_text: str, suggestions: dict, sequence: int) -> dict:
    row = utils.empty_schema_row()
    county = candidate.get("matched_county") or ""
    row["Record_ID"] = utils.generate_record_id(county or "Unknown", sequence)
    row["County"] = county
    row["Sub_Location"] = suggestions["suggested_locations"]  # SUGGESTION -- verify against county boundaries
    row["Date_Start"] = suggestions["suggested_dates"]         # SUGGESTION -- confirm which date is the incident date
    row["Incident_Summary"] = candidate.get("title", "")
    row["Parties_Involved"] = suggestions["suggested_parties"]
    row["Full_Text_Description"] = (article_text or "")[:5000]  # truncated; read the Source_URL for the full text
    row["Source_Name"] = candidate.get("source_name", "") or utils.get_domain(candidate["url"])
    row["Source_URL"] = candidate["url"]
    row["Data_Source_Type"] = utils.classify_source_type(candidate["url"])
    row["Casualties_Reported"] = suggestions["casualty_context_SENTENCE_TO_VERIFY"]
    row["Displaced_Persons"] = suggestions["displacement_context_SENTENCE_TO_VERIFY"]
    row["NLP_Keywords"] = suggestions["nlp_keywords_suggested"]
    row["Verified"] = False
    # A rough completeness proxy, NOT a factual-accuracy score -- it
    # only reflects how many fields NER/regex managed to suggest
    # something for, out of the fields that can be reasonably
    # auto-suggested at all (Conflict_Type, Legal_Status, and
    # Outcome_Status are excluded from this count since they are
    # deliberately left for manual entry, not auto-suggested).
    suggestable = ["Sub_Location", "Date_Start", "Parties_Involved", "Casualties_Reported", "Displaced_Persons"]
    filled = sum(1 for f in suggestable if row[f])
    row["Confidence_Score"] = round(filled / len(suggestable), 2)
    row["Notes"] = ("AUTO-SUGGESTED, NOT VERIFIED. Read Source_URL and confirm every field, "
                    "especially Sub_Location, Date_Start, Casualties_Reported, and "
                    "Displaced_Persons, before setting Verified=True. Conflict_Type, "
                    "Conflict_Subtype, Legal_Status, and Outcome_Status were left blank "
                    "deliberately -- fill these in after reading the source.")
    return row


def main():
    print("Stage 2: Extracting and suggesting fields\n")
    if not config.CANDIDATES_PATH.exists():
        print(f"ERROR: {config.CANDIDATES_PATH} not found -- run 01_discover_candidates.py first.")
        return

    candidates = list(csv.DictReader(open(config.CANDIDATES_PATH, encoding="utf-8")))
    print(f"Processing {len(candidates)} candidates (rate-limited per source domain, "
          f"~{config.REQUEST_DELAY_SECONDS}s between requests to the same domain)...\n")

    rows = []
    for i, candidate in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {candidate['url']}")
        text = extract_article_text(candidate["url"])
        if not text:
            print("    SKIPPED (could not extract article text)")
            continue
        suggestions = suggest_fields(text, candidate.get("matched_county", ""))
        rows.append(build_schema_row(candidate, text, suggestions, sequence=i))
        print(f"    OK -- completeness proxy {rows[-1]['Confidence_Score']}")

    with open(config.REVIEW_QUEUE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.SCHEMA)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {config.REVIEW_QUEUE_PATH}")
    print("Next: run 03_dedupe_check.py, then open the review queue and manually confirm "
          "every field before merging any row into your master dataset.")


if __name__ == "__main__":
    main()
