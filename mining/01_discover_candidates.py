"""
01_discover_candidates.py

Stage 1 of the data collection pipeline. Runs every enabled connector
(RSS feeds, the manual URL list, and the search API if configured),
filters results to those whose title or snippet contains at least one
conflict-relevant keyword combined with at least one target-county
mention (so a generic "land dispute" story from outside the four study
counties doesn't get carried into review), and writes the result to
01_candidates.csv for stage 2 to process.

This stage does NOT fetch full article pages -- it works only from
feed/search metadata (title, snippet, URL), so it is fast and places
minimal load on source servers. Full-page fetching, respecting
robots.txt and rate limits, happens in stage 2.

Usage:
    python 01_discover_candidates.py
"""

import csv
from datetime import datetime

import config
import connectors


def matches_scope(title: str, snippet: str) -> tuple[bool, str, str]:
    """Returns (matched, matched_keyword, matched_county). A candidate
    must mention at least one conflict keyword AND at least one target
    county to pass -- this is a coarse relevance filter, not a
    guarantee of relevance; stage 2's human review is what actually
    confirms a record belongs in the dataset.

    Keyword matching checks whether every significant word of a
    keyword phrase appears somewhere in the text, rather than requiring
    the exact phrase as a contiguous substring -- real headlines
    frequently insert words between them (e.g. "riparian LAND
    encroachment" for the keyword "riparian encroachment"), and a
    missed genuine candidate here is a permanent loss, while a false
    positive is simply filtered out during stage 2/3 review."""
    text = f"{title} {snippet}".lower()
    text_words = set(text.split())

    def keyword_present(keyword: str) -> bool:
        sig_words = [w for w in keyword.split() if len(w) > 3]  # skip short connector words
        return all(any(sw in tw for tw in text_words) for sw in sig_words)

    matched_keyword = next((kw for kw in config.CONFLICT_KEYWORDS if keyword_present(kw)), None)
    matched_county = next((c for c in config.TARGET_COUNTIES if c.lower() in text), None)
    return bool(matched_keyword and matched_county), matched_keyword or "", matched_county or ""


def main():
    print("Stage 1: Discovering candidates\n")
    all_items = []

    print("[RSS feeds]")
    all_items.extend(connectors.rss_discover())

    print("\n[Manual URL list]")
    all_items.extend(connectors.manual_url_discover())

    if config.SEARCH_API_KEY:
        print("\n[Search API]")
        for county in config.TARGET_COUNTIES:
            for keyword in config.CONFLICT_KEYWORDS:
                query = f"{keyword} {county} Kenya"
                print(f"  Query: {query!r}")
                all_items.extend(connectors.search_api_discover(query))
    else:
        print("\n[Search API] Disabled (config.SEARCH_API_KEY is empty) -- "
              "relying on RSS + manual URL list only.")

    print(f"\nTotal items collected before filtering: {len(all_items)}")

    # Manually-provided URLs bypass the keyword/county filter entirely --
    # if you pasted a URL into manual_urls.txt, you already decided it
    # was relevant, so it goes straight to stage 2.
    candidates = []
    seen_urls = set()
    for item in all_items:
        if not item["url"] or item["url"] in seen_urls:
            continue
        seen_urls.add(item["url"])

        if item["discovered_via"] == "manual":
            candidates.append({**item, "matched_keyword": "", "matched_county": "", "manual_override": True})
            continue

        matched, keyword, county = matches_scope(item["title"], item["snippet"])
        if matched:
            candidates.append({**item, "matched_keyword": keyword, "matched_county": county, "manual_override": False})

    print(f"Candidates after keyword+county filtering (or manual override): {len(candidates)}")

    with open(config.CANDIDATES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "url", "title", "snippet", "source_name", "discovered_via",
            "matched_keyword", "matched_county", "manual_override", "discovery_date",
        ])
        writer.writeheader()
        for c in candidates:
            writer.writerow({**c, "discovery_date": datetime.now().strftime("%Y-%m-%d")})

    print(f"\nSaved {len(candidates)} candidates to {config.CANDIDATES_PATH}")
    print("Next: review this list if you want (drop obviously irrelevant rows), "
          "then run 02_extract_and_suggest.py.")


if __name__ == "__main__":
    main()
