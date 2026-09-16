# Water-Land Conflict Data Collection Pipeline

A semi-automated tool for extending your compiled conflict dataset with new
records, following the same discover-then-manually-verify process actually
used to build the existing dataset (Chapter Three) -- not a fully autonomous
"fact extraction" tool, and deliberately so. See "What this pipeline does
and does not do" below before running it.

## Setup

```bash
pip install requests beautifulsoup4 trafilatura feedparser spacy pandas --break-system-packages
python -m spacy download en_core_web_sm
```

Edit `config.py`:
- Set your real email in `USER_AGENT` (identifies your requests honestly to
  the sites you fetch from -- don't leave the placeholder in).
- If you have a search API key (Bing Web Search, SerpAPI, etc.), set
  `SEARCH_API_PROVIDER` and `SEARCH_API_KEY` to enable broader automated
  discovery. Leave blank to rely on RSS + your own manual URL list only.
- If you want new candidates checked against your existing compiled
  dataset (recommended), set `EXISTING_DATASET_PATH` to its file path.

## Running it

```bash
python 01_discover_candidates.py   # find candidate URLs
python 02_extract_and_suggest.py   # fetch pages, suggest field values
python 03_dedupe_check.py          # flag likely repeats of the same incident
```

Then: **open `collection_outputs/03_ready_for_review.csv` yourself and read
every row against its `Source_URL` before touching anything else.**

## What this pipeline does and does not do

**Automates safely:**
- Discovering candidate articles (RSS feeds, your own manual URL list, an
  optional search API)
- Fetching pages politely: respects `robots.txt`, rate-limits per domain,
  identifies itself honestly via `User-Agent`
- Extracting clean article text from the raw page
- *Suggesting* dates, locations, and party names via NLP (spaCy NER)
- *Suggesting* casualty/displacement mentions, with the surrounding
  sentence included so you can check it against the source
- Flagging likely duplicate coverage of the same real incident

**Does NOT automate, on purpose:**
- `Conflict_Type`, `Conflict_Subtype`, `Legal_Status`, `Outcome_Status` are
  left blank. These need someone to actually read the article.
- `Verified` is always written `False`. Nothing in this pipeline sets it to
  `True` -- that's your call, after you've checked the record.
- `Confidence_Score` measures how many fields NER managed to *suggest
  something for*, not whether those suggestions are factually correct.
  Read it as "how complete is this draft," not "how trustworthy."

This mirrors your own documented methodology (Chapter Three): the real
dataset was built through automated retrieval combined with manual
case-by-case review, not full automation -- Twitter/X automated collection
was in fact tried and abandoned specifically because it couldn't meet this
project's accuracy needs. This pipeline is designed the same way for the
same reason: automatically-extracted casualty counts, dates, and legal
outcomes from free-text news are exactly the kind of unverified "facts"
that would undermine the data-quality standard the rest of this project has
been built on.

## What's been tested, and what hasn't

I tested the actual logic of every stage -- keyword/county filtering, NLP
field suggestion, and duplicate detection -- against realistic synthetic
text, and fixed two real bugs that testing caught:
1. Keyword matching originally required an exact phrase and missed real
   headlines that split it across words (e.g. "riparian **land**
   encroachment"); it now checks that all significant words appear
   anywhere in the text.
2. Duplicate detection originally used character-level text similarity,
   which scored two outlets' paraphrased coverage of the *same real event*
   at only 55% -- below its own threshold. It now uses word-overlap
   (Jaccard) similarity, which separates genuine duplicates (~45%) from
   merely-similar-topic stories (~12%) far more reliably.

**What I could not test**: live fetches against real Kenyan news sites and
Kenya Law from this environment -- my sandbox's network access is
restricted to a fixed allowlist that excludes them. I did independently
verify that `nation.africa/kenya/rss.xml` is a real, live feed (fetched and
inspected its actual content before adding it to `connectors.py`), and
confirmed `feedparser` parses that real content correctly. But I could not
run stage 2's live page-fetching end-to-end against a real target site.
Run a small test batch first (a handful of candidates) and check the output
before doing a full run.

Kenya Law's current search interface appears to be JavaScript-driven, and I
could not reliably determine a stable query URL for it without risking a
scraper built on guessed selectors that could silently break or silently
return nothing. Rather than ship that, use `manual_urls.txt`: search Kenya
Law yourself, paste the resulting judgment URLs in (one per line), and
stages 2 and 3 will process them identically to anything found
automatically.

## Before you run this at scale

- Check each source's Terms of Service for automated access, separately
  from the `robots.txt` check this pipeline already performs.
- Start with a small batch (a handful of URLs) and read the output before
  scaling up.
- `REQUEST_DELAY_SECONDS` in `config.py` defaults to a conservative 3
  seconds per domain -- increase it if a source asks you to.
