"""
connectors.py -- pluggable discovery connectors for 01_discover_candidates.py.

Each connector returns a list of dicts: {"url", "title", "snippet",
"source_name", "discovered_via"}. Candidates are filtered by keyword
match against title+snippet in 01_discover_candidates.py, not here.

Honesty about what has and hasn't been verified:
  - RSS_FEEDS below includes ONE feed (Daily Nation) that was fetched
    and confirmed live and parseable while building this script.
  - Kenya Law's case search does not have a simple, stable public API
    or RSS feed as far as could be established -- its current search
    interface is JavaScript-driven and its exact query parameters were
    not reliably determined. Rather than ship a scraper built on
    guessed selectors that could silently break or silently return
    nothing, MANUAL_URL_LIST is provided as the robust path for Kenya
    Law and any other source without a clean feed: search that site
    yourself, paste the resulting judgment/article URLs into
    manual_urls.txt (one per line), and this pipeline will fetch and
    process them identically to anything found automatically.
  - The search-API connector (search_api_discover) requires your own
    API key and has not been tested against a live key from this
    environment; verify it works with a small query before a full run.
"""

from pathlib import Path
import feedparser

import config

# Verified working as of the time this script was written -- re-check
# periodically, since sites do restructure their feeds.
RSS_FEEDS = [
    {"url": "https://nation.africa/kenya/rss.xml", "source_name": "Daily Nation"},
    # Add more here as you verify them, e.g. Standard Media, Business
    # Daily, The Star -- fetch the URL directly first (curl, or a
    # browser) to confirm it returns valid RSS/Atom XML before adding
    # it, the same way this one was checked.
]

MANUAL_URLS_PATH = config.PROJECT_ROOT / "manual_urls.txt"


def rss_discover() -> list[dict]:
    """Pulls current items from every feed in RSS_FEEDS. These feeds
    are general news firehoses, not conflict-specific -- keyword
    filtering happens afterward in 01_discover_candidates.py, not
    here."""
    results = []
    for feed in RSS_FEEDS:
        print(f"  Fetching RSS feed: {feed['source_name']} ({feed['url']})")
        try:
            parsed = feedparser.parse(feed["url"])
            if parsed.bozo and not parsed.entries:
                print(f"    WARNING: feed did not parse cleanly and returned no entries -- "
                      f"verify the URL is still current.")
                continue
            for entry in parsed.entries:
                results.append({
                    "url": entry.get("link", ""),
                    "title": entry.get("title", ""),
                    "snippet": entry.get("description", ""),
                    "source_name": feed["source_name"],
                    "discovered_via": "rss",
                })
            print(f"    {len(parsed.entries)} items retrieved")
        except Exception as e:
            print(f"    SKIPPED feed (error: {e})")
    return results


def manual_url_discover() -> list[dict]:
    """Reads manual_urls.txt (one URL per line, blank lines and lines
    starting with # ignored) -- the robust fallback for any source
    without a clean automated feed, e.g. Kenya Law case search
    results, or any specific article you found yourself and want
    pulled into the same review pipeline."""
    if not MANUAL_URLS_PATH.exists():
        MANUAL_URLS_PATH.write_text(
            "# Paste one URL per line below (lines starting with # are ignored).\n"
            "# Use this for sources without a working RSS feed or search API --\n"
            "# e.g. search kenyalaw.org yourself and paste judgment URLs here.\n"
        )
        print(f"  Created empty {MANUAL_URLS_PATH.name} -- add URLs to it and re-run to include them.")
        return []
    lines = [l.strip() for l in MANUAL_URLS_PATH.read_text().splitlines()]
    urls = [l for l in lines if l and not l.startswith("#")]
    print(f"  {len(urls)} URL(s) loaded from {MANUAL_URLS_PATH.name}")
    return [{"url": u, "title": "", "snippet": "", "source_name": "", "discovered_via": "manual"} for u in urls]


def search_api_discover(query: str) -> list[dict]:
    """Optional: queries a search API if config.SEARCH_API_KEY is set.
    NOT independently verified against a live key in this environment
    -- test with one query and inspect the results before a full run.
    Currently implements the Bing Web Search API's request shape;
    adjust the request/response handling here if you use a different
    provider."""
    if not config.SEARCH_API_KEY:
        return []
    import requests
    if config.SEARCH_API_PROVIDER == "bing":
        try:
            resp = requests.get(
                "https://api.bing.microsoft.com/v7.0/search",
                headers={"Ocp-Apim-Subscription-Key": config.SEARCH_API_KEY},
                params={"q": query, "count": config.MAX_RESULTS_PER_QUERY, "mkt": "en-KE"},
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            return [{
                "url": r["url"], "title": r.get("name", ""), "snippet": r.get("snippet", ""),
                "source_name": "", "discovered_via": f"search_api:{query}",
            } for r in data.get("webPages", {}).get("value", [])]
        except Exception as e:
            print(f"    Search API query failed (query={query!r}, error={e})")
            return []
    print(f"    NOTE: SEARCH_API_PROVIDER={config.SEARCH_API_PROVIDER!r} is not implemented "
          f"in search_api_discover() -- add handling for it or use 'bing'.")
    return []
