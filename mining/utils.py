"""
utils.py -- shared helpers for the data collection pipeline: polite,
robots.txt-respecting HTTP requests, source-type classification, and
CSV I/O consistent with the rest of this project.
"""

import time
import urllib.robotparser
from urllib.parse import urlparse

import pandas as pd
import requests

import config

_last_request_time = {}  # per-domain, so rate limiting is per-source, not global
_robots_cache = {}


def read_csv_robust(path, **kwargs) -> pd.DataFrame:
    """Same encoding fallback chain used throughout the rest of this
    project -- this project's own source files were found not to read
    correctly under plain UTF-8."""
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not read {path} with utf-8, cp1252, or latin-1.")


def get_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def classify_source_type(url: str) -> str:
    domain = get_domain(url)
    for known_domain, source_type in config.SOURCE_TYPE_MAP.items():
        if known_domain in domain:
            return source_type
    return config.DEFAULT_SOURCE_TYPE


def robots_allows(url: str) -> bool:
    """Checks robots.txt for the URL's domain before it is ever fetched.
    Cached per-domain so this doesn't re-fetch robots.txt on every URL."""
    if not config.RESPECT_ROBOTS_TXT:
        return True
    domain = get_domain(url)
    if domain not in _robots_cache:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception:
            # If robots.txt can't be fetched at all, default to allowing
            # but note this rather than silently proceeding -- some
            # sites simply don't have one, which is not the same as
            # disallowing everything.
            print(f"    NOTE: could not read robots.txt for {domain} -- proceeding, but verify manually.")
            _robots_cache[domain] = None
            return True
        _robots_cache[domain] = rp
    rp = _robots_cache[domain]
    if rp is None:
        return True
    return rp.can_fetch(config.USER_AGENT, url)


def polite_get(url: str, session: requests.Session = None) -> requests.Response | None:
    """Rate-limited (per-domain), robots.txt-respecting GET request.
    Returns None (rather than raising) on any failure, so a single bad
    URL doesn't stop a batch run -- the caller is expected to check for
    None and log/skip accordingly."""
    if not robots_allows(url):
        print(f"    SKIPPED (robots.txt disallows): {url}")
        return None

    domain = get_domain(url)
    now = time.time()
    elapsed = now - _last_request_time.get(domain, 0)
    if elapsed < config.REQUEST_DELAY_SECONDS:
        time.sleep(config.REQUEST_DELAY_SECONDS - elapsed)
    _last_request_time[domain] = time.time()

    sess = session or requests.Session()
    try:
        resp = sess.get(url, headers={"User-Agent": config.USER_AGENT},
                          timeout=config.REQUEST_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            print(f"    SKIPPED (HTTP {resp.status_code}): {url}")
            return None
        return resp
    except requests.RequestException as e:
        print(f"    SKIPPED (request error: {e}): {url}")
        return None


def generate_record_id(county: str, sequence: int) -> str:
    county_code = {"Nairobi": "NBI", "Kiambu": "KBU", "Machakos": "MCK", "Turkana": "TRK"}.get(county, "UNK")
    return f"KEN-{county_code}-{sequence:04d}"


def empty_schema_row() -> dict:
    return {field: "" for field in config.SCHEMA}
