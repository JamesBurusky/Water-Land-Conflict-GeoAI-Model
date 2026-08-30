"""
name_cleaning.py

Canonical sub-county name resolution for the GeoAI conflict-prediction
project.

Why this module exists
-----------------------
Every source dataset (WRA abstraction records, WRUA governance data, the
conflict dataset, NDVI/CHIRPS remote-sensing extracts) spells sub-county
names differently: "ATHI RIVER" vs "Athi-River" vs "Athi River". Left
alone, a naive groupby/join treats these as different places, silently
fragmenting population, abstraction, and environmental signal across
duplicate keys. This module builds ONE canonical sub-county list (from
the KNBS 2019 census -- the cleanest, most authoritative source we have)
and provides a matcher that maps any messy variant onto it, with a
confidence score so low-confidence matches can be routed to manual
review instead of silently accepted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process

TARGET_COUNTIES = ["Nairobi", "Kiambu", "Machakos", "Turkana"]

# Manual overrides for cases fuzzy matching alone will get wrong --
# e.g. apostrophes, abbreviations, or genuinely different string forms
# for the same place. Extend this table as new mismatches are found;
# treat it as a living artifact of the audit, not a one-off patch.
MANUAL_ALIASES = {
    "langata": "lang'ata",
    "kiambu west": "kiambu",  # placeholder -- verify against actual KNBS
                                # sub-county list; "Kiambu West" is not a
                                # standard KNBS sub-county name and may
                                # refer to a WRA-internal admin zone.
}


def normalize_name(raw: str) -> str:
    """
    Lowercase, strip accents/punctuation, collapse whitespace/hyphens.
    This is the "coarse" normalization applied before fuzzy matching --
    it deliberately does NOT try to fix real spelling differences
    (that's what rapidfuzz is for), only formatting noise.
    """
    if pd.isna(raw):
        return ""
    s = str(raw).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z'\s]", "", s)   # keep apostrophes (e.g. lang'ata)
    s = re.sub(r"\s+", " ", s).strip()
    return MANUAL_ALIASES.get(s, s)


@dataclass
class CanonicalLookup:
    """Canonical sub-county table plus the matcher built from it."""
    table: pd.DataFrame          # columns: County, SubCounty, _norm
    choices: dict                # {_norm: (County, SubCounty)}

    def match(self, raw_name: str, county_hint: str | None = None,
              score_cutoff: float = 80.0):
        """
        Fuzzy-match a raw sub-county string to the canonical list.

        Returns (matched_county, matched_subcounty, score, method) where
        method is 'exact', 'fuzzy', or 'unmatched'. If county_hint is
        given, matching is restricted to that county's sub-counties
        first (much safer -- avoids matching e.g. a Machakos sub-county
        name to a similarly-spelled Kiambu one).
        """
        norm = normalize_name(raw_name)
        if not norm:
            return None, None, 0.0, "unmatched"

        pool = self.choices
        if county_hint:
            hint_norm = normalize_name(county_hint)
            restricted = {
                k: v for k, v in self.choices.items()
                if normalize_name(v[0]) == hint_norm
            }
            if restricted:
                pool = restricted

        if norm in pool:
            county, subcounty = pool[norm]
            return county, subcounty, 100.0, "exact"

        result = process.extractOne(
            norm, list(pool.keys()), scorer=fuzz.WRatio
        )
        if result is None:
            return None, None, 0.0, "unmatched"
        best_key, score, _ = result
        if score < score_cutoff:
            return None, None, score, "unmatched"
        county, subcounty = pool[best_key]
        return county, subcounty, score, "fuzzy"


def build_canonical_lookup(census_df: pd.DataFrame,
                            county_col: str = "County",
                            subcounty_col: str = "SubCounty") -> CanonicalLookup:
    """
    Build the canonical (County, SubCounty) table from the census data,
    restricted to the four target counties.
    """
    df = census_df[[county_col, subcounty_col]].drop_duplicates().copy()
    df = df[df[county_col].isin(TARGET_COUNTIES)].reset_index(drop=True)
    df["_norm"] = df[subcounty_col].map(normalize_name)
    choices = {
        norm_val: (county_val, subcounty_val)
        for norm_val, county_val, subcounty_val in zip(
            df["_norm"], df[county_col], df[subcounty_col]
        )
    }
    return CanonicalLookup(table=df, choices=choices)


def match_dataframe(df: pd.DataFrame, subcounty_col: str,
                     lookup: CanonicalLookup,
                     county_col: str | None = None,
                     score_cutoff: float = 80.0) -> pd.DataFrame:
    """
    Apply the matcher to every row of a dataframe. Adds:
      matched_county, matched_subcounty, match_score, match_method
    Rows with match_method == 'unmatched' need manual review -- they are
    NOT silently dropped, so nothing disappears from the audit trail.
    """
    out = df.copy()
    results = out.apply(
        lambda r: lookup.match(
            r[subcounty_col],
            county_hint=r[county_col] if county_col and county_col in out.columns else None,
            score_cutoff=score_cutoff,
        ),
        axis=1,
    )
    out["matched_county"] = results.map(lambda t: t[0])
    out["matched_subcounty"] = results.map(lambda t: t[1])
    out["match_score"] = results.map(lambda t: t[2])
    out["match_method"] = results.map(lambda t: t[3])
    return out
