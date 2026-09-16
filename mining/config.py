"""
config.py -- shared configuration for the water-land conflict data
collection pipeline. Edit the values below before running; nothing
here needs to be touched inside the other scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Scope -- matches the four counties and schema used throughout the rest of
# this project (Chapter Three, section 3.4).
# ---------------------------------------------------------------------------
TARGET_COUNTIES = ["Nairobi", "Kiambu", "Machakos", "Turkana"]

# The exact field order requested -- every output CSV in this pipeline uses
# this schema, so it drops directly into the same structure as the existing
# compiled dataset.
SCHEMA = [
    "Record_ID", "Original_Record_ID", "County", "Sub_Location", "Date_Start",
    "Date_End", "Incident_Summary", "Conflict_Type", "Conflict_Subtype",
    "Parties_Involved", "Full_Text_Description", "Source_Name", "Source_URL",
    "Data_Source_Type", "Latitude", "Longitude", "Casualties_Reported",
    "Displaced_Persons", "Legal_Status", "Outcome_Status", "NLP_Keywords",
    "Verified", "Confidence_Score", "Notes",
]

# Search keyword fragments -- combined with each county name to form a
# query. Matches the keyword/date-range logic described in Chapter Three,
# section 3.5.2 (land disputes, water access disputes, riparian
# encroachment, eviction, and related terms).
CONFLICT_KEYWORDS = [
    "land dispute", "water conflict", "riparian encroachment", "eviction",
    "land grabbing", "water access dispute", "pastoralist conflict",
    "cattle raid", "grazing rights dispute", "water abstraction dispute",
    "boundary dispute", "informal settlement demolition",
]

# ---------------------------------------------------------------------------
# Known source domains, mapped to Data_Source_Type. Extend this list with
# any additional sources you compile from; anything not listed here falls
# back to "Print/Online News" as a default (edit that default below if a
# new, unlisted domain is not actually news).
# ---------------------------------------------------------------------------
SOURCE_TYPE_MAP = {
    "kenyalaw.org": "Court Record",
    "nation.africa": "Print/Online News",
    "standardmedia.co.ke": "Print/Online News",
    "the-star.co.ke": "Print/Online News",
    "businessdailyafrica.com": "Print/Online News",
    "capitalfm.co.ke": "Print/Online News",
    "nema.go.ke": "Government",
    "wra.go.ke": "Government",
    "water.go.ke": "Government",
    "knchr.org": "NGO",
    "hrw.org": "NGO",
    "amnesty.org": "NGO",
    "acleddata.com": "Conflict Data",
}
DEFAULT_SOURCE_TYPE = "Print/Online News"

# ---------------------------------------------------------------------------
# Networking behaviour -- deliberately conservative. Increase REQUEST_DELAY
# if a source's own rate limits require it; this pipeline identifies itself
# honestly via USER_AGENT rather than disguising its requests.
# ---------------------------------------------------------------------------
USER_AGENT = (
    "WaterLandConflictResearchBot/1.0 "
    "(MSc research data collection; contact: <INSERT YOUR EMAIL HERE>)"
)
REQUEST_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 15
RESPECT_ROBOTS_TXT = True  # do not set False without a specific, considered reason

# ---------------------------------------------------------------------------
# Optional search API -- if you have a key for a search API (e.g. Bing Web
# Search, SerpAPI, Google Custom Search), set the provider and key here to
# enable automated query-based discovery in 01_discover_candidates.py.
# Leaving SEARCH_API_KEY empty disables this and falls back to the direct
# RSS/sitemap connectors defined in connectors.py, which need no key but
# cover fewer sources.
# ---------------------------------------------------------------------------
SEARCH_API_PROVIDER = ""     # "bing" | "serpapi" | "" (disabled)
SEARCH_API_KEY = ""          # leave blank if you don't have one yet
MAX_RESULTS_PER_QUERY = 15

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "collection_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CANDIDATES_PATH = OUTPUT_DIR / "01_candidates.csv"
REVIEW_QUEUE_PATH = OUTPUT_DIR / "02_review_queue.csv"
DEDUPE_FLAGGED_PATH = OUTPUT_DIR / "03_dedupe_flagged.csv"
DEDUPE_CLEAR_PATH = OUTPUT_DIR / "03_ready_for_review.csv"

# Path to your EXISTING compiled dataset, used by 03_dedupe_check.py to
# avoid re-collecting records you already have. Leave as None to skip
# checking against the existing dataset (new candidates will still be
# checked against each other).
EXISTING_DATASET_PATH = None  # e.g. Path("data/Final_Main_kenya_land_water_conflicts.csv")

# Near-duplicate detection parameters. NOTE: this is 0.30, not the 0.80
# documented for the main pipeline's own near-duplicate check (Chapter
# Three, section 3.7.4) -- that difference is intentional, not an
# inconsistency. This script uses word-overlap (Jaccard) similarity,
# which testing found gives a clean, well-separated signal at this
# threshold (~0.45 for two outlets' paraphrased coverage of the same
# real event, ~0.12 for two genuinely different incidents sharing only
# a topic and county); character-level similarity, the likely method
# behind the main pipeline's 80% figure, badly under-scored genuine
# paraphrased duplicates in testing (55%, below its own threshold).
# Recalibrate this value if you switch metrics again.
DEDUPE_SIMILARITY_THRESHOLD = 0.30
DEDUPE_DATE_WINDOW_DAYS = 30
