from fastapi import APIRouter, Query

from .. import data_access

router = APIRouter(prefix="/api/conflicts", tags=["conflicts"])


@router.get("")
def conflicts(
    county: str | None = None,
    subcounty: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    topic_id: int | None = None,
    domain: str | None = Query(None, description="Land-only / Water-only / Mixed / Neither"),
    search: str | None = Query(None, description="Free-text search over incident summaries"),
    limit: int = Query(500, le=5000),
):
    """
    Individual conflict records -- powers the map's point layer and a
    searchable/filterable incident list view. This is the "search"
    half of the filter+search requirement.
    """
    df = data_access.query_conflicts(
        county=county, subcounty=subcounty, year_min=year_min, year_max=year_max,
        topic_id=topic_id, search=search, domain=domain,
    )
    df = df.head(limit)
    return data_access.df_to_json_records(df)
