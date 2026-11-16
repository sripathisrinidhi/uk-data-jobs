"""Adzuna UK job search.

Free key limits (ToS): 25/min, 250/day, 1,000/week, 2,500/month.
Descriptions are ~500-char snippets, so description_is_full=False.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator

from jobsintel.http import ApiClient
from jobsintel.models import JobPosting, clean_text

log = logging.getLogger(__name__)

SEARCH_URL = "https://api.adzuna.com/v1/api/jobs/gb/search/{page}"


def parse_adzuna(item: dict, search_term: str) -> JobPosting:
    created = item.get("created")
    return JobPosting(
        source="adzuna",
        source_id=str(item["id"]),
        title=clean_text(item.get("title")),
        company=(item.get("company") or {}).get("display_name"),
        location=(item.get("location") or {}).get("display_name"),
        location_area=(item.get("location") or {}).get("area") or [],
        salary_min=item.get("salary_min"),
        salary_max=item.get("salary_max"),
        salary_is_predicted=str(item.get("salary_is_predicted", "0")) == "1",
        contract_type=item.get("contract_type"),
        contract_time=item.get("contract_time"),
        category=(item.get("category") or {}).get("label"),
        description=clean_text(item.get("description")),
        description_is_full=False,
        url=item.get("redirect_url", ""),
        posted_at=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else None,
        search_terms=[search_term],
        raw=item,
    )


class AdzunaSource:
    name = "adzuna"

    def __init__(self, app_id: str, app_key: str, client: ApiClient,
                 results_per_page: int = 50, max_days_old: int = 3):
        self.app_id = app_id
        self.app_key = app_key
        self.client = client
        self.results_per_page = results_per_page
        self.max_days_old = max_days_old

    def search(self, term: str, max_pages: int = 3) -> Iterator[JobPosting]:
        for page in range(1, max_pages + 1):
            params = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "what_phrase": term,          # exact phrase: avoids "data entry" noise
                "results_per_page": self.results_per_page,
                "max_days_old": self.max_days_old,
                "sort_by": "date",
                "content-type": "application/json",
            }
            data = self.client.get_json(SEARCH_URL.format(page=page), params=params)
            results = data.get("results", [])
            log.info("adzuna '%s' page %d: %d results (total %s)",
                     term, page, len(results), data.get("count"))
            for item in results:
                try:
                    yield parse_adzuna(item, term)
                except Exception as exc:  # one bad row shouldn't kill the run
                    log.warning("skip adzuna item %s: %s", item.get("id"), exc)
            if len(results) < self.results_per_page:
                break  # last page, don't waste a call
