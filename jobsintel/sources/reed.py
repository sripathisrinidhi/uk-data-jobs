"""Reed.co.uk job search + details.

Search returns a short snippet; /jobs/{id} returns the full description,
which is what the LLM extraction (Week 2) and RAG (Week 5) need.
Auth: HTTP Basic with the API key as username and an empty password.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator

from jobsintel.http import ApiClient
from jobsintel.models import JobPosting, clean_text

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.reed.co.uk/api/1.0/search"
DETAIL_URL = "https://www.reed.co.uk/api/1.0/jobs/{job_id}"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def _contract(item: dict) -> tuple[str | None, str | None]:
    ctype = "contract" if item.get("contractType") == "Contract" or item.get("contract") \
        else "permanent" if item.get("permanent") or item.get("contractType") == "Permanent" else None
    ctime = "full_time" if item.get("fullTime") else "part_time" if item.get("partTime") else None
    return ctype, ctime


def parse_reed(item: dict, search_term: str, full: bool = False) -> JobPosting:
    ctype, ctime = _contract(item)
    return JobPosting(
        source="reed",
        source_id=str(item["jobId"]),
        title=clean_text(item.get("jobTitle")),
        company=item.get("employerName"),
        location=item.get("locationName"),
        salary_min=item.get("minimumSalary"),
        salary_max=item.get("maximumSalary"),
        salary_is_predicted=False,
        contract_type=ctype,
        contract_time=ctime,
        description=clean_text(item.get("jobDescription")),
        description_is_full=full,
        url=item.get("jobUrl") or f"https://www.reed.co.uk/jobs/{item['jobId']}",
        posted_at=_parse_date(item.get("datePosted") or item.get("date")),
        search_terms=[search_term],
        raw=item,
    )


class ReedSource:
    name = "reed"

    def __init__(self, api_key: str, client: ApiClient, page_size: int = 100):
        self.auth = (api_key, "")
        self.client = client
        self.page_size = page_size

    def search(self, term: str, max_pages: int = 3, location: str | None = None) -> Iterator[JobPosting]:
        for page in range(max_pages):
            params = {"keywords": term, "resultsToTake": self.page_size,
                      "resultsToSkip": page * self.page_size}
            if location:
                params["locationName"] = location
            data = self.client.get_json(SEARCH_URL, params=params, auth=self.auth)
            results = data.get("results", [])
            log.info("reed '%s' page %d: %d results (total %s)",
                     term, page + 1, len(results), data.get("totalResults"))
            for item in results:
                try:
                    yield parse_reed(item, term)
                except Exception as exc:
                    log.warning("skip reed item %s: %s", item.get("jobId"), exc)
            if len(results) < self.page_size:
                break

    def fetch_full(self, posting: JobPosting) -> JobPosting:
        """Replace the snippet with the full description from the details endpoint."""
        detail = self.client.get_json(DETAIL_URL.format(job_id=posting.source_id), auth=self.auth)
        merged = {**posting.raw, **detail}
        full = parse_reed(merged, posting.search_terms[0], full=True)
        full.search_terms = posting.search_terms
        return full
