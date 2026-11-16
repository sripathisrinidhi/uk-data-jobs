"""Normalised job posting model shared by every source."""
from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field

SalaryPeriod = Literal["annual", "daily", "hourly", "unknown"]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def infer_salary_period(salary_min: float | None, salary_max: float | None) -> SalaryPeriod:
    """UK postings mix annual, day-rate and hourly pay in the same fields.

    Heuristic: <£150 is hourly, <£2,000 is a day rate, otherwise annual.
    Keeps day rates (common for contract data roles) out of the salary model.
    """
    value = salary_max or salary_min
    if not value:
        return "unknown"
    if value < 150:
        return "hourly"
    if value < 2000:
        return "daily"
    return "annual"


class JobPosting(BaseModel):
    source: Literal["adzuna", "reed"]
    source_id: str
    title: str
    company: str | None = None
    location: str | None = None
    location_area: list[str] = Field(default_factory=list)
    salary_min: float | None = None
    salary_max: float | None = None
    # Adzuna fills in salaries it estimated itself. Never train on these.
    salary_is_predicted: bool = False
    contract_type: str | None = None   # permanent / contract
    contract_time: str | None = None   # full_time / part_time
    category: str | None = None
    description: str = ""
    description_is_full: bool = False  # False = snippet only
    url: str
    posted_at: datetime | None = None
    search_terms: list[str] = Field(default_factory=list)
    raw: dict = Field(default_factory=dict, repr=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def salary_period(self) -> SalaryPeriod:
        return infer_salary_period(self.salary_min, self.salary_max)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def description_hash(self) -> str:
        """Cache key for LLM enrichment later: same text -> pay once."""
        return hashlib.sha256(self.description.encode()).hexdigest()[:16]

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.source_id)
