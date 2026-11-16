"""Settings from environment variables (.env locally, GitHub Secrets in CI)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

DEFAULT_TERMS = [
    "data scientist",
    "data analyst",
    "machine learning engineer",
    "data engineer",
    "ai engineer",
    "analytics engineer",
    "mlops engineer",
]


@dataclass
class Settings:
    adzuna_app_id: str | None = field(default_factory=lambda: os.getenv("ADZUNA_APP_ID"))
    adzuna_app_key: str | None = field(default_factory=lambda: os.getenv("ADZUNA_APP_KEY"))
    reed_api_key: str | None = field(default_factory=lambda: os.getenv("REED_API_KEY"))
    database_url: str | None = field(default_factory=lambda: os.getenv("DATABASE_URL"))
    search_terms: list[str] = field(default_factory=lambda: [
        t.strip() for t in os.getenv("SEARCH_TERMS", ",".join(DEFAULT_TERMS)).split(",") if t.strip()])
    max_pages: int = field(default_factory=lambda: int(os.getenv("MAX_PAGES", "3")))
    # 7 terms x 3 pages = 21 calls/day ~= 630/month, well under Adzuna's 2,500.
    adzuna_call_budget: int = field(default_factory=lambda: int(os.getenv("ADZUNA_CALL_BUDGET", "40")))
    reed_call_budget: int = field(default_factory=lambda: int(os.getenv("REED_CALL_BUDGET", "300")))
    # Details calls for full descriptions, only for Reed jobs we haven't enriched yet.
    reed_max_details: int = field(default_factory=lambda: int(os.getenv("REED_MAX_DETAILS", "250")))
