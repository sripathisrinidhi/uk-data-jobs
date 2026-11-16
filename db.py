"""Storage backends. PostgresStore for real runs, JsonlStore for --dry-run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from jobsintel.models import JobPosting

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "001_init.sql"

COLUMNS = [
    "source", "source_id", "title", "company", "location", "location_area",
    "salary_min", "salary_max", "salary_is_predicted", "salary_period",
    "contract_type", "contract_time", "category", "description",
    "description_is_full", "description_hash", "url", "posted_at",
    "search_terms", "raw",
]

UPSERT_SQL = f"""
INSERT INTO jobs ({", ".join(COLUMNS)})
VALUES ({", ".join(["%s"] * len(COLUMNS))})
ON CONFLICT (source, source_id) DO UPDATE SET
    last_seen           = now(),
    title               = EXCLUDED.title,
    salary_min          = EXCLUDED.salary_min,
    salary_max          = EXCLUDED.salary_max,
    salary_is_predicted = EXCLUDED.salary_is_predicted,
    salary_period       = EXCLUDED.salary_period,
    -- never overwrite a full description with a snippet
    description = CASE WHEN EXCLUDED.description_is_full OR NOT jobs.description_is_full
                       THEN EXCLUDED.description ELSE jobs.description END,
    description_hash = CASE WHEN EXCLUDED.description_is_full OR NOT jobs.description_is_full
                            THEN EXCLUDED.description_hash ELSE jobs.description_hash END,
    description_is_full = jobs.description_is_full OR EXCLUDED.description_is_full,
    search_terms = ARRAY(SELECT DISTINCT unnest(jobs.search_terms || EXCLUDED.search_terms)),
    raw = EXCLUDED.raw
RETURNING (xmax = 0) AS inserted
"""


class Store(Protocol):
    def ids_with_full_description(self, source: str, ids: Iterable[str]) -> set[str]: ...
    def upsert(self, postings: list[JobPosting]) -> tuple[int, int]: ...
    def log_run(self, started_at: datetime, source: str, api_calls: int, fetched: int,
                inserted: int, updated: int, errors: list[str]) -> None: ...


def _row(p: JobPosting) -> list:
    d = p.model_dump()
    return [json.dumps(d[c], default=str) if c == "raw" else d[c] for c in COLUMNS]


class PostgresStore:
    def __init__(self, dsn: str):
        import psycopg  # imported lazily so --dry-run works without it
        self.conn = psycopg.connect(dsn, autocommit=False)

    def init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        self.conn.commit()

    def ids_with_full_description(self, source: str, ids: Iterable[str]) -> set[str]:
        ids = list(ids)
        if not ids:
            return set()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT source_id FROM jobs WHERE source = %s AND source_id = ANY(%s) "
                "AND description_is_full", (source, ids))
            return {r[0] for r in cur.fetchall()}

    def upsert(self, postings: list[JobPosting]) -> tuple[int, int]:
        from psycopg.types.json import Jsonb
        raw_idx = COLUMNS.index("raw")
        inserted = updated = 0
        with self.conn.cursor() as cur:
            for p in postings:
                row = _row(p)
                row[raw_idx] = Jsonb(p.raw)
                cur.execute(UPSERT_SQL, row)
                if cur.fetchone()[0]:
                    inserted += 1
                else:
                    updated += 1
        self.conn.commit()
        return inserted, updated

    def log_run(self, started_at, source, api_calls, fetched, inserted, updated, errors) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO collection_runs (started_at, source, api_calls, fetched, inserted, updated, errors)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (started_at, source, api_calls, fetched, inserted, updated, errors))
        self.conn.commit()


class JsonlStore:
    """Writes to data/*.jsonl. Use it to test your API keys before setting up Postgres."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = out_dir / f"jobs_{stamp}.jsonl"
        self.seen: set[tuple[str, str]] = set()
        self.runs: list[dict] = []

    def ids_with_full_description(self, source, ids) -> set[str]:
        return set()

    def upsert(self, postings):
        with self.path.open("a") as f:
            for p in postings:
                f.write(p.model_dump_json(exclude={"raw"}) + "\n")
        new = sum(1 for p in postings if p.key not in self.seen)
        self.seen.update(p.key for p in postings)
        return new, len(postings) - new

    def log_run(self, started_at, source, api_calls, fetched, inserted, updated, errors):
        self.runs.append(dict(source=source, api_calls=api_calls, fetched=fetched,
                              inserted=inserted, updated=updated, errors=errors))
