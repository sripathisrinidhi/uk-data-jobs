"""Daily collector.

    python -m jobsintel.collect --dry-run     # test keys, writes data/*.jsonl
    python -m jobsintel.collect --init-db     # create tables in Postgres
    python -m jobsintel.collect               # normal daily run
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jobsintel.config import Settings
from jobsintel.db import JsonlStore, PostgresStore, Store
from jobsintel.http import ApiClient, BudgetExceeded, CallBudget
from jobsintel.models import JobPosting
from jobsintel.sources.adzuna import AdzunaSource
from jobsintel.sources.reed import ReedSource

log = logging.getLogger("jobsintel.collect")


@dataclass
class SourceStats:
    source: str
    api_calls: int = 0
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    errors: list[str] = field(default_factory=list)


def gather(source, terms: list[str], max_pages: int, stats: SourceStats) -> list[JobPosting]:
    """Run every search term; merge duplicates found under several terms."""
    by_key: dict[tuple[str, str], JobPosting] = {}
    for term in terms:
        try:
            for p in source.search(term, max_pages=max_pages):
                if p.key in by_key:
                    existing = by_key[p.key]
                    existing.search_terms = sorted(set(existing.search_terms) | set(p.search_terms))
                else:
                    by_key[p.key] = p
        except BudgetExceeded as exc:
            stats.errors.append(f"{term}: {exc}")
            log.warning("%s budget hit during '%s', stopping searches", source.name, term)
            break
        except Exception as exc:
            stats.errors.append(f"{term}: {exc!r}")
            log.exception("%s search failed for '%s'", source.name, term)
    return list(by_key.values())


def enrich_reed(reed: ReedSource, store: Store, postings: list[JobPosting],
                max_details: int, stats: SourceStats) -> list[JobPosting]:
    have_full = store.ids_with_full_description("reed", [p.source_id for p in postings])
    todo = [p for p in postings if p.source_id not in have_full][:max_details]
    log.info("reed: fetching full descriptions for %d jobs (%d already full)", len(todo), len(have_full))
    replaced: dict[str, JobPosting] = {}
    for p in todo:
        try:
            replaced[p.source_id] = reed.fetch_full(p)
        except BudgetExceeded as exc:
            stats.errors.append(f"details: {exc}")
            break
        except Exception as exc:
            stats.errors.append(f"details {p.source_id}: {exc!r}")
    return [replaced.get(p.source_id, p) for p in postings]


def run_source(name, source, store: Store, settings: Settings, budget: CallBudget) -> SourceStats:
    started = datetime.now(timezone.utc)
    stats = SourceStats(source=name)
    postings = gather(source, settings.search_terms, settings.max_pages, stats)
    if name == "reed" and postings:
        postings = enrich_reed(source, store, postings, settings.reed_max_details, stats)
    stats.fetched = len(postings)
    if postings:
        stats.inserted, stats.updated = store.upsert(postings)
    stats.api_calls = budget.used
    store.log_run(started, name, stats.api_calls, stats.fetched, stats.inserted, stats.updated, stats.errors)
    log.info("%s done: %d calls, %d fetched, %d new, %d updated, %d errors",
             name, stats.api_calls, stats.fetched, stats.inserted, stats.updated, len(stats.errors))
    return stats


def build_sources(settings: Settings, wanted: list[str]):
    sources = []
    if "adzuna" in wanted:
        if settings.adzuna_app_id and settings.adzuna_app_key:
            b = CallBudget(settings.adzuna_call_budget)
            # 25 calls/minute limit -> keep >= 2.5s between calls
            client = ApiClient(b, min_interval=2.5)
            sources.append(("adzuna", AdzunaSource(settings.adzuna_app_id, settings.adzuna_app_key, client), b))
        else:
            log.warning("ADZUNA_APP_ID / ADZUNA_APP_KEY not set, skipping Adzuna")
    if "reed" in wanted:
        if settings.reed_api_key:
            b = CallBudget(settings.reed_call_budget)
            client = ApiClient(b, min_interval=0.5)
            sources.append(("reed", ReedSource(settings.reed_api_key, client), b))
        else:
            log.warning("REED_API_KEY not set, skipping Reed")
    return sources


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect UK data/ML job postings")
    ap.add_argument("--dry-run", action="store_true", help="write JSONL to data/ instead of Postgres")
    ap.add_argument("--init-db", action="store_true", help="create tables and exit")
    ap.add_argument("--sources", default="adzuna,reed")
    ap.add_argument("--max-pages", type=int)
    ap.add_argument("--terms", help="comma-separated override, e.g. 'data scientist'")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    if args.max_pages:
        settings.max_pages = args.max_pages
    if args.terms:
        settings.search_terms = [t.strip() for t in args.terms.split(",") if t.strip()]

    if args.dry_run:
        store: Store = JsonlStore(Path("data"))
    else:
        if not settings.database_url:
            log.error("DATABASE_URL not set (use --dry-run to test without a database)")
            return 2
        store = PostgresStore(settings.database_url)
        if args.init_db:
            store.init_schema()
            log.info("schema created")
            return 0

    sources = build_sources(settings, [s.strip() for s in args.sources.split(",")])
    if not sources:
        log.error("no sources configured; check your .env")
        return 2

    all_stats = [run_source(name, src, store, settings, budget) for name, src, budget in sources]
    if args.dry_run:
        log.info("wrote %s", store.path)  # type: ignore[attr-defined]
    # Fail the CI job only if every source came back empty with errors.
    return 1 if all(s.fetched == 0 and s.errors for s in all_stats) else 0


if __name__ == "__main__":
    sys.exit(main())
