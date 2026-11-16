"""Week 2: turn job descriptions into structured fields with an LLM.

    python -m jobsintel.enrich --limit 3 --show     # try a few, print results
    python -m jobsintel.enrich --limit 200          # normal run
    python -m jobsintel.enrich --stats              # tokens, cost, field coverage
    python -m jobsintel.enrich --list-models        # if the model name stops working
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass

from pydantic import ValidationError

from jobsintel.config import Settings
from jobsintel.extraction import GEMINI_SCHEMA, PROMPT_VERSION, JobExtraction, build_prompt
from jobsintel.llm import GeminiClient, LLMError, LLMResult

log = logging.getLogger("jobsintel.enrich")

MIN_DESCRIPTION_CHARS = 200
MAX_CONSECUTIVE_ERRORS = 5  # usually means the daily free quota is used up


def cost_usd(result: LLMResult, settings: Settings) -> float:
    return (result.input_tokens * settings.llm_input_usd_per_m
            + result.output_tokens * settings.llm_output_usd_per_m) / 1_000_000


def extract_one(client: GeminiClient, job: dict) -> tuple[JobExtraction, LLMResult]:
    prompt = build_prompt(job["title"], job.get("company"), job.get("location"), job["description"])
    result = client.generate_json(prompt, GEMINI_SCHEMA)
    return JobExtraction.model_validate(result.data), result


class EnrichmentRepo:
    """All SQL for enrichment lives here."""

    def __init__(self, conn):
        self.conn = conn

    def _rows(self, sql: str, params: tuple) -> list[dict]:
        from psycopg.rows import dict_row
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def pending(self, limit: int, only_full: bool) -> list[dict]:
        """Jobs never enriched with this prompt version, or whose description changed since."""
        return self._rows("""
            SELECT j.id, j.title, j.company, j.location, j.description, j.description_hash
            FROM jobs j
            LEFT JOIN job_enrichments e ON e.job_id = j.id AND e.prompt_version = %s
            WHERE (e.id IS NULL OR e.description_hash IS DISTINCT FROM j.description_hash)
              AND (j.description_is_full OR NOT %s)
              AND length(j.description) >= %s
            ORDER BY j.first_seen DESC
            LIMIT %s""", (PROMPT_VERSION, only_full, MIN_DESCRIPTION_CHARS, limit))

    def jobs_by_ids(self, ids: list[int]) -> list[dict]:
        return self._rows("""
            SELECT id, title, company, location, description, description_hash
            FROM jobs WHERE id = ANY(%s)""", (ids,))

    def enrichments(self, ids: list[int]) -> dict[int, dict]:
        rows = self._rows("""
            SELECT job_id, extracted FROM job_enrichments
            WHERE job_id = ANY(%s) AND prompt_version = %s""", (ids, PROMPT_VERSION))
        return {r["job_id"]: r["extracted"] for r in rows}

    def save(self, job: dict, ext: JobExtraction, result: LLMResult, model: str, cost: float) -> None:
        from psycopg.types.json import Jsonb
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO job_enrichments (job_id, prompt_version, model, description_hash, skills,
                    experience_years_min, seniority, work_mode, visa_sponsorship, role_family,
                    extracted, input_tokens, output_tokens, cost_usd, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id, prompt_version) DO UPDATE SET
                    model = EXCLUDED.model, description_hash = EXCLUDED.description_hash,
                    skills = EXCLUDED.skills, experience_years_min = EXCLUDED.experience_years_min,
                    seniority = EXCLUDED.seniority, work_mode = EXCLUDED.work_mode,
                    visa_sponsorship = EXCLUDED.visa_sponsorship, role_family = EXCLUDED.role_family,
                    extracted = EXCLUDED.extracted, input_tokens = EXCLUDED.input_tokens,
                    output_tokens = EXCLUDED.output_tokens, cost_usd = EXCLUDED.cost_usd,
                    latency_ms = EXCLUDED.latency_ms, created_at = now()""",
                (job["id"], PROMPT_VERSION, model, job.get("description_hash"), ext.skills,
                 ext.experience_years_min, ext.seniority, ext.work_mode, ext.visa_sponsorship,
                 ext.role_family, Jsonb(ext.model_dump()), result.input_tokens,
                 result.output_tokens, cost, result.latency_ms))
        self.conn.commit()

    def stats(self) -> dict:
        return self._rows("""
            SELECT count(*)                                   AS jobs_enriched,
                   round(avg(input_tokens))                   AS avg_input_tokens,
                   round(avg(output_tokens))                  AS avg_output_tokens,
                   round(avg(latency_ms))                     AS avg_latency_ms,
                   round(sum(cost_usd)::numeric, 4)           AS total_cost_usd,
                   round((avg(cost_usd) * 1000)::numeric, 4)  AS cost_per_1000_jobs_usd,
                   round(100.0 * avg((visa_sponsorship <> 'unknown')::int), 1) AS pct_visa_known,
                   round(100.0 * avg((work_mode <> 'unknown')::int), 1)        AS pct_work_mode_known,
                   round(100.0 * avg((experience_years_min IS NOT NULL)::int), 1) AS pct_years_known,
                   round(avg(cardinality(skills)), 1)          AS avg_skills_per_job
            FROM job_enrichments WHERE prompt_version = %s""", (PROMPT_VERSION,))[0]


@dataclass
class RunStats:
    ok: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


def enrich_jobs(jobs: list[dict], client: GeminiClient, repo, settings: Settings,
                show: bool = False) -> RunStats:
    stats = RunStats()
    consecutive_errors = 0
    for i, job in enumerate(jobs, 1):
        try:
            ext, result = extract_one(client, job)
        except (LLMError, ValidationError) as exc:
            stats.failed += 1
            consecutive_errors += 1
            log.warning("job %s failed: %s", job["id"], str(exc)[:200])
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error("%d errors in a row (daily quota?), stopping", consecutive_errors)
                break
            continue
        consecutive_errors = 0
        cost = cost_usd(result, settings)
        repo.save(job, ext, result, settings.gemini_model, cost)
        stats.ok += 1
        stats.input_tokens += result.input_tokens
        stats.output_tokens += result.output_tokens
        stats.cost += cost
        if show:
            print(f"\n[{job['id']}] {job['title']} @ {job.get('company')}")
            print(json.dumps(ext.model_dump(), indent=2))
        elif i % 10 == 0:
            log.info("%d/%d done", i, len(jobs))
    log.info("enriched %d, failed %d, tokens in/out %d/%d, est. cost $%.4f",
             stats.ok, stats.failed, stats.input_tokens, stats.output_tokens, stats.cost)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM extraction of skills, seniority, visa, ...")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--include-snippets", action="store_true",
                    help="also enrich Adzuna snippets (less info, mostly 'unknown')")
    ap.add_argument("--show", action="store_true", help="print each extraction")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    if not settings.gemini_api_key:
        log.warning("GEMINI_API_KEY not set, skipping enrichment")
        return 0  # don't fail the daily workflow just because the key isn't configured
    client = GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.gemini_min_interval)

    if args.list_models:
        names = client.list_models()
        print("\n".join(n for n in names if "gemini" in n))
        print(f"\nCurrent GEMINI_MODEL: {settings.gemini_model}")
        return 0

    if not settings.database_url:
        log.error("DATABASE_URL not set")
        return 2
    import psycopg
    repo = EnrichmentRepo(psycopg.connect(settings.database_url))

    if args.stats:
        for k, v in repo.stats().items():
            print(f"{k:28} {v}")
        return 0

    jobs = repo.pending(args.limit, only_full=not args.include_snippets)
    log.info("%d jobs to enrich (model %s, prompt %s)", len(jobs), settings.gemini_model, PROMPT_VERSION)
    stats = enrich_jobs(jobs, client, repo, settings, show=args.show)
    return 1 if jobs and stats.ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
