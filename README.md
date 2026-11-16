# UK Data Jobs Intelligence Platform

Collects UK Data / ML job postings every day from **Adzuna** and **Reed**, stores them in Postgres, and (in later weeks) enriches them with an LLM, analyses them, predicts salaries, and answers questions via RAG.

> **Status:** Week 1: data collection ✅

## Architecture (Week 1)

```
GitHub Actions (daily cron)
        │
        ▼
 jobsintel.collect ──► Adzuna API  (search, snippets, 250 calls/day cap)
        │         ──► Reed API    (search + /jobs/{id} for full descriptions)
        ▼
 Postgres (Neon / Supabase)
   ├─ jobs              one row per posting, upserted by (source, source_id)
   ├─ collection_runs   calls / inserted / updated / errors per run
   └─ jobs_salary_clean view: real annual salaries only (no Adzuna estimates, no day rates)
```

## Design decisions worth mentioning in interviews

| Decision | Why |
|---|---|
| Two sources | Adzuna only returns ~500-char snippets. Reed's details endpoint gives full descriptions for LLM extraction / RAG. |
| `salary_is_predicted` kept | Adzuna estimates missing salaries itself. Training on those means learning Adzuna's model: target leakage. |
| `salary_period` inferred | Contract roles quote £500/day in the same field as £60k/year. |
| `first_seen` / `last_seen` | Lets you measure how long roles stay open. |
| Full description never overwritten by a snippet | The upsert keeps the richer text. |
| Hard call budget per run | A bug can't burn the monthly API quota. |
| `description_hash` | Cache key so each description is paid for once in the LLM step. |

## Setup

1. **API keys** (free)
   - Adzuna: https://developer.adzuna.com → App ID + App Key
   - Reed: https://www.reed.co.uk/developers/jobseeker → API key
2. **Database**: create a free Postgres on [Neon](https://neon.tech) (pick region *London / eu-west-2*) and copy the connection string.
3. **Local run**
   ```bash
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env                                   # fill in keys
   pytest -q                                              # 14 tests, no network needed

   python -m jobsintel.collect --dry-run --terms "data scientist" --max-pages 1   # test keys -> data/*.jsonl
   python -m jobsintel.collect --init-db                  # create tables
   python -m jobsintel.collect                            # real run
   ```
4. **Daily automation**: push to GitHub, then *Settings → Secrets and variables → Actions* and add `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`, `REED_API_KEY`, `DATABASE_URL`. Trigger once from the *Actions* tab ("Run workflow") to check.

## Useful queries

```sql
-- How much data so far?
SELECT source, count(*), count(*) FILTER (WHERE description_is_full) AS full_desc
FROM jobs GROUP BY source;

-- New jobs per day
SELECT date(first_seen), count(*) FROM jobs GROUP BY 1 ORDER BY 1 DESC;

-- API usage per run
SELECT started_at, source, api_calls, inserted, updated, cardinality(errors) AS errs
FROM collection_runs ORDER BY started_at DESC LIMIT 10;
```

## Roadmap

- [x] Week 1: collection
- [ ] Week 2: LLM enrichment (skills, seniority, remote, visa) + 50-job labelled eval set
- [ ] Week 3: SQL analysis + dashboard
- [ ] Week 4: salary model (baseline → LightGBM, SHAP, MLflow)
- [ ] Week 5: RAG chatbot + CV matching
- [ ] Week 6: FastAPI, Docker, deploy, CI/CD
- [ ] Week 7–8: write-up, demo video

Job data: Jobs by Adzuna, Reed.co.uk.
