-- UK Data Jobs Intelligence Platform: core schema (idempotent)

CREATE TABLE IF NOT EXISTS jobs (
    id                  BIGSERIAL PRIMARY KEY,
    source              TEXT        NOT NULL,          -- adzuna | reed
    source_id           TEXT        NOT NULL,
    title               TEXT        NOT NULL,
    company             TEXT,
    location            TEXT,
    location_area       TEXT[]      NOT NULL DEFAULT '{}',
    salary_min          NUMERIC,
    salary_max          NUMERIC,
    salary_is_predicted BOOLEAN     NOT NULL DEFAULT FALSE,
    salary_period       TEXT        NOT NULL DEFAULT 'unknown',  -- annual | daily | hourly | unknown
    contract_type       TEXT,
    contract_time       TEXT,
    category            TEXT,
    description         TEXT        NOT NULL DEFAULT '',
    description_is_full BOOLEAN     NOT NULL DEFAULT FALSE,
    description_hash    TEXT,
    url                 TEXT        NOT NULL,
    posted_at           TIMESTAMPTZ,
    search_terms        TEXT[]      NOT NULL DEFAULT '{}',
    first_seen          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen           TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw                 JSONB       NOT NULL DEFAULT '{}',
    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS jobs_posted_at_idx ON jobs (posted_at);
CREATE INDEX IF NOT EXISTS jobs_last_seen_idx ON jobs (last_seen);

-- One row per collector run: calls used, rows in/updated, errors.
CREATE TABLE IF NOT EXISTS collection_runs (
    id           BIGSERIAL PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL,
    finished_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    source       TEXT        NOT NULL,
    api_calls    INT         NOT NULL,
    fetched      INT         NOT NULL,
    inserted     INT         NOT NULL,
    updated      INT         NOT NULL,
    errors       TEXT[]      NOT NULL DEFAULT '{}'
);

-- Handy view for Week 3: rows that are safe to use as salary-model targets.
CREATE OR REPLACE VIEW jobs_salary_clean AS
SELECT *, (COALESCE(salary_min, salary_max) + COALESCE(salary_max, salary_min)) / 2.0 AS salary_mid
FROM jobs
WHERE salary_is_predicted = FALSE
  AND salary_period = 'annual'
  AND COALESCE(salary_min, salary_max) IS NOT NULL;
