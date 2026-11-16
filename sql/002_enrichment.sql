-- Week 2: LLM-extracted structure for each job (idempotent)

CREATE TABLE IF NOT EXISTS job_enrichments (
    id                   BIGSERIAL PRIMARY KEY,
    job_id               BIGINT      NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    prompt_version       TEXT        NOT NULL,
    model                TEXT        NOT NULL,
    description_hash     TEXT,
    skills               TEXT[]      NOT NULL DEFAULT '{}',
    experience_years_min INT,
    seniority            TEXT,       -- intern | junior | mid | senior | lead | manager | unknown
    work_mode            TEXT,       -- remote | hybrid | onsite | unknown
    visa_sponsorship     TEXT,       -- yes | no | unknown
    role_family          TEXT,
    extracted            JSONB       NOT NULL,   -- full validated output
    input_tokens         INT         NOT NULL DEFAULT 0,
    output_tokens        INT         NOT NULL DEFAULT 0,
    cost_usd             NUMERIC(12, 8) NOT NULL DEFAULT 0,
    latency_ms           INT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, prompt_version)
);

CREATE INDEX IF NOT EXISTS job_enrichments_skills_idx ON job_enrichments USING GIN (skills);

-- Jobs + their latest enrichment, ready for Week 3 analysis.
CREATE OR REPLACE VIEW jobs_enriched AS
SELECT DISTINCT ON (j.id)
       j.*, e.prompt_version, e.model, e.skills, e.experience_years_min, e.seniority,
       e.work_mode, e.visa_sponsorship, e.role_family
FROM jobs j
JOIN job_enrichments e ON e.job_id = j.id
ORDER BY j.id, e.created_at DESC;
