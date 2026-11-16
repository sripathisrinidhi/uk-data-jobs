import json

import pytest

from jobsintel.config import Settings
from jobsintel.enrich import RunStats, enrich_jobs
from jobsintel.evaluate import score, split_skills
from jobsintel.extraction import (GEMINI_SCHEMA, JobExtraction, build_prompt,
                                  normalize_skills)
from jobsintel.llm import GeminiClient, LLMError, LLMResult

GOOD = {"skills": ["python", "Amazon Web Services", "SQL", "sql", "PowerBI"],
        "experience_years_min": 3, "seniority": "senior", "work_mode": "hybrid",
        "visa_sponsorship": "yes", "role_family": "data_scientist"}


# ---------- extraction ----------

def test_normalize_skills_dedupes_and_canonicalises():
    assert normalize_skills(["python", "Python ", "AWS", "amazon web services", "sklearn", ""]) == \
        ["AWS", "Python", "scikit-learn"]
    assert normalize_skills(["Amazon Web Services", "PowerBI", "k8s"]) == ["AWS", "Kubernetes", "Power BI"]


def test_extraction_model_validates_and_normalises():
    ext = JobExtraction.model_validate(GOOD)
    assert ext.skills == ["AWS", "Power BI", "Python", "SQL"]
    assert ext.seniority == "senior"


def test_extraction_rejects_bad_enum_and_drops_silly_years():
    with pytest.raises(Exception):
        JobExtraction.model_validate({**GOOD, "seniority": "wizard"})
    assert JobExtraction.model_validate({**GOOD, "experience_years_min": 99}).experience_years_min is None


def test_gemini_schema_matches_pydantic_fields():
    assert set(GEMINI_SCHEMA["properties"]) == set(JobExtraction.model_fields)


def test_prompt_truncates_long_descriptions():
    p = build_prompt("Data Scientist", None, "London", "x" * 20000)
    assert "Data Scientist" in p and "unknown" in p
    assert len(p) < 9000


# ---------- Gemini client ----------

class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code, self._p = status, payload or {}
        self.text = json.dumps(self._p)

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append((url, json))
        return self.responses.pop(0)


def gemini_ok(data):
    return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(data)}]}}],
                          "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 60,
                                            "thoughtsTokenCount": 10}})


def client_with(*responses):
    return GeminiClient("k", "gemini-test", min_interval=0, session=FakeSession(responses), sleep=lambda s: None)


def test_gemini_parses_json_and_counts_tokens():
    c = client_with(gemini_ok(GOOD))
    r = c.generate_json("prompt", GEMINI_SCHEMA)
    assert r.data["seniority"] == "senior"
    assert (r.input_tokens, r.output_tokens) == (900, 70)
    url, body = c.session.calls[0]
    assert url.endswith("/models/gemini-test:generateContent")
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_retries_429():
    c = client_with(FakeResp(429), gemini_ok(GOOD))
    assert c.generate_json("p", GEMINI_SCHEMA).data["role_family"] == "data_scientist"
    assert len(c.session.calls) == 2


def test_gemini_404_explains_list_models():
    with pytest.raises(LLMError, match="list-models"):
        client_with(FakeResp(404)).generate_json("p", GEMINI_SCHEMA)


def test_gemini_blocked_response_is_llm_error():
    with pytest.raises(LLMError):
        client_with(FakeResp(200, {"candidates": [{"finishReason": "SAFETY"}]})).generate_json("p", GEMINI_SCHEMA)


# ---------- enrich loop ----------

class FakeRepo:
    def __init__(self):
        self.saved = []

    def save(self, job, ext, result, model, cost):
        self.saved.append((job["id"], ext, cost))


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def generate_json(self, prompt, schema):
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return LLMResult(o, 1000, 100, 50)


JOB = {"id": 1, "title": "Data Scientist", "company": "Acme", "location": "London",
       "description": "We need Python and SQL..." * 20}


def test_enrich_saves_and_tracks_cost():
    repo = FakeRepo()
    s = Settings(llm_input_usd_per_m=0.10, llm_output_usd_per_m=0.40)
    stats = enrich_jobs([JOB, {**JOB, "id": 2}], FakeClient([GOOD, GOOD]), repo, s)
    assert stats.ok == 2 and len(repo.saved) == 2
    assert stats.cost == pytest.approx(2 * (1000 * 0.10 + 100 * 0.40) / 1e6)


def test_enrich_skips_invalid_output_and_stops_after_repeated_errors():
    repo = FakeRepo()
    outcomes = [{**GOOD, "seniority": "wizard"}] + [LLMError("429")] * 10
    stats = enrich_jobs([{**JOB, "id": i} for i in range(11)], FakeClient(outcomes), repo, Settings())
    assert stats.ok == 0 and stats.failed == 5  # stopped early


# ---------- evaluation metrics ----------

def test_split_skills_handles_commas_and_semicolons():
    assert split_skills("Python; sql, Amazon Web Services") == ["AWS", "Python", "SQL"]


def test_score_precision_recall_and_accuracy():
    labels = [
        {"job_id": "1", "title": "DS", "skills": "Python; SQL; AWS", "seniority": "senior",
         "work_mode": "hybrid", "visa_sponsorship": "unknown", "role_family": "data_scientist",
         "experience_years_min": "3+"},
        {"job_id": "2", "title": "DA", "skills": "Excel; Power BI", "seniority": "junior",
         "work_mode": "", "visa_sponsorship": "no", "role_family": "data_analyst",
         "experience_years_min": "none"},
        {"job_id": "3", "title": "not enriched", "skills": "Python", "seniority": "mid"},
    ]
    preds = {
        1: {"skills": ["python", "SQL", "Docker"], "seniority": "senior", "work_mode": "remote",
            "visa_sponsorship": "unknown", "role_family": "data_scientist", "experience_years_min": 3},
        2: {"skills": ["Excel", "PowerBI"], "seniority": "mid", "work_mode": "onsite",
            "visa_sponsorship": "no", "role_family": "data_analyst", "experience_years_min": None},
    }
    r = score(labels, preds)
    assert r["n_jobs"] == 2
    # gold {python, sql, aws, excel, power bi}; pred {python, sql, docker, excel, power bi}
    assert (r["skills"]["tp"], r["skills"]["fp"], r["skills"]["fn"]) == (4, 1, 1)
    assert r["skills"]["precision"] == pytest.approx(0.8)
    assert r["skills"]["recall"] == pytest.approx(0.8)
    assert r["fields"]["seniority"] == {"accuracy": 0.5, "n": 2}
    assert r["fields"]["work_mode"] == {"accuracy": 0.0, "n": 1}  # blank label skipped
    assert r["fields"]["experience_years_min"]["accuracy"] == 1.0
    assert ("AWS", 1) in r["skills"]["top_missed"]


def test_interactive_label_saves_each_job_and_resumes():
    from jobsintel.evaluate import label_rows
    rows = [{"job_id": "1", "title": "DS"}, {"job_id": "2", "title": "DA"}]
    jobs = {1: {"title": "DS", "description": "..."}, 2: {"title": "DA", "description": "..."}}
    answers = iter(["Python; SQL", "3", "4", "2", "x", "1", "1",   # job 1 ("x" invalid -> re-asked)
                    "Excel", "q"])                                  # quit during job 2
    saves = []
    n = label_rows(rows, jobs, lambda r: saves.append([dict(x) for x in r]),
                   ask=lambda prompt: next(answers), show=lambda *a: None)
    assert n == 1 and len(saves) == 1
    assert rows[0] == {"job_id": "1", "title": "DS", "skills": "Python; SQL", "experience_years_min": "3",
                       "seniority": "senior", "work_mode": "hybrid", "visa_sponsorship": "yes",
                       "role_family": "data_scientist"}
    assert "seniority" not in rows[1]
    # resume: only job 2 is asked
    answers2 = iter(["", "", "7", "4", "3", "7"])
    n2 = label_rows(rows, jobs, lambda r: None, ask=lambda p: next(answers2), show=lambda *a: None)
    assert n2 == 1 and rows[1]["experience_years_min"] == "none" and rows[1]["role_family"] == "other"


def test_find_hints_pulls_relevant_sentences():
    from jobsintel.evaluate import find_hints
    h = find_hints("We are great. You need 3+ years of Python. This is a hybrid role, 2 days a week. "
                   "Applicants must have the right to work in the UK. Senior position.")
    assert any("3+ years" in l for l in h["EXPERIENCE"])
    assert any("hybrid" in l for l in h["WORK MODE"])
    assert any("right to work" in l for l in h["VISA"])
    assert any("Senior" in l for l in h["LEVEL"])
    assert find_hints("")["VISA"] == []


def test_label_rejects_number_as_skills():
    from jobsintel.evaluate import label_rows
    rows = [{"job_id": "1", "title": "DS"}]
    answers = iter(["0", "Python", "", "7", "4", "3", "7"])
    label_rows(rows, {1: {"description": ""}}, lambda r: None, ask=lambda p: next(answers), show=lambda *a: None)
    assert rows[0]["skills"] == "Python"
