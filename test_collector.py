from datetime import datetime

import pytest

from jobsintel.collect import SourceStats, enrich_reed, gather
from jobsintel.db import JsonlStore
from jobsintel.http import ApiClient, BudgetExceeded, CallBudget
from jobsintel.models import clean_text, infer_salary_period
from jobsintel.sources.adzuna import AdzunaSource, parse_adzuna
from jobsintel.sources.reed import ReedSource, parse_reed

ADZUNA_ITEM = {
    "id": "4912345678",
    "title": "Senior <strong>Data Scientist</strong>",
    "description": "Python, SQL and AWS &amp; LLMs. Visa sponsorship…",
    "created": "2026-11-14T09:12:00Z",
    "redirect_url": "https://www.adzuna.co.uk/jobs/land/ad/4912345678",
    "company": {"display_name": "Acme Analytics"},
    "location": {"display_name": "London, UK", "area": ["UK", "London"]},
    "salary_min": 70000, "salary_max": 85000, "salary_is_predicted": "1",
    "contract_type": "permanent", "contract_time": "full_time",
    "category": {"label": "IT Jobs"},
}

REED_ITEM = {
    "jobId": 55501234, "employerName": "Beta Bank", "jobTitle": "Data Analyst",
    "locationName": "Manchester", "minimumSalary": 450.0, "maximumSalary": 550.0,
    "date": "13/11/2026", "jobDescription": "Snippet only...",
    "jobUrl": "https://www.reed.co.uk/jobs/data-analyst/55501234",
}


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def get(self, url, params=None, auth=None, timeout=None):
        self.calls.append((url, params))
        return self.responder(url, params)


def client(responder, budget=50):
    return ApiClient(CallBudget(budget), session=FakeSession(responder), sleep=lambda s: None)


# ---------- parsing ----------

def test_parse_adzuna_flags_predicted_salary_and_snippet():
    p = parse_adzuna(ADZUNA_ITEM, "data scientist")
    assert p.title == "Senior Data Scientist"
    assert p.description == "Python, SQL and AWS & LLMs. Visa sponsorship…"
    assert p.salary_is_predicted is True
    assert p.description_is_full is False
    assert p.salary_period == "annual"
    assert p.location_area == ["UK", "London"]
    assert p.posted_at.year == 2026


def test_parse_reed_day_rate_detected():
    p = parse_reed(REED_ITEM, "data analyst")
    assert p.source_id == "55501234"
    assert p.salary_period == "daily"
    assert p.posted_at == datetime(2026, 11, 13)


@pytest.mark.parametrize("lo,hi,period", [
    (None, None, "unknown"), (15, 20, "hourly"), (400, 600, "daily"), (45000, 60000, "annual")])
def test_salary_period(lo, hi, period):
    assert infer_salary_period(lo, hi) == period


def test_clean_text_strips_html():
    assert clean_text("<p>Hello&nbsp;<b>world</b></p>\n\n") == "Hello world"


# ---------- pagination, budget, retries ----------

def test_adzuna_stops_on_short_page():
    def responder(url, params):
        page = int(url.rsplit("/", 1)[1])
        n = 50 if page == 1 else 7
        return FakeResp(200, {"count": 57, "results": [
            {**ADZUNA_ITEM, "id": f"{page}-{i}"} for i in range(n)]})
    c = client(responder)
    jobs = list(AdzunaSource("id", "key", c).search("data scientist", max_pages=5))
    assert len(jobs) == 57
    assert c.budget.used == 2  # didn't request page 3


def test_retry_on_429_then_success():
    state = {"n": 0}
    def responder(url, params):
        state["n"] += 1
        return FakeResp(429, headers={"Retry-After": "1"}) if state["n"] == 1 else FakeResp(200, {"ok": 1})
    c = client(responder)
    assert c.get_json("https://x") == {"ok": 1}
    assert c.budget.used == 2


def test_budget_is_hard_cap():
    c = client(lambda u, p: FakeResp(200, {}), budget=1)
    c.get_json("https://x")
    with pytest.raises(BudgetExceeded):
        c.get_json("https://x")


# ---------- collector ----------

def test_gather_merges_same_job_found_by_two_terms():
    def responder(url, params):
        return FakeResp(200, {"results": [ADZUNA_ITEM]})
    src = AdzunaSource("id", "key", client(responder))
    stats = SourceStats("adzuna")
    jobs = gather(src, ["data scientist", "machine learning engineer"], 1, stats)
    assert len(jobs) == 1
    assert jobs[0].search_terms == ["data scientist", "machine learning engineer"]


def test_gather_stops_cleanly_when_budget_runs_out():
    src = AdzunaSource("id", "key", client(lambda u, p: FakeResp(200, {"results": [ADZUNA_ITEM]}), budget=1))
    stats = SourceStats("adzuna")
    jobs = gather(src, ["a", "b", "c"], 1, stats)
    assert len(jobs) == 1 and "budget" in stats.errors[0]


def test_reed_enrich_fetches_full_description(tmp_path):
    def responder(url, params):
        if "/jobs/" in url:
            return FakeResp(200, {"jobId": 55501234, "jobDescription": "<p>Full text: Python, dbt, Snowflake</p>"})
        return FakeResp(200, {"results": [REED_ITEM], "totalResults": 1})
    reed = ReedSource("k", client(responder))
    stats = SourceStats("reed")
    jobs = gather(reed, ["data analyst"], 1, stats)
    jobs = enrich_reed(reed, JsonlStore(tmp_path), jobs, 10, stats)
    assert jobs[0].description_is_full
    assert jobs[0].description == "Full text: Python, dbt, Snowflake"
    assert jobs[0].company == "Beta Bank"  # search fields kept


def test_jsonl_store_counts_new_vs_seen(tmp_path):
    store = JsonlStore(tmp_path)
    p = parse_adzuna(ADZUNA_ITEM, "data scientist")
    assert store.upsert([p]) == (1, 0)
    assert store.upsert([p]) == (0, 1)
    assert store.path.read_text().count("\n") == 2
