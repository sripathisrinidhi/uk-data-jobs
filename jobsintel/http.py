"""Polite HTTP helper: call budget, rate-limit spacing, retries with backoff."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    """Hard cap on API calls per run so a bug can't burn the monthly quota."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(f"call budget of {self.limit} reached")
        self.used += 1


class ApiClient:
    def __init__(
        self,
        budget: CallBudget,
        min_interval: float = 0.0,
        session: requests.Session | None = None,
        max_retries: int = 4,
        sleep=time.sleep,
    ):
        self.budget = budget
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self._sleep = sleep
        self._last_call = 0.0

    def _wait_for_slot(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if self.min_interval and elapsed < self.min_interval:
            self._sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def get_json(self, url: str, params: dict | None = None, auth: Any = None) -> dict:
        for attempt in range(self.max_retries + 1):
            self.budget.spend()  # retries count too: they hit the quota
            self._wait_for_slot()
            resp = self.session.get(url, params=params, auth=auth, timeout=30)
            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
                log.warning("HTTP %s from %s, retrying in %.0fs", resp.status_code, url, delay)
                self._sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")
