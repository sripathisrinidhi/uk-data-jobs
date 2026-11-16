"""Measure how accurate the LLM extraction is against hand-made labels.

    python -m jobsintel.evaluate sample --n 50   # pick 50 jobs, write files to label
    (fill in eval/labels.csv by hand, reading eval/jobs_to_label.md)
    python -m jobsintel.evaluate score           # compare LLM output to your labels

Label first, look at the LLM output after. Otherwise you bias the test.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from jobsintel.extraction import (ROLE_FAMILY, SENIORITY, VISA, WORK_MODE,
                                  normalize_skills)

log = logging.getLogger("jobsintel.evaluate")

EVAL_DIR = Path("eval")
LABELS = EVAL_DIR / "labels.csv"
TO_LABEL = EVAL_DIR / "jobs_to_label.md"
REPORT = EVAL_DIR / "report.md"

CATEGORICAL = ["seniority", "work_mode", "visa_sponsorship", "role_family"]
FIELDS = ["job_id", "title", "company", "skills", "experience_years_min",
          *CATEGORICAL, "notes"]
ALLOWED = {"seniority": SENIORITY, "work_mode": WORK_MODE,
           "visa_sponsorship": VISA, "role_family": ROLE_FAMILY}


# ---------------- metrics (pure functions, unit-tested) ----------------

def split_skills(text: str) -> list[str]:
    parts = text.replace(",", ";").split(";")
    return normalize_skills([p for p in parts if p.strip()])


def _years(value: str) -> int | None | str:
    v = (value or "").strip().lower()
    if v == "":
        return "SKIP"
    if v in {"none", "null", "unknown", "-"}:
        return None
    m = re.search(r"\d+", v)  # "3", "3+", "3 years"
    return int(m.group()) if m else None


def score(labels: list[dict], preds: dict[int, dict]) -> dict:
    out: dict = {"n_jobs": 0, "skills": {}, "fields": {}, "errors": {f: [] for f in CATEGORICAL}}
    tp = fp = fn = 0
    missed, extra = Counter(), Counter()
    field_hits = {f: [0, 0] for f in [*CATEGORICAL, "experience_years_min"]}
    out["errors"]["experience_years_min"] = []

    for row in labels:
        jid = int(row["job_id"])
        pred = preds.get(jid)
        if pred is None:
            continue
        out["n_jobs"] += 1

        if (row.get("skills") or "").strip():
            gold = {s.lower(): s for s in split_skills(row["skills"])}
            got = {s.lower(): s for s in normalize_skills(pred.get("skills", []))}
            tp += len(gold.keys() & got.keys())
            fp += len(got.keys() - gold.keys())
            fn += len(gold.keys() - got.keys())
            missed.update(gold[k] for k in gold.keys() - got.keys())
            extra.update(got[k] for k in got.keys() - gold.keys())

        for f in CATEGORICAL:
            g = (row.get(f) or "").strip().lower()
            if not g:
                continue
            p = str(pred.get(f, "")).lower()
            field_hits[f][1] += 1
            if g == p:
                field_hits[f][0] += 1
            else:
                out["errors"][f].append((jid, row.get("title", ""), g, p))

        g_years = _years(row.get("experience_years_min", ""))
        if g_years != "SKIP":
            p_years = pred.get("experience_years_min")
            field_hits["experience_years_min"][1] += 1
            if g_years == p_years:
                field_hits["experience_years_min"][0] += 1
            else:
                out["errors"]["experience_years_min"].append(
                    (jid, row.get("title", ""), g_years, p_years))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out["skills"] = {"precision": precision, "recall": recall, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn,
                     "top_missed": missed.most_common(10), "top_extra": extra.most_common(10)}
    out["fields"] = {f: {"accuracy": (h[0] / h[1] if h[1] else None), "n": h[1]}
                     for f, h in field_hits.items()}
    return out


def render_report(res: dict, model: str) -> str:
    s = res["skills"]
    lines = [
        "# LLM extraction evaluation", "",
        f"Model: `{model}` · Jobs scored: **{res['n_jobs']}**", "",
        "## Skills (micro-averaged over all job-skill pairs)", "",
        "| Precision | Recall | F1 | TP | FP | FN |", "|---|---|---|---|---|---|",
        f"| {s['precision']:.1%} | {s['recall']:.1%} | {s['f1']:.1%} | {s['tp']} | {s['fp']} | {s['fn']} |", "",
        "- Most often **missed** (in label, not extracted): "
        + (", ".join(f"{k} ({v})" for k, v in s["top_missed"]) or "none"),
        "- Most often **extra** (extracted, not in label): "
        + (", ".join(f"{k} ({v})" for k, v in s["top_extra"]) or "none"), "",
        "## Other fields", "", "| Field | Accuracy | n |", "|---|---|---|",
    ]
    for f, v in res["fields"].items():
        acc = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.1%}"
        lines.append(f"| {f} | {acc} | {v['n']} |")
    lines += ["", "## Errors (for error analysis)", ""]
    for f, errs in res["errors"].items():
        if errs:
            lines.append(f"**{f}**")
            lines += [f"- job {jid} *{title}*: label `{g}`, LLM `{p}`" for jid, title, g, p in errs[:15]]
            lines.append("")
    return "\n".join(lines)


# ---------------- commands ----------------

def cmd_sample(conn, n: int, seed: int, force: bool) -> int:
    if LABELS.exists() and not force:
        log.error("%s already exists. Use --force to overwrite (you'd lose your labels!)", LABELS)
        return 2
    from psycopg.rows import dict_row
    with conn.cursor(row_factory=dict_row) as cur:
        # One job per company: some recruiters post the same advert hundreds of times,
        # which would make the eval set repetitive and unrepresentative.
        cur.execute("""
            SELECT * FROM (
                SELECT DISTINCT ON (lower(coalesce(company, '')))
                       id, title, company, location, url, description
                FROM jobs
                WHERE description_is_full AND length(description) >= 200
                ORDER BY lower(coalesce(company, '')), md5(id::text || %s)
            ) one_per_company
            ORDER BY md5(id::text || %s) LIMIT %s""", (str(seed), str(seed), n))
        jobs = cur.fetchall()
        if len(jobs) < n:
            log.warning("only %d distinct companies with full descriptions so far", len(jobs))
    EVAL_DIR.mkdir(exist_ok=True)
    with LABELS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for j in jobs:
            w.writerow({"job_id": j["id"], "title": j["title"], "company": j["company"]})
    md = ["# Jobs to label", "",
          "Fill one row per job in `eval/labels.csv`. Allowed values:", ""]
    md += [f"- **{k}**: {', '.join(v)}" for k, v in ALLOWED.items()]
    md += ["- **skills**: separated by `;` e.g. `Python; SQL; AWS`",
           "- **experience_years_min**: a number, or `none` if not stated", "", "---", ""]
    for i, j in enumerate(jobs, 1):
        md += [f"## {i}. job_id {j['id']}: {j['title']}",
               f"*{j['company']} · {j['location']}* · [link]({j['url']})", "", j["description"], "", "---", ""]
    TO_LABEL.write_text("\n".join(md))
    log.info("wrote %s (%d jobs) and %s", LABELS, len(jobs), TO_LABEL)
    return 0


def cmd_score(conn, settings) -> int:
    from jobsintel.enrich import EnrichmentRepo, enrich_jobs
    from jobsintel.llm import GeminiClient

    with LABELS.open() as f:
        labels = [r for r in csv.DictReader(f)
                  if any((r.get(k) or "").strip() for k in ["skills", *CATEGORICAL])]
    if not labels:
        log.error("no labelled rows in %s yet", LABELS)
        return 2
    bad = [(r["job_id"], k, r[k]) for r in labels for k, allowed in ALLOWED.items()
           if (r.get(k) or "").strip() and r[k].strip().lower() not in allowed]
    if bad:
        for jid, k, v in bad:
            log.error("job %s: '%s' is not a valid %s (allowed: %s)", jid, v, k, ", ".join(ALLOWED[k]))
        return 2

    ids = [int(r["job_id"]) for r in labels]
    repo = EnrichmentRepo(conn)
    preds = repo.enrichments(ids)
    missing = [i for i in ids if i not in preds]
    if missing:
        if not settings.gemini_api_key:
            log.error("%d labelled jobs not enriched yet and GEMINI_API_KEY not set", len(missing))
            return 2
        log.info("enriching %d labelled jobs first", len(missing))
        client = GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.gemini_min_interval)
        enrich_jobs(repo.jobs_by_ids(missing), client, repo, settings)
        preds = repo.enrichments(ids)

    res = score(labels, preds)
    report = render_report(res, settings.gemini_model)
    REPORT.write_text(report)
    print(report)
    log.info("saved %s", REPORT)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate LLM extraction against hand labels")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--force", action="store_true")
    sub.add_parser("score")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from jobsintel.config import Settings
    settings = Settings()
    if not settings.database_url:
        log.error("DATABASE_URL not set")
        return 2
    import psycopg
    conn = psycopg.connect(settings.database_url)
    if args.cmd == "sample":
        return cmd_sample(conn, args.n, args.seed, args.force)
    return cmd_score(conn, settings)


if __name__ == "__main__":
    sys.exit(main())
