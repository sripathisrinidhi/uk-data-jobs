"""What we extract from each job description, and how.

The Pydantic model is the contract: the LLM is forced to return JSON matching
GEMINI_SCHEMA, then we validate it with JobExtraction and normalise skills so
"Amazon Web Services", "aws" and "AWS" all count as one skill.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PROMPT_VERSION = "v1"
MAX_DESCRIPTION_CHARS = 6000

Seniority = Literal["intern", "junior", "mid", "senior", "lead", "manager", "unknown"]
WorkMode = Literal["remote", "hybrid", "onsite", "unknown"]
Visa = Literal["yes", "no", "unknown"]
RoleFamily = Literal["data_scientist", "data_analyst", "ml_engineer", "data_engineer",
                     "ai_engineer", "analytics_engineer", "other"]

SENIORITY = list(Seniority.__args__)
WORK_MODE = list(WorkMode.__args__)
VISA = list(Visa.__args__)
ROLE_FAMILY = list(RoleFamily.__args__)

# Canonical names. Keys are lower-cased aliases.
SKILL_ALIASES = {
    "python": "Python", "python3": "Python", "py": "Python", "sql": "SQL",
    "structured query language": "SQL", "t-sql": "SQL", "tsql": "SQL", "pl/sql": "SQL",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "mysql": "MySQL",
    "amazon web services": "AWS", "aws": "AWS",
    "google cloud": "GCP", "google cloud platform": "GCP", "gcp": "GCP",
    "microsoft azure": "Azure", "azure": "Azure",
    "power bi": "Power BI", "powerbi": "Power BI", "microsoft power bi": "Power BI",
    "tableau": "Tableau", "looker": "Looker",
    "excel": "Excel", "microsoft excel": "Excel", "ms excel": "Excel",
    "scikit learn": "scikit-learn", "sklearn": "scikit-learn", "scikit-learn": "scikit-learn",
    "pytorch": "PyTorch", "torch": "PyTorch", "tensorflow": "TensorFlow", "keras": "Keras",
    "pandas": "pandas", "numpy": "NumPy", "spark": "Spark", "apache spark": "Spark", "pyspark": "Spark",
    "databricks": "Databricks", "snowflake": "Snowflake", "dbt": "dbt", "airflow": "Airflow",
    "apache airflow": "Airflow", "kafka": "Kafka", "apache kafka": "Kafka",
    "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes", "git": "Git",
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "nlp": "NLP", "natural language processing": "NLP",
    "computer vision": "Computer Vision", "llm": "LLMs", "llms": "LLMs",
    "large language models": "LLMs", "generative ai": "Generative AI", "genai": "Generative AI",
    "rag": "RAG", "retrieval augmented generation": "RAG",
    "statistics": "Statistics", "a/b testing": "A/B Testing", "ab testing": "A/B Testing",
    "r": "R", "scala": "Scala", "java": "Java", "mlflow": "MLflow", "bigquery": "BigQuery",
    "google bigquery": "BigQuery", "redshift": "Redshift", "amazon redshift": "Redshift",
    "sagemaker": "SageMaker", "aws sagemaker": "SageMaker", "ci/cd": "CI/CD",
}


def normalize_skill(skill: str) -> str:
    s = re.sub(r"\s+", " ", skill).strip().strip(".,;:")
    return SKILL_ALIASES.get(s.lower(), s)


def normalize_skills(skills: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for raw in skills:
        if not raw or not raw.strip():
            continue
        canon = normalize_skill(raw)
        seen.setdefault(canon.lower(), canon)
    return sorted(seen.values(), key=str.lower)


class JobExtraction(BaseModel):
    skills: list[str] = Field(default_factory=list)
    experience_years_min: int | None = None
    seniority: Seniority = "unknown"
    work_mode: WorkMode = "unknown"
    visa_sponsorship: Visa = "unknown"
    role_family: RoleFamily = "other"

    @field_validator("skills")
    @classmethod
    def _norm(cls, v: list[str]) -> list[str]:
        return normalize_skills(v)

    @field_validator("experience_years_min")
    @classmethod
    def _sane_years(cls, v: int | None) -> int | None:
        return v if v is None or 0 <= v <= 30 else None


# Gemini responseSchema (OpenAPI subset). Mirrors JobExtraction.
GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "experience_years_min": {"type": "INTEGER", "nullable": True},
        "seniority": {"type": "STRING", "enum": SENIORITY},
        "work_mode": {"type": "STRING", "enum": WORK_MODE},
        "visa_sponsorship": {"type": "STRING", "enum": VISA},
        "role_family": {"type": "STRING", "enum": ROLE_FAMILY},
    },
    "required": ["skills", "experience_years_min", "seniority", "work_mode",
                 "visa_sponsorship", "role_family"],
}

PROMPT_TEMPLATE = """You extract structured facts from UK job adverts for data and ML roles.
Only use what the advert says. If something is not stated, use "unknown" (or null for years).

Fields:
- skills: concrete technical skills, tools, languages, platforms and methods the candidate
  should have (e.g. Python, SQL, AWS, Power BI, PyTorch, dbt, Statistics, Machine Learning).
  Use common short names. Exclude soft skills (communication, teamwork), degrees,
  and generic words like "data" or "analytics".
- experience_years_min: the minimum years of experience required, as an integer
  ("3+ years" -> 3, "2-4 years" -> 2). null if not stated.
- seniority: from the title and text. "Graduate"/"entry level" -> junior,
  "Senior" -> senior, "Lead"/"Principal"/"Staff" -> lead, "Head of"/"Manager" -> manager.
  No signal -> unknown.
- work_mode: remote, hybrid (any mix of office and home), onsite, or unknown.
- visa_sponsorship: "yes" only if sponsorship is offered or available; "no" if the advert
  says no sponsorship / must already have the right to work in the UK; otherwise unknown.
- role_family: closest of {role_families}.

Job title: {title}
Company: {company}
Location: {location}

Advert:
\"\"\"
{description}
\"\"\"
"""


def build_prompt(title: str, company: str | None, location: str | None, description: str) -> str:
    return PROMPT_TEMPLATE.format(
        role_families=", ".join(ROLE_FAMILY),
        title=title, company=company or "unknown", location=location or "unknown",
        description=description[:MAX_DESCRIPTION_CHARS],
    )
