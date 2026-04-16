"""
models.py — Pydantic data shapes used across the whole app.

Pydantic does two things here:
1. Validates incoming API request bodies automatically
2. Serializes outgoing responses to JSON

If someone sends pr_size="big" (str), Pydantic raises a 422 before your code runs.
Optional fields default to None so callers can send minimal payloads.
"""

from pydantic import BaseModel
from typing import Optional


class DeploymentEvent(BaseModel):
    """What the caller sends to /score-deploy"""
    repo_name: str
    pr_title: str
    author: str
    branch: str
    pr_size_lines_changed: int          # Total lines added + deleted
    files_changed: int
    new_test_files: int                 # How many test files were added/modified
    days_since_last_deploy: float       # 0.5 = 12 hours ago
    deploys_last_7_days: int            # Recent deploy cadence for this repo
    hour_of_day: int                    # 0-23, Friday afternoons are risky
    day_of_week: int                    # 0=Monday, 6=Sunday
    commit_sha: Optional[str] = None
    pr_number: Optional[int] = None


class RiskFactor(BaseModel):
    """One contributing reason to the overall risk score"""
    factor: str             # e.g. "Large PR size"
    impact: str             # "high" | "medium" | "low"
    detail: str             # Human-readable explanation


class DeploymentRiskResult(BaseModel):
    """What /score-deploy returns"""
    repo_name: str
    pr_title: str
    risk_score: float                       # 0.0 - 1.0 (raw model probability)
    risk_level: str                         # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    risk_factors: list[RiskFactor]          # Top reasons driving the score
    llm_advisory: str                       # Natural language advisory from LLM
    recommendation: str                     # "DEPLOY" | "DEPLOY_WITH_CAUTION" | "HOLD"
    dora_metrics_snapshot: dict             # Current DORA metrics for this repo
    scored_at: str                          # ISO 8601 timestamp


class DeployOutcome(BaseModel):
    """What the caller sends to /record-outcome after a deploy completes"""
    repo_name: str
    pr_title: str
    author: str
    branch: str
    pr_size_lines_changed: int
    files_changed: int
    new_test_files: int
    days_since_last_deploy: float
    deploys_last_7_days: int
    hour_of_day: int
    day_of_week: int
    failed: bool                # True = deploy caused an incident/rollback
    commit_sha: Optional[str] = None


class DORAMetrics(BaseModel):
    """DORA 4 key metrics for a given repo"""
    repo_name: str
    deployment_frequency_per_week: float    # Elite: >1/day, High: 1/week, etc.
    lead_time_days: float                   # Time from commit to production
    change_failure_rate_pct: float          # % of deploys that caused an incident
    mttr_hours: float                       # Mean time to restore after failure
    dora_band: str                          # "Elite" | "High" | "Medium" | "Low"
    total_deploys_recorded: int
