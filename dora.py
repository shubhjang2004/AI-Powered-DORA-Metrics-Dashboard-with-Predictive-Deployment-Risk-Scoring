"""
dora.py — DORA (DevOps Research and Assessment) metrics computation.

The 4 DORA metrics are the industry standard for measuring software delivery performance.
We compute them from the deploy_outcomes table in SQLite.

Metrics:
  1. Deployment Frequency  — how often do you deploy?
  2. Lead Time             — how long from commit to production?
                             (we approximate from days_since_last_deploy as a proxy)
  3. Change Failure Rate   — what % of deploys caused an incident?
  4. MTTR                  — mean time to restore (estimated from failure patterns)

DORA Performance Bands (from the 2023 DORA State of DevOps Report):
  Elite:  Deploy on-demand (multiple/day), lead time <1hr, CFR <5%,  MTTR <1hr
  High:   Deploy 1/day-1/week, lead time <1day, CFR 5-10%, MTTR <1day
  Medium: Deploy 1/week-1/month, lead time 1-7days, CFR 10-15%, MTTR <1week
  Low:    Deploy <1/month, lead time >6months, CFR >15%, MTTR >1week

Note on lead time:
  We don't have commit timestamps, so we estimate lead time from deploy cadence.
  In a real system, you'd ingest GitHub webhook events (push → deploy timestamps).
"""

from models import DORAMetrics


def compute_dora_metrics(repo_name: str, outcomes: list[dict]) -> DORAMetrics:
    """
    Compute DORA metrics for a single repo from its deploy history.

    outcomes: list of dicts from db.get_outcomes_for_repo()
    """
    total = len(outcomes)

    if total == 0:
        return DORAMetrics(
            repo_name=repo_name,
            deployment_frequency_per_week=0.0,
            lead_time_days=0.0,
            change_failure_rate_pct=0.0,
            mttr_hours=0.0,
            dora_band="Unknown",
            total_deploys_recorded=0,
        )

    # ── 1. Deployment Frequency ──────────────────────────────────────────────
    # Count deploys over the span of recorded history
    # We use deploys_last_7_days average as a proxy for current cadence
    avg_deploys_7d = sum(r["deploys_last_7_days"] for r in outcomes) / total
    deploy_freq_per_week = avg_deploys_7d  # Already a weekly number

    # ── 2. Lead Time ─────────────────────────────────────────────────────────
    # Proxy: average days_since_last_deploy tells us how long code sits before deploy
    # Lower = faster delivery pipeline
    avg_lead_time = sum(r["days_since_last_deploy"] for r in outcomes) / total

    # ── 3. Change Failure Rate ───────────────────────────────────────────────
    failures = sum(1 for r in outcomes if r["failed"] == 1)
    cfr_pct = (failures / total) * 100

    # ── 4. MTTR (estimated) ──────────────────────────────────────────────────
    # We estimate: if failures cluster on large PRs or stale deploys,
    # MTTR is proportional to PR size (more code = longer to diagnose).
    # In production, you'd record actual incident resolution timestamps.
    failed_outcomes = [r for r in outcomes if r["failed"] == 1]
    if failed_outcomes:
        avg_lines_on_failure = sum(r["pr_size_lines_changed"] for r in failed_outcomes) / len(failed_outcomes)
        # Heuristic: every 200 lines of change = ~1 hour to diagnose
        estimated_mttr = avg_lines_on_failure / 200.0
        estimated_mttr = max(0.5, min(estimated_mttr, 72.0))  # Cap at 72hrs
    else:
        estimated_mttr = 0.0

    # ── 5. DORA Band Classification ───────────────────────────────────────────
    band = _classify_dora_band(deploy_freq_per_week, avg_lead_time, cfr_pct, estimated_mttr)

    return DORAMetrics(
        repo_name=repo_name,
        deployment_frequency_per_week=round(deploy_freq_per_week, 2),
        lead_time_days=round(avg_lead_time, 2),
        change_failure_rate_pct=round(cfr_pct, 1),
        mttr_hours=round(estimated_mttr, 1),
        dora_band=band,
        total_deploys_recorded=total,
    )


def _classify_dora_band(
    freq: float,
    lead_time: float,
    cfr: float,
    mttr: float
) -> str:
    """
    Classify into DORA performance band based on the 4 metrics.

    We score each metric separately and take the worst (weakest link).
    A team deploying 10x/day but with 30% failure rate is not "Elite".
    """
    scores = []

    # Deployment frequency bands
    if freq >= 7:       scores.append(4)   # Elite: ≥1/day
    elif freq >= 1:     scores.append(3)   # High: ≥1/week
    elif freq >= 0.25:  scores.append(2)   # Medium: ≥1/month
    else:               scores.append(1)   # Low

    # Lead time bands (in days)
    if lead_time <= 0.042:   scores.append(4)   # Elite: <1hr
    elif lead_time <= 1:     scores.append(3)   # High: <1day
    elif lead_time <= 7:     scores.append(2)   # Medium: <1week
    else:                    scores.append(1)   # Low

    # Change failure rate bands
    if cfr <= 5:    scores.append(4)   # Elite: <5%
    elif cfr <= 10: scores.append(3)   # High: <10%
    elif cfr <= 15: scores.append(2)   # Medium: <15%
    else:           scores.append(1)   # Low

    # MTTR bands (in hours)
    if mttr <= 1:    scores.append(4)   # Elite: <1hr
    elif mttr <= 24: scores.append(3)   # High: <1day
    elif mttr <= 168: scores.append(2)  # Medium: <1week
    else:            scores.append(1)   # Low

    # Weakest link determines the overall band
    worst = min(scores)
    return {4: "Elite", 3: "High", 2: "Medium", 1: "Low"}[worst]


def compute_all_repo_metrics(all_outcomes: list[dict]) -> dict[str, DORAMetrics]:
    """
    Compute DORA metrics for every repo in the dataset.
    Groups outcomes by repo_name then computes per-repo.
    """
    by_repo: dict[str, list[dict]] = {}
    for outcome in all_outcomes:
        repo = outcome["repo_name"]
        by_repo.setdefault(repo, []).append(outcome)

    return {
        repo: compute_dora_metrics(repo, outcomes)
        for repo, outcomes in by_repo.items()
    }
