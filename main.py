import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from models import DeploymentEvent, DeploymentRiskResult, DeployOutcome, DORAMetrics
from scorer import score_deployment, score_to_level, score_to_recommendation
from advisor import generate_advisory
from dora import compute_dora_metrics, compute_all_repo_metrics
from db import init_db, record_outcome, get_all_outcomes, get_outcomes_for_repo, count_outcomes, get_all_repos

# Initialize SQLite tables on startup
init_db()

app = FastAPI(
    title="DORA Deployment Risk Scorer",
    description=(
        "AI-powered pre-deployment risk scoring using DORA metrics + XGBoost. "
        "Predicts whether a deploy is likely to cause an incident BEFORE it happens. "
        "Endpoints: /score-deploy (risk score), /record-outcome (feedback loop), "
        "/dora-metrics (DORA dashboard), /dora-metrics/{repo} (per-repo metrics)."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Quick check that the service is up and shows how many outcomes are recorded."""
    total = count_outcomes()
    repos = get_all_repos()
    return {
        "status": "ok",
        "outcomes_recorded": total,
        "repos_tracked": repos,
        "ml_model_active": total >= 20,
        "tip": (
            "Run seed.py to load demo deploy history. "
            "ML model activates after 20+ outcomes are recorded."
        )
    }


@app.post("/score-deploy", response_model=DeploymentRiskResult)
def score_deploy(event: DeploymentEvent):
    """
    Score a deployment's failure risk BEFORE it happens.

    Send PR metadata. The system will:
    1. Extract engineered features (test coverage ratio, friday-afternoon flag, etc.)
    2. Score via XGBoost (if 20+ outcomes recorded) or rule-based heuristics
    3. Compute current DORA metrics for this repo from historical outcomes
    4. Ask LLM to translate the score into a natural-language advisory

    The response includes risk_factors so you can see WHAT is driving the score.
    """
    try:
        # Load all outcomes for ML training + per-repo outcomes for DORA metrics
        all_outcomes = get_all_outcomes()
        repo_outcomes = get_outcomes_for_repo(event.repo_name)

        # Score the deployment
        risk_score, risk_factors, method = score_deployment(event, all_outcomes)
        risk_level = score_to_level(risk_score)
        recommendation = score_to_recommendation(risk_score)

        # Compute DORA metrics for this specific repo
        dora = compute_dora_metrics(event.repo_name, repo_outcomes)

        # LLM advisory — translates numbers into natural language
        advisory = generate_advisory(
            event, risk_score, risk_level, risk_factors, dora, method
        )

        return DeploymentRiskResult(
            repo_name=event.repo_name,
            pr_title=event.pr_title,
            risk_score=round(risk_score, 3),
            risk_level=risk_level,
            risk_factors=risk_factors,
            llm_advisory=advisory,
            recommendation=recommendation,
            dora_metrics_snapshot=dora.model_dump(),
            scored_at=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/record-outcome", status_code=201)
def record_deploy_outcome(outcome: DeployOutcome):
    """
    Record the actual result of a deployment (success or failure).

    Call this after a deploy completes — whether it succeeded or caused an incident.
    This is the feedback loop that makes the ML model smarter over time.

    Example:
        POST /record-outcome
        {
            "repo_name": "api-service",
            "pr_title": "feat: add caching layer",
            "author": "alice",
            "branch": "feature/cache",
            "pr_size_lines_changed": 340,
            "files_changed": 8,
            "new_test_files": 2,
            "days_since_last_deploy": 3.5,
            "deploys_last_7_days": 4,
            "hour_of_day": 14,
            "day_of_week": 2,
            "failed": false
        }
    """
    row_id = record_outcome({
        "repo_name": outcome.repo_name,
        "pr_title": outcome.pr_title,
        "author": outcome.author,
        "branch": outcome.branch,
        "pr_size_lines_changed": outcome.pr_size_lines_changed,
        "files_changed": outcome.files_changed,
        "new_test_files": outcome.new_test_files,
        "days_since_last_deploy": outcome.days_since_last_deploy,
        "deploys_last_7_days": outcome.deploys_last_7_days,
        "hour_of_day": outcome.hour_of_day,
        "day_of_week": outcome.day_of_week,
        "failed": int(outcome.failed),
        "commit_sha": outcome.commit_sha,
    })

    total = count_outcomes()
    return {
        "message": "Outcome recorded. Model will use this on next /score-deploy call.",
        "id": row_id,
        "total_outcomes": total,
        "ml_model_status": (
            "active" if total >= 20
            else f"heuristic mode — need {20 - total} more outcomes to activate ML model"
        )
    }


@app.get("/dora-metrics", response_model=dict)
def all_dora_metrics():
    """
    Get DORA metrics for all tracked repos.

    Returns a dict keyed by repo_name with DORA metrics for each.
    Useful for a dashboard showing which teams are Elite vs Low performers.
    """
    all_outcomes = get_all_outcomes()
    if not all_outcomes:
        return {
            "message": "No outcomes recorded yet. Run seed.py or POST to /record-outcome.",
            "repos": {}
        }

    metrics = compute_all_repo_metrics(all_outcomes)
    return {
        "repos": {repo: m.model_dump() for repo, m in metrics.items()},
        "total_deploys_across_all_repos": len(all_outcomes)
    }


@app.get("/dora-metrics/{repo_name}", response_model=DORAMetrics)
def repo_dora_metrics(repo_name: str):
    """
    Get DORA metrics for a single repo.

    Returns Deployment Frequency, Lead Time, Change Failure Rate, MTTR,
    and the DORA performance band (Elite / High / Medium / Low).
    """
    outcomes = get_outcomes_for_repo(repo_name)
    if not outcomes:
        raise HTTPException(
            status_code=404,
            detail=f"No outcomes recorded for repo '{repo_name}'. "
                   f"POST to /record-outcome first."
        )
    return compute_dora_metrics(repo_name, outcomes)
