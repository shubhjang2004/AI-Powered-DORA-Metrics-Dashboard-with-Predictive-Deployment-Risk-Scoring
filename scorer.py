"""
scorer.py — The ML risk scoring layer.

What this does:
- Trains an XGBoost binary classifier on recorded deploy outcomes
  (features = PR metadata, label = failed/success)
- At score time, takes a DeploymentEvent and returns a risk probability 0-1
- Also explains which features drove the score (risk factors)

Why XGBoost?
- Handles small tabular datasets well (unlike neural nets which need thousands of rows)
- Built-in feature importance — we can tell the user WHY the score is high
- Fast inference, no GPU needed
- Works out of the box when we only have 50-100 historical deploys

Fallback behavior:
- If fewer than MIN_SAMPLES outcomes are recorded, we can't train a meaningful model
- In that case we fall back to a rule-based heuristic scorer
  (large PR + Friday afternoon + no tests = high risk)
- This means the system is useful from day 1 even with zero history

The model is retrained on every /score-deploy call if new data was added.
We keep it simple — no MLflow, no model registry. This is a demo, not production.
"""

import os
import numpy as np
from typing import Optional
from models import DeploymentEvent, RiskFactor

# We import XGBoost at call time so the app still starts even if xgboost is missing
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Minimum outcomes needed before we trust the ML model over heuristics
MIN_SAMPLES = 20

# Feature names — must stay in sync with _extract_features()
FEATURE_NAMES = [
    "pr_size_lines_changed",
    "files_changed",
    "new_test_files",
    "days_since_last_deploy",
    "deploys_last_7_days",
    "hour_of_day",
    "day_of_week",
    "test_coverage_ratio",       # derived: new_test_files / max(files_changed, 1)
    "is_friday_afternoon",        # derived: day_of_week==4 and hour_of_day>=14
    "is_large_pr",                # derived: lines_changed > 500
    "is_stale_deploy",            # derived: days_since_last_deploy > 7
    "is_low_cadence",             # derived: deploys_last_7_days < 2
]


def _extract_features(event: DeploymentEvent) -> np.ndarray:
    """
    Convert a DeploymentEvent into a fixed-length feature vector.

    Derived features capture domain knowledge that raw numbers miss:
    - A PR with 500 changed lines and 0 new test files is riskier than
      500 changed lines with 10 new test files
    - Friday 4pm deploys are culturally understood as high-risk
    - Not deploying for 7+ days means more accumulated change risk
    """
    test_coverage_ratio = event.new_test_files / max(event.files_changed, 1)
    is_friday_afternoon = int(event.day_of_week == 4 and event.hour_of_day >= 14)
    is_large_pr = int(event.pr_size_lines_changed > 500)
    is_stale_deploy = int(event.days_since_last_deploy > 7)
    is_low_cadence = int(event.deploys_last_7_days < 2)

    return np.array([[
        event.pr_size_lines_changed,
        event.files_changed,
        event.new_test_files,
        event.days_since_last_deploy,
        event.deploys_last_7_days,
        event.hour_of_day,
        event.day_of_week,
        test_coverage_ratio,
        is_friday_afternoon,
        is_large_pr,
        is_stale_deploy,
        is_low_cadence,
    ]], dtype=np.float32)


def _heuristic_score(event: DeploymentEvent) -> tuple[float, list[RiskFactor]]:
    """
    Rule-based fallback scorer used when we don't have enough ML training data.

    Each rule adds to a raw score. We normalize to 0-1 at the end.
    Rules are ordered by typical impact on deployment failure rate.
    """
    raw = 0.0
    factors: list[RiskFactor] = []

    # Large PRs are harder to review and more likely to introduce bugs
    if event.pr_size_lines_changed > 1000:
        raw += 0.35
        factors.append(RiskFactor(
            factor="Very large PR",
            impact="high",
            detail=f"{event.pr_size_lines_changed} lines changed — hard to review thoroughly"
        ))
    elif event.pr_size_lines_changed > 500:
        raw += 0.2
        factors.append(RiskFactor(
            factor="Large PR",
            impact="medium",
            detail=f"{event.pr_size_lines_changed} lines changed — above safe threshold of 500"
        ))

    # No new tests despite code changes is a strong failure predictor
    if event.new_test_files == 0 and event.files_changed > 3:
        raw += 0.25
        factors.append(RiskFactor(
            factor="No new tests added",
            impact="high",
            detail=f"{event.files_changed} files changed but 0 test files added/modified"
        ))

    # Friday afternoon deploys are notoriously risky — no one wants to debug on Friday night
    if event.day_of_week == 4 and event.hour_of_day >= 14:
        raw += 0.2
        factors.append(RiskFactor(
            factor="Friday afternoon deploy",
            impact="high",
            detail="Deploying on Friday afternoon leaves no time to fix issues before weekend"
        ))

    # Long gaps mean more accumulated change, less deployment muscle memory
    if event.days_since_last_deploy > 14:
        raw += 0.2
        factors.append(RiskFactor(
            factor="Long gap since last deploy",
            impact="medium",
            detail=f"{event.days_since_last_deploy:.1f} days since last deploy — more accumulated risk"
        ))
    elif event.days_since_last_deploy > 7:
        raw += 0.1
        factors.append(RiskFactor(
            factor="Extended gap since last deploy",
            impact="low",
            detail=f"{event.days_since_last_deploy:.1f} days since last deploy"
        ))

    # Low deploy cadence means the team is less practiced at releases
    if event.deploys_last_7_days == 0:
        raw += 0.15
        factors.append(RiskFactor(
            factor="No recent deploys",
            impact="medium",
            detail="Zero deploys in the last 7 days — team may be out of practice"
        ))

    # Many files changed increases blast radius
    if event.files_changed > 20:
        raw += 0.1
        factors.append(RiskFactor(
            factor="Many files changed",
            impact="medium",
            detail=f"{event.files_changed} files modified — large blast radius"
        ))

    # Normalize to 0-1
    score = min(raw, 1.0)

    # If nothing was flagged, give a base low score
    if not factors:
        score = 0.1
        factors.append(RiskFactor(
            factor="No significant risk signals",
            impact="low",
            detail="PR size, timing, and test coverage all look healthy"
        ))

    return score, factors


def _ml_score(
    event: DeploymentEvent,
    model: "xgb.XGBClassifier"
) -> tuple[float, list[RiskFactor]]:
    """
    Score using the trained XGBoost model.
    Returns probability of failure + top feature importance factors.
    """
    features = _extract_features(event)
    prob = float(model.predict_proba(features)[0][1])  # P(failure)

    # Extract top 3 features by importance for this prediction
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:4]

    factors: list[RiskFactor] = []
    feature_values = features[0]

    for idx in top_indices:
        val = feature_values[idx]
        name = FEATURE_NAMES[idx]
        importance = importances[idx]

        if importance < 0.05:  # Skip negligible contributors
            continue

        impact = "high" if importance > 0.25 else ("medium" if importance > 0.1 else "low")

        # Map feature names to human-readable explanations
        detail = _feature_to_detail(name, val, event)
        if detail:
            factors.append(RiskFactor(factor=_feature_to_label(name), impact=impact, detail=detail))

    if not factors:
        factors.append(RiskFactor(
            factor="Model assessment",
            impact="low",
            detail="No dominant risk signals detected by the model"
        ))

    return prob, factors


def _feature_to_label(name: str) -> str:
    labels = {
        "pr_size_lines_changed": "PR size",
        "files_changed": "Files changed",
        "new_test_files": "Test coverage",
        "days_since_last_deploy": "Deploy gap",
        "deploys_last_7_days": "Deploy cadence",
        "hour_of_day": "Deploy time",
        "day_of_week": "Deploy day",
        "test_coverage_ratio": "Test-to-change ratio",
        "is_friday_afternoon": "Friday afternoon deploy",
        "is_large_pr": "Large PR",
        "is_stale_deploy": "Stale deploy",
        "is_low_cadence": "Low deploy cadence",
    }
    return labels.get(name, name)


def _feature_to_detail(name: str, val: float, event: DeploymentEvent) -> Optional[str]:
    if name == "pr_size_lines_changed":
        return f"{int(val)} lines changed in this PR"
    if name == "files_changed":
        return f"{int(val)} files modified — wide blast radius"
    if name == "new_test_files":
        return f"Only {int(val)} test files added for {event.files_changed} changed files"
    if name == "days_since_last_deploy":
        return f"{val:.1f} days since last deploy to this repo"
    if name == "deploys_last_7_days":
        return f"Only {int(val)} deploys in the last 7 days — low cadence"
    if name == "is_friday_afternoon" and val == 1:
        return "Deploying Friday afternoon leaves no recovery window before weekend"
    if name == "is_large_pr" and val == 1:
        return "PR exceeds 500 lines — historically correlates with higher failure rate"
    if name == "is_stale_deploy" and val == 1:
        return f"No deploy for {event.days_since_last_deploy:.0f}+ days — accumulated change risk"
    if name == "test_coverage_ratio":
        ratio = event.new_test_files / max(event.files_changed, 1)
        return f"Test-to-file ratio: {ratio:.2f} ({event.new_test_files} tests / {event.files_changed} files)"
    return None


def train_model(outcomes: list[dict]) -> Optional["xgb.XGBClassifier"]:
    """
    Train XGBoost on recorded deploy outcomes.
    Returns None if not enough data or XGBoost not installed.

    We rebuild the model from scratch every time rather than incrementally updating.
    This is fine for hundreds of rows. At thousands of rows, switch to warm_start=True.
    """
    if not XGB_AVAILABLE:
        return None

    if len(outcomes) < MIN_SAMPLES:
        return None

    # Build feature matrix and label vector from stored outcomes
    X_rows = []
    y = []

    for row in outcomes:
        # Reconstruct a DeploymentEvent-like object from the DB row
        event = DeploymentEvent(
            repo_name=row["repo_name"],
            pr_title=row["pr_title"],
            author=row["author"],
            branch=row["branch"],
            pr_size_lines_changed=row["pr_size_lines_changed"],
            files_changed=row["files_changed"],
            new_test_files=row["new_test_files"],
            days_since_last_deploy=row["days_since_last_deploy"],
            deploys_last_7_days=row["deploys_last_7_days"],
            hour_of_day=row["hour_of_day"],
            day_of_week=row["day_of_week"],
        )
        X_rows.append(_extract_features(event)[0])
        y.append(int(row["failed"]))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Handle class imbalance: typically ~10-20% of deploys fail
    # scale_pos_weight = count(negatives) / count(positives) balances this
    n_pos = max(y.sum(), 1)
    n_neg = len(y) - n_pos
    scale = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale,  # Handle class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def score_deployment(
    event: DeploymentEvent,
    outcomes: list[dict]
) -> tuple[float, list[RiskFactor], str]:
    """
    Main entry point for scoring.

    Returns:
        - risk_score: float 0-1
        - risk_factors: list of RiskFactor
        - method: "ml" | "heuristic" (so callers can tell users which was used)

    Logic:
        1. Try to train XGBoost on all outcomes
        2. If enough data + XGBoost available → use ML score
        3. Otherwise fall back to rule-based heuristics
    """
    model = train_model(outcomes)

    if model is not None:
        score, factors = _ml_score(event, model)
        return score, factors, "ml"
    else:
        score, factors = _heuristic_score(event)
        return score, factors, "heuristic"


def score_to_level(score: float) -> str:
    """Convert raw 0-1 probability to a human-readable risk level."""
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


def score_to_recommendation(score: float) -> str:
    """Translate risk level to a deployment recommendation."""
    if score >= 0.75:
        return "HOLD"
    elif score >= 0.5:
        return "DEPLOY_WITH_CAUTION"
    else:
        return "DEPLOY"
