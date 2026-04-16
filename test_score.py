"""
test_score.py — End-to-end test runner for the DORA Risk Scorer.

Same pattern as test_analyze.py in Pipeline-Failure-Analyzer:
- Hits /score-deploy with realistic deploy scenarios
- Prints the response in a readable format
- Covers LOW, MEDIUM, HIGH, and CRITICAL risk scenarios

Usage:
    1. python seed.py          (load demo data, activates ML model)
    2. uvicorn main:app --reload
    3. python test_score.py
"""

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 60


def run(deploy: dict):
    print(f"\n{'=' * 65}")
    print(f"→ Testing: {deploy['repo_name']} — \"{deploy['pr_title']}\"")
    print("  Sending to /score-deploy...")

    resp = httpx.post(f"{BASE_URL}/score-deploy", json=deploy, timeout=TIMEOUT)

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text}")
        return

    r = resp.json()
    print(f"\n  Risk Score   : {r['risk_score']:.3f}  →  {r['risk_level']}")
    print(f"  Recommendation: {r['recommendation']}")
    print(f"\n  Risk Factors:")
    for f in r["risk_factors"]:
        print(f"    [{f['impact'].upper()}] {f['factor']}: {f['detail']}")
    print(f"\n  LLM Advisory:\n  {r['llm_advisory']}")
    print(f"\n  DORA Snapshot ({r['repo_name']}):")
    d = r["dora_metrics_snapshot"]
    print(f"    Deploy Freq : {d['deployment_frequency_per_week']:.1f}/week")
    print(f"    CFR         : {d['change_failure_rate_pct']:.1f}%")
    print(f"    DORA Band   : {d['dora_band']}")


# ─── Test Case 1: LOW risk — small PR, good tests, mid-week ──────────────────
LOW_RISK_DEPLOY = {
    "repo_name": "api-service",
    "pr_title": "fix: correct null check in auth middleware",
    "author": "alice",
    "branch": "fix/auth-null",
    "pr_size_lines_changed": 38,
    "files_changed": 2,
    "new_test_files": 1,
    "days_since_last_deploy": 1.0,
    "deploys_last_7_days": 5,
    "hour_of_day": 11,
    "day_of_week": 2,      # Wednesday
    "commit_sha": "abc1234"
}

# ─── Test Case 2: MEDIUM risk — decent PR but no new tests ───────────────────
MEDIUM_RISK_DEPLOY = {
    "repo_name": "api-service",
    "pr_title": "feat: add user activity tracking",
    "author": "bob",
    "branch": "feat/activity-tracking",
    "pr_size_lines_changed": 310,
    "files_changed": 9,
    "new_test_files": 0,       # No tests added
    "days_since_last_deploy": 3.0,
    "deploys_last_7_days": 3,
    "hour_of_day": 14,
    "day_of_week": 3,          # Thursday
    "commit_sha": "def5678"
}

# ─── Test Case 3: HIGH risk — large PR, Friday deploy, stale ─────────────────
HIGH_RISK_DEPLOY = {
    "repo_name": "payments-service",
    "pr_title": "feat: complete checkout redesign with new payment flow",
    "author": "dan",
    "branch": "feat/checkout-redesign",
    "pr_size_lines_changed": 820,
    "files_changed": 18,
    "new_test_files": 1,
    "days_since_last_deploy": 9.0,
    "deploys_last_7_days": 1,
    "hour_of_day": 15,
    "day_of_week": 4,          # Friday afternoon
    "commit_sha": "ghi9012"
}

# ─── Test Case 4: CRITICAL risk — everything wrong ───────────────────────────
CRITICAL_RISK_DEPLOY = {
    "repo_name": "ml-pipeline",
    "pr_title": "feat: migrate entire inference stack to new model architecture",
    "author": "frank",
    "branch": "feat/new-arch",
    "pr_size_lines_changed": 1400,
    "files_changed": 28,
    "new_test_files": 0,
    "days_since_last_deploy": 21.0,   # 3 weeks since last deploy
    "deploys_last_7_days": 0,         # Team hasn't deployed recently at all
    "hour_of_day": 16,
    "day_of_week": 4,                 # Friday afternoon
    "commit_sha": "jkl3456"
}


if __name__ == "__main__":
    # Quick health check first
    health = httpx.get(f"{BASE_URL}/health").json()
    print(f"Server status: {health['status']}")
    print(f"Outcomes recorded: {health['outcomes_recorded']}")
    print(f"ML model active: {health['ml_model_active']}")

    if health["outcomes_recorded"] == 0:
        print("\n  No outcomes recorded. Run: python seed.py")
        print("   The scorer will still work in heuristic mode.\n")

    run(LOW_RISK_DEPLOY)
    run(MEDIUM_RISK_DEPLOY)
    run(HIGH_RISK_DEPLOY)
    run(CRITICAL_RISK_DEPLOY)
