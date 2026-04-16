"""
seed.py — Seeds realistic deploy history into SQLite.

Why seeding matters:
- XGBoost needs 20+ labeled outcomes before it activates
- Without seed data, the system runs in heuristic mode (rules-based)
- With seed data, the ML model kicks in and learns from patterns

The seed data covers 3 repos across 40 deployments with realistic
success/failure patterns. Failures are seeded to correlate with:
- Large PRs (>500 lines)
- No new tests
- Friday afternoon deploys
- Long gaps since last deploy

This mirrors real-world failure patterns found in DORA research.

Run this once:
    python seed.py

Delete dora.db to re-seed from scratch.
"""

from db import init_db, record_outcome, count_outcomes

# Realistic deploy history: (repo, pr_title, author, branch, lines, files, 
#                             test_files, days_gap, deploys_7d, hour, dow, failed)
SEED_DEPLOYS = [
    # api-service — healthy team, mostly small PRs, frequent deploys
    ("api-service", "fix: correct null check in user resolver",    "alice", "fix/null-check",      45,  3, 1, 1.0, 5, 10, 1, False),
    ("api-service", "feat: add pagination to /users endpoint",     "bob",   "feat/pagination",     180, 6, 2, 0.5, 5, 11, 0, False),
    ("api-service", "chore: upgrade express to 4.18.2",            "alice", "chore/express-bump",  20,  2, 0, 1.0, 4, 14, 2, False),
    ("api-service", "feat: JWT refresh token support",             "carol", "feat/jwt-refresh",    310, 9, 3, 2.0, 5, 10, 1, False),
    ("api-service", "fix: rate limiter bypass on /health",         "bob",   "fix/rate-limit",      55,  2, 1, 1.0, 5,  9, 2, False),
    ("api-service", "feat: GraphQL subscriptions",                 "carol", "feat/subscriptions", 890, 18, 2, 3.0, 3, 15, 4, True),   # Large PR, Friday, failed
    ("api-service", "fix: revert GraphQL subscriptions",           "alice", "revert/subscriptions",50,  3, 0, 0.2, 4, 16, 4, False),
    ("api-service", "feat: add Redis caching",                     "bob",   "feat/redis-cache",    420, 11, 4, 2.0, 5, 11, 0, False),
    ("api-service", "chore: update Dockerfile base image",         "carol", "chore/docker",         30,  1, 0, 1.0, 5, 10, 3, False),
    ("api-service", "feat: bulk user import endpoint",             "alice", "feat/bulk-import",    670, 14, 1, 1.0, 5, 16, 4, True),   # Large PR, no tests, Friday, failed
    ("api-service", "fix: bulk import validation",                 "bob",   "fix/bulk-validation", 120,  4, 2, 0.5, 5,  9, 1, False),
    ("api-service", "feat: API versioning v2",                     "carol", "feat/api-v2",         280,  8, 3, 2.0, 4, 11, 2, False),
    ("api-service", "fix: CORS headers for mobile clients",        "alice", "fix/cors",             35,  2, 1, 1.0, 5, 14, 3, False),
    ("api-service", "feat: webhook delivery system",               "bob",   "feat/webhooks",       550, 12, 2, 4.0, 3, 10, 1, True),   # Large PR, stale, failed
    ("api-service", "fix: webhook retry logic",                    "carol", "fix/webhook-retry",   160,  5, 2, 1.0, 5, 11, 2, False),

    # payments-service — slower team, bigger PRs, higher CFR
    ("payments-service", "feat: Stripe payment intent flow",       "dan",   "feat/stripe",         720, 15, 2, 7.0, 2, 14, 2, True),   # Large PR, stale, few deploys, failed
    ("payments-service", "fix: revert Stripe integration",         "emily", "revert/stripe",        80,  3, 0, 0.3, 2, 10, 3, False),
    ("payments-service", "feat: Stripe payment intent v2",         "dan",   "feat/stripe-v2",      340,  9, 3, 5.0, 2, 11, 1, False),
    ("payments-service", "fix: idempotency key collision",         "emily", "fix/idempotency",     110,  4, 2, 2.0, 2, 10, 2, False),
    ("payments-service", "feat: PayPal integration",               "dan",   "feat/paypal",         980, 20, 1, 14.0,1, 15, 4, True),   # Very large, no tests, Friday, stale, failed
    ("payments-service", "fix: revert PayPal integration",         "emily", "revert/paypal",        65,  2, 0, 0.2, 2, 10, 0, False),
    ("payments-service", "chore: add PCI compliance headers",      "dan",   "chore/pci",           140,  5, 1, 3.0, 2, 11, 1, False),
    ("payments-service", "feat: refund workflow",                  "emily", "feat/refunds",        460, 11, 3, 5.0, 2, 10, 2, False),
    ("payments-service", "fix: refund amount calculation",         "dan",   "fix/refund-calc",      95,  3, 2, 1.0, 3, 14, 3, False),
    ("payments-service", "feat: subscription billing",             "emily", "feat/subscriptions",  830, 17, 2, 10.0,1, 16, 4, True),   # Large, stale, Friday, failed
    ("payments-service", "fix: billing cycle off-by-one",         "dan",   "fix/billing-cycle",   140,  4, 2, 0.5, 2, 10, 0, False),
    ("payments-service", "chore: upgrade node 18→20",             "emily", "chore/node-upgrade",   30,  1, 0, 2.0, 2, 11, 1, False),

    # ml-pipeline — data science team, irregular deploy cadence
    ("ml-pipeline", "feat: BERT fine-tuning pipeline",             "frank", "feat/bert",           580, 12, 1, 14.0,1, 10, 1, True),   # Large, no tests, stale, failed
    ("ml-pipeline", "fix: data loader memory leak",                "grace", "fix/memory-leak",     220,  6, 2, 1.0, 2, 11, 2, False),
    ("ml-pipeline", "feat: model versioning with MLflow",          "frank", "feat/mlflow",         390, 10, 3, 7.0, 1, 10, 0, True),   # Stale, low cadence, failed
    ("ml-pipeline", "fix: MLflow tracking URI config",             "grace", "fix/mlflow-config",    75,  3, 1, 0.5, 2, 11, 1, False),
    ("ml-pipeline", "feat: A/B testing framework",                 "frank", "feat/ab-test",        460, 11, 2, 5.0, 2, 10, 2, False),
    ("ml-pipeline", "chore: update Python 3.10→3.12",             "grace", "chore/python-bump",    40,  2, 0, 3.0, 2, 14, 3, False),
    ("ml-pipeline", "feat: feature store integration",             "frank", "feat/feature-store",  750, 16, 2, 21.0,0, 15, 4, True),   # Very stale, no recent deploys, Friday, failed
    ("ml-pipeline", "fix: revert feature store",                   "grace", "revert/feature-store", 60,  2, 0, 0.3, 1, 10, 0, False),
    ("ml-pipeline", "feat: feature store v2 (smaller PR)",        "frank", "feat/feature-store-v2",310, 8, 3, 2.0, 2, 11, 1, False),
    ("ml-pipeline", "fix: training job timeout on large datasets", "grace", "fix/timeout",         185,  5, 2, 1.5, 2, 10, 2, False),
    ("ml-pipeline", "feat: online inference endpoint",             "frank", "feat/inference",      420, 10, 4, 3.0, 3, 11, 3, False),
]


def seed():
    existing = count_outcomes()
    if existing > 0:
        print(f"Database already has {existing} deploy outcomes. Skipping seed.")
        print("Delete dora.db to re-seed from scratch.")
        return

    init_db()
    seeded = 0
    for row in SEED_DEPLOYS:
        (repo, pr_title, author, branch, lines, files, test_files,
         days_gap, deploys_7d, hour, dow, failed) = row

        record_outcome({
            "repo_name": repo,
            "pr_title": pr_title,
            "author": author,
            "branch": branch,
            "pr_size_lines_changed": lines,
            "files_changed": files,
            "new_test_files": test_files,
            "days_since_last_deploy": days_gap,
            "deploys_last_7_days": deploys_7d,
            "hour_of_day": hour,
            "day_of_week": dow,
            "failed": int(failed),
            "commit_sha": None,
        })
        seeded += 1

    failures = sum(1 for row in SEED_DEPLOYS if row[-1])
    print(f"Seeded {seeded} deploy outcomes across 3 repos.")
    print(f"  Failures: {failures}/{seeded} ({failures/seeded*100:.0f}% change failure rate)")
    print(f"  Repos: api-service, payments-service, ml-pipeline")
    print(f"  ML model will activate (20+ outcomes): YES")


if __name__ == "__main__":
    seed()
