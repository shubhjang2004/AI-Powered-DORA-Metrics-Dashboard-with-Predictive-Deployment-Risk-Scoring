# AI-Powered DORA Metrics Dashboard with Predictive Deployment Risk Scoring

> A **production-ready pre-deployment risk scoring system** that predicts CI/CD failures before they happen — using XGBoost, DORA metrics, and LLM advisory, all exposed via a FastAPI backend with a self-improving feedback loop.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

**Deployment failures are hard to predict** — teams lack pre-deploy signals, causing production incidents that take hours to recover from. This system flips the script: instead of diagnosing failures *after* they happen, it scores deployment risk *before* you deploy.

> **Companion project:** [Pipeline-Failure-Analyzer](https://github.com/shubhjang2004/Pipeline-Failure-Analyzer) diagnoses failures *after* they happen. This project prevents them *before* they happen.

**What it does:**
- Scores each deployment 0–1 for incident risk using 12 engineered PR features
- Computes 4 DORA metrics with Elite/High/Medium/Low performance bands
- Generates natural-language advisory using Groq LLaMA-3.3-70b
- Self-improves: retrains XGBoost automatically as you record real outcomes
- Works from day one: heuristic fallback activates until 20+ outcomes are recorded

---

## Architecture

```
DeploymentEvent (PR metadata)
        │
        ▼
┌──────────────────────┐
│  Feature Engineering  │  ← 12 features: test_coverage_ratio, is_friday_afternoon,
│  (scorer.py)          │    is_large_pr, is_stale_deploy, churn, cadence, ...
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  XGBoost Classifier   │  ← Trained on historical deploy outcomes (SQLite)
│  (scorer.py)          │    Heuristic fallback if < 20 outcomes recorded
└──────────────────────┘
        │
        ├──── risk_score (0–1) + risk_factors
        │
        ▼
┌──────────────────────┐
│  DORA Metrics Layer   │  ← Deployment Frequency, Lead Time, CFR, MTTR
│  (dora.py)            │    Classified as Elite / High / Medium / Low
└──────────────────────┘
        │
        ├──── dora_metrics_snapshot
        │
        ▼
┌──────────────────────┐
│  LLM Advisory         │  ← Groq LLaMA-3.3-70b translates numbers into
│  (advisor.py)         │    actionable natural-language recommendations
└──────────────────────┘
        │
        ▼
   DeploymentRiskResult → FastAPI → /score-deploy
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/shubhjang2004/AI-Powered-DORA-Metrics-Dashboard-with-Predictive-Deployment-Risk-Scoring.git
cd AI-Powered-DORA-Metrics-Dashboard-with-Predictive-Deployment-Risk-Scoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Seed demo deploy history (activates ML model)
python seed.py

# 5. Start the server
uvicorn main:app --reload

# 6. Run test scenarios (LOW / MEDIUM / HIGH / CRITICAL)
python test_score.py

# 7. Explore the interactive Swagger UI
open http://localhost:8000/docs
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/score-deploy` | POST | Score a deployment before it happens |
| `/record-outcome` | POST | Record actual deploy result (feedback loop) |
| `/dora-metrics` | GET | DORA metrics across all repos |
| `/dora-metrics/{repo}` | GET | DORA metrics for a specific repo |
| `/health` | GET | Service health + ML model status |

---

## Features

### 12 Engineered PR Risk Features
- `pr_size` — lines changed
- `churn_ratio` — files modified / files added
- `test_coverage_delta` — test file change ratio
- `deploy_cadence` — recent deploy frequency
- `is_friday_afternoon` — high-risk time flag
- `is_large_pr` — threshold-based size flag
- `is_stale_deploy` — time since last successful deploy
- ...and more in `scorer.py`

### DORA Performance Bands

| Band | Deploy Freq | Lead Time | Change Failure Rate | MTTR |
|---|---|---|---|---|
| Elite | > 1/day | < 1hr | < 5% | < 1hr |
| High | 1/week | < 1day | < 10% | < 1day |
| Medium | 1/month | < 1week | < 15% | < 1week |
| Low | < 1/month | > 6mo | > 15% | > 1week |

### Self-Improving Feedback Loop
Record real outcomes via `/record-outcome`. The model retrains automatically on the next `/score-deploy` call — no manual intervention needed.

---

## Project Structure

```
├── main.py              # FastAPI app — all endpoints
├── scorer.py            # Feature engineering + XGBoost classifier
├── dora.py              # DORA metrics computation + band classification
├── advisor.py           # Groq LLM advisory layer
├── models.py            # Pydantic request/response schemas
├── db.py                # SQLite persistence layer
├── seed.py              # Demo data seeder (activates ML model)
├── test_score.py        # Test scenarios: LOW / MEDIUM / HIGH / CRITICAL
└── requirements.txt     # Dependencies
```

---

## Key Design Decisions

**Why XGBoost, not an LLM, for scoring?**
LLMs hallucinate numbers. XGBoost gives deterministic probabilities from real historical data. The LLM only generates the *explanation* — all numbers come from code.

**Why SQLite, not ChromaDB?**
Deploy outcomes are structured tabular data (features + labels). We need `GROUP BY`, `ORDER BY`, and aggregate queries — SQL is the right tool. ChromaDB is for unstructured text embeddings (see Pipeline-Failure-Analyzer).

**Why heuristic fallback?**
The system is useful from day one with zero history. Rule-based scoring activates until 20+ outcomes are recorded, then XGBoost takes over automatically — cold-start problem solved.

---

## Requirements

- **Python** >= 3.8
- **FastAPI** + **uvicorn**
- **XGBoost** >= 1.7
- **Groq API key** (free tier available at [console.groq.com](https://console.groq.com))

See `requirements.txt` for the full list.

---

## Related Project

**[Pipeline-Failure-Analyzer](https://github.com/shubhjang2004/Pipeline-Failure-Analyzer)** — diagnoses CI/CD failures *after* they happen using LangGraph + RAG + ChromaDB. Together, these two systems form a complete CI/CD intelligence layer: predict before, diagnose after.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
