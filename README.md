# DORA Deployment Risk Scorer

**AI-powered pre-deployment risk scoring using DORA metrics + XGBoost + LLM advisory.**

Predicts whether a deploy is likely to cause an incident **before it happens** — the opposite of the Pipeline-Failure-Analyzer which diagnoses failures **after they happen**.

## Architecture

```
DeploymentEvent (PR metadata)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  ← test_coverage_ratio, is_friday_afternoon,
│  (scorer.py)         │    is_large_pr, is_stale_deploy, ...
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  XGBoost Classifier  │  ← Trained on historical deploy outcomes from SQLite
│  (scorer.py)         │    Falls back to heuristics if <20 outcomes recorded
└─────────────────────┘
        │
        ├──── risk_score (0-1) + risk_factors
        │
        ▼
┌─────────────────────┐
│  DORA Metrics Layer  │  ← Deployment Frequency, Lead Time, CFR, MTTR
│  (dora.py)           │    Classified as Elite/High/Medium/Low band
└─────────────────────┘
        │
        ├──── dora_metrics_snapshot
        │
        ▼
┌─────────────────────┐
│  LLM Advisory        │  ← Groq llama-3.3-70b translates numbers into
│  (advisor.py)        │    actionable natural-language recommendations
└─────────────────────┘
        │
        ▼
   DeploymentRiskResult → FastAPI → /score-deploy
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/score-deploy` | POST | Score a deployment before it happens |
| `/record-outcome` | POST | Record actual deploy result (feedback loop) |
| `/dora-metrics` | GET | DORA metrics for all repos |
| `/dora-metrics/{repo}` | GET | DORA metrics for one repo |
| `/health` | GET | Service health + ML model status |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Seed demo deploy history (activates ML model)
python seed.py

# 4. Start the server
uvicorn main:app --reload

# 5. Run test scenarios (LOW / MEDIUM / HIGH / CRITICAL)
python test_score.py

# 6. Explore the Swagger UI
open http://localhost:8000/docs
```

## Key Design Decisions

**Why XGBoost, not an LLM, for scoring?**
LLMs hallucinate numbers. XGBoost gives deterministic probabilities from real historical data. The LLM only generates the *explanation* — numbers come from code.

**Why SQLite, not ChromaDB?**
Deploy outcomes are structured tabular data (features + labels). We need GROUP BY, ORDER BY, and aggregate queries — SQL is the right tool. ChromaDB is for unstructured text embeddings.

**Heuristic fallback:**
The system works on day 1 with zero history. Rule-based scoring activates until 20+ outcomes are recorded, then XGBoost takes over automatically.

**Feedback loop:**
POST to `/record-outcome` after each deploy. The model retrains on the next `/score-deploy` call. No manual retraining needed.

## DORA Performance Bands

| Band | Deploy Freq | Lead Time | Change Failure Rate | MTTR |
|---|---|---|---|---|
| Elite | >1/day | <1hr | <5% | <1hr |
| High | 1/week | <1day | <10% | <1day |
| Medium | 1/month | <1week | <15% | <1week |
| Low | <1/month | >6mo | >15% | >1week |
