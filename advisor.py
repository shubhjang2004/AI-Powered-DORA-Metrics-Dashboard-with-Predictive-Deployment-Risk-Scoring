"""
advisor.py — The LLM advisory layer.

Swapped from Groq/langchain-groq to Anthropic Claude (claude-haiku-4-5).
Same interface as before — generate_advisory() returns a plain string.

Why Claude instead of Groq?
- You already have an Anthropic API key
- claude-haiku-4-5 is fast and cheap — ideal for advisory text generation
- Same separation of concerns: XGBoost does the numbers, Claude writes the words

The rest of the codebase (main.py, scorer.py, dora.py, db.py, models.py) is unchanged.
"""

import os
import anthropic
from models import DeploymentEvent, RiskFactor, DORAMetrics

# Lazy singleton — created on first call, reused after
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. "
                "Add it to your .env file or environment."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_advisory(
    event: DeploymentEvent,
    risk_score: float,
    risk_level: str,
    risk_factors: list[RiskFactor],
    dora: DORAMetrics,
    scoring_method: str,
) -> str:
    """
    Generate a natural-language deployment advisory using Claude.

    Returns a concise advisory string (3-5 sentences).
    Falls back to a template string if the API call fails — the
    /score-deploy endpoint should never crash due to an LLM error.
    """
    factors_text = "\n".join(
        f"  - [{f.impact.upper()}] {f.factor}: {f.detail}"
        for f in risk_factors
    )

    prompt = f"""You are a senior SRE at a software company. A developer is about to deploy.
Give them a brief, actionable deployment advisory based on the risk assessment below.

== Deployment Info ==
Repo: {event.repo_name}
PR: "{event.pr_title}"
Author: {event.author}
Branch: {event.branch}

== Risk Assessment ==
Risk Score: {risk_score:.2f} / 1.0
Risk Level: {risk_level}
Scoring Method: {scoring_method} model

Risk Factors Detected:
{factors_text}

== Current DORA Metrics for {event.repo_name} ==
Deployment Frequency: {dora.deployment_frequency_per_week:.1f} deploys/week
Change Failure Rate: {dora.change_failure_rate_pct:.1f}%
DORA Band: {dora.dora_band}
Total Deploys Recorded: {dora.total_deploys_recorded}

== Instructions ==
- Be direct and specific. Mention actual numbers from the risk factors.
- If risk is LOW: confirm it's safe to deploy and briefly note what looks good.
- If risk is MEDIUM: suggest one concrete precaution (e.g., deploy during business hours,
  have rollback ready, monitor for 30 min post-deploy).
- If risk is HIGH/CRITICAL: explain the top 1-2 reasons clearly and suggest either
  splitting the PR, adding tests, or waiting for a safer deployment window.
- Keep it to 3-5 sentences maximum. No bullet points. Write as if talking to the developer.
- Do NOT just repeat the risk level back to them. Give new, actionable information.

Return ONLY the advisory text. No JSON. No preamble. No sign-off."""

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        # Graceful fallback — never crash /score-deploy because of an LLM failure
        top_concern = risk_factors[0].detail if risk_factors else "None detected"
        return (
            f"Risk level: {risk_level} ({risk_score:.0%}). "
            f"Top concern: {top_concern}. "
            f"LLM advisory unavailable ({str(e)[:80]}). "
            "Proceed based on the risk score and factors above."
        )