"""
advisor.py — The LLM advisory layer.

What this does:
- Takes the ML risk score, risk factors, and DORA metrics
- Asks the LLM to generate a natural-language deployment advisory
- The advisory is actionable: not "risk is high" but "delay 2hrs until off-hours
  and split this PR into smaller chunks next time"

Why Groq (same as Pipeline Analyzer)?
- Free tier, fast inference, llama-3.3-70b is very capable for structured prompts
- Same pattern as agent.py in Pipeline-Failure-Analyzer for consistency

The LLM is NOT doing the risk scoring (that's XGBoost's job).
The LLM is only translating the score + factors into human language.
This avoids hallucination in the critical path — numbers come from code, 
words come from the LLM.
"""

import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from models import DeploymentEvent, RiskFactor, DORAMetrics

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_advisory(
    event: DeploymentEvent,
    risk_score: float,
    risk_level: str,
    risk_factors: list[RiskFactor],
    dora: DORAMetrics,
    scoring_method: str
) -> str:
    """
    Generate a natural language advisory for this deployment.

    Returns a concise advisory string — typically 3-5 sentences.
    Falls back to a template string if LLM call fails.
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
  have rollback ready, monitor for 30min post-deploy).
- If risk is HIGH/CRITICAL: explain the top 1-2 reasons clearly and suggest either 
  splitting the PR, adding tests, or waiting for a safer window.
- Keep it to 3-5 sentences maximum. No bullet points. Write as if talking to the developer.
- Do NOT just repeat the risk level back to them. Give new information.

Return ONLY the advisory text. No JSON. No preamble."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        # Graceful fallback — never crash the /score-deploy endpoint due to LLM error
        return (
            f"Risk level: {risk_level} ({risk_score:.0%}). "
            f"Top concern: {risk_factors[0].detail if risk_factors else 'None detected'}. "
            f"LLM advisory unavailable ({str(e)[:60]}). Proceed based on risk score."
        )
