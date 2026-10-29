"""
Cloud AI Assistant: takes structured output from the scanner, backtest
engine, or position sizing calculator, and asks Gemini (Google's free-tier
LLM API) to explain it in plain, conversational language -- turning a
list of reasoning strings and numbers into something that reads like an
analyst's summary, not a JSON dump.

Uses Gemini 2.5 Flash -- fits comfortably within Google's free tier for
this kind of occasional, on-demand explanation task (you're calling this
a handful of times per session, not thousands of times a minute).

IMPORTANT: this module explains what the deterministic detectors ALREADY
found -- it does not generate new signals, invent numbers, or override
anything. If it ever seems to add a fact not present in the input data,
that's the LLM hallucinating, not a new finding -- treat the underlying
detector output as the source of truth, and this as a readability layer
on top of it.
"""
from __future__ import annotations

import os

from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = """You are explaining the output of a rule-based intraday trading \
system to a trader who built it themselves and understands trading concepts. \
Your job is ONLY to explain what the data already says in plain, natural language -- \
never invent new facts, numbers, or opinions not present in the input. \
If confidence/scores are low or the setup looks marginal, say so honestly -- \
do not oversell any signal. Keep it to 3-5 sentences. Do not use markdown formatting."""


class AssistantError(RuntimeError):
    pass


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise AssistantError(
            "GEMINI_API_KEY is not set. Get a free key from "
            "https://aistudio.google.com/apikey and add it to your .env file."
        )
    return genai.Client(api_key=api_key)


def _call_gemini(prompt: str) -> str:
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.3,  # low temperature -- this is an explanation
                # task grounded in specific data, not creative writing;
                # keeping it low reduces the chance of embellishment.
                max_output_tokens=300,
            ),
        )
        if not response.text:
            raise AssistantError("Gemini returned an empty response.")
        return response.text.strip()
    except Exception as e:
        # Covers auth errors, rate limits, network issues -- surfaced as
        # one clear error rather than a raw SDK exception, since a
        # failure here should never crash whatever called this.
        raise AssistantError(f"Gemini API call failed: {e}") from e


def explain_scan_result(symbol: str, setup_type: str, direction: str,
                          score: float, reasoning: list[str]) -> str:
    prompt = (
        f"Scanner result for {symbol}:\n"
        f"Setup type: {setup_type}\n"
        f"Direction: {direction}\n"
        f"Score: {score:.2f}\n"
        f"Reasoning:\n" + "\n".join(f"- {r}" for r in reasoning)
    )
    return _call_gemini(prompt)


def explain_backtest_result(strategy: str, symbols: list[str],
                              performance: dict, trade_count: int) -> str:
    prompt = (
        f"Backtest result for strategy '{strategy}' on {', '.join(symbols)}:\n"
        f"Total trades: {trade_count}\n"
        f"Win rate: {performance.get('win_rate', 0):.1%}\n"
        f"Profit factor: {performance.get('profit_factor', 0):.2f}\n"
        f"Avg R-multiple: {performance.get('avg_r_multiple', 0):.2f}\n"
        f"Max drawdown: {performance.get('max_drawdown_points', 0):.2f} points\n"
        f"Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}\n"
    )
    return _call_gemini(prompt)


def explain_position_size(quantity: int, position_value: float,
                            actual_risk_pct: float, capped: bool,
                            reasoning: list[str]) -> str:
    prompt = (
        f"Position sizing result:\n"
        f"Quantity: {quantity} shares\n"
        f"Position value: Rs.{position_value:,.0f}\n"
        f"Actual risk: {actual_risk_pct:.2%} of capital\n"
        f"Capped by exposure limit: {capped}\n"
        f"Reasoning:\n" + "\n".join(f"- {r}" for r in reasoning)
    )
    return _call_gemini(prompt)
