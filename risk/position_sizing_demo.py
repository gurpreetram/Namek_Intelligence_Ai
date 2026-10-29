"""
Demonstrates position sizing across realistic scenarios -- run this
directly with your own capital/leverage numbers to see what quantity the
system would suggest for a given setup.

IMPORTANT on leverage: Indian brokers (Groww, Upstox, etc.) offer
intraday margin (often called MIS) that lets you take a position larger
than your raw capital, closed out automatically before market close.
Check your broker's actual intraday leverage for cash equity (varies by
stock, commonly 3x-5x for liquid large-caps) and set that as `leverage`
below -- with leverage=1.0 (no margin), you'll often see the exposure cap
kick in on completely normal stop distances, as shown in Scenario 1.

Run: python -m risk.position_sizing_demo
"""
from __future__ import annotations

from risk.position_sizing import calculate_position_size


def show(title: str, **kwargs) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    result = calculate_position_size(**kwargs)
    if result is None:
        print("  Degenerate setup (zero/negative risk per share) -- skip this trade.")
        return
    for line in result.reasoning:
        print(f"  {line}")


def main() -> None:
    capital = 100_000  # change this to your actual trading capital

    show(
        "Scenario 1: Normal breakout stop, NO leverage (cash only)",
        capital=capital, entry_price=1300, stop_loss_price=1290,
        risk_pct_per_trade=0.01, leverage=1.0,
    )

    show(
        "Scenario 2: Same trade, WITH broker's intraday leverage (e.g. 4x)",
        capital=capital, entry_price=1300, stop_loss_price=1290,
        risk_pct_per_trade=0.01, leverage=4.0,
    )

    show(
        "Scenario 3: Tighter stop from a pullback-based strategy",
        capital=capital, entry_price=800, stop_loss_price=797,
        risk_pct_per_trade=0.01, leverage=4.0,
    )

    show(
        "Scenario 4: Being more conservative (0.5% risk instead of 1%)",
        capital=capital, entry_price=1300, stop_loss_price=1290,
        risk_pct_per_trade=0.005, leverage=4.0,
    )

    print(f"\n{'=' * 60}")
    print("Change `capital` and `leverage` at the top of main() to your own "
          "numbers, and check your broker app for your actual intraday "
          "margin multiplier on cash equity -- it varies by stock, not a "
          "single fixed number across your whole watchlist.")


if __name__ == "__main__":
    main()
