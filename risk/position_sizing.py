"""
Converts a trade setup (entry, stop-loss) into an actual share quantity,
based on how much of your total capital you're willing to risk on this
ONE trade -- not "how many shares can I afford" (which is what most
beginners default to, and is exactly how a string of losses compounds
into serious capital damage).

This directly targets "overtrading" and "no defined setup" from your
notes in a way none of the signal detectors can: even a perfectly valid
signal, sized wrong, can blow up an account. Position sizing is risk
management, not signal generation -- it doesn't care whether the setup
is a breakout or a pullback, only how much of your capital is on the
line if the stop-loss gets hit.

Method: fixed-fractional position sizing -- the standard, widely used
approach in professional risk management (popularized in trading
literature going back to Ralph Vince's work on optimal f, though this
uses the much simpler and more conservative fixed-percentage variant,
not Vince's more aggressive optimal-f formula). Risk a FIXED percentage
of capital per trade (commonly 0.5%-2% for intraday), regardless of how
"confident" the setup looks -- confidence should affect whether you take
the trade at all, not how much you risk once you've decided to.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from backtest.strategy_base import Signal, SignalDirection


@dataclass
class PositionSizeResult:
    quantity: int
    position_value: float
    actual_risk_amount: float
    actual_risk_pct_of_capital: float
    capped_by_exposure_limit: bool
    reasoning: list[str]


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct_per_trade: float = 0.01,
    max_capital_pct_per_trade: float = 0.25,
    leverage: float = 1.0,
) -> PositionSizeResult | None:
    """
    capital: total trading capital available.
    entry_price / stop_loss_price: the trade's actual entry and stop --
        risk_per_share is derived from these, not assumed.
    risk_pct_per_trade: fraction of capital you're willing to lose if
        this ONE trade hits its stop. 0.01 = 1% -- a common conservative
        default for intraday. Going materially above 2-3% per trade is
        how a short losing streak (which WILL happen to any strategy,
        including a good one) turns into serious capital damage.
    max_capital_pct_per_trade: separate cap on total position VALUE
        (not risk) as a fraction of capital*leverage -- this prevents a
        very tight stop from sizing you into a position so large that a
        normal price swing (not even hitting your stop) causes stress
        selling or margin issues. This is the safeguard against exactly
        the kind of tiny/degenerate stop distances we found earlier in
        backtesting (see trend_continuation_strategy.py's min_stop_distance
        fix) -- risk-based sizing alone can produce huge share counts on
        a tight stop; this caps that.
    leverage: intraday margin multiplier your broker provides for cash
        equity (MIS-type products). Defaults to 1.0 (no leverage) --
        pass your broker's actual intraday margin multiplier if you
        intend to use it, but be aware leverage scales losses as much
        as gains.

    Returns None if the setup is degenerate (zero/negative risk per share).
    """
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share <= 0:
        return None

    reasoning = []
    risk_amount = capital * risk_pct_per_trade
    reasoning.append(
        f"Risking {risk_pct_per_trade:.1%} of capital (Rs.{capital:,.0f}) "
        f"= Rs.{risk_amount:,.2f} on this trade"
    )

    raw_quantity = risk_amount / risk_per_share
    reasoning.append(
        f"Risk per share = Rs.{risk_per_share:.2f} (entry Rs.{entry_price:.2f} "
        f"- stop Rs.{stop_loss_price:.2f}) -> raw size = {raw_quantity:.1f} shares"
    )

    max_position_value = capital * leverage * max_capital_pct_per_trade
    raw_position_value = raw_quantity * entry_price
    capped = False

    if raw_position_value > max_position_value:
        capped = True
        final_quantity = math.floor(max_position_value / entry_price)
        reasoning.append(
            f"Risk-based size would use Rs.{raw_position_value:,.0f} of capital -- "
            f"exceeds the {max_capital_pct_per_trade:.0%} exposure cap "
            f"(Rs.{max_position_value:,.0f}). Capping position size instead of "
            f"risking over-concentration in one trade -- this means your ACTUAL "
            f"risk on this trade will be less than {risk_pct_per_trade:.1%}, which "
            f"is the safe direction to be wrong in."
        )
    else:
        final_quantity = math.floor(raw_quantity)

    if final_quantity <= 0:
        reasoning.append(
            "Computed quantity rounds down to 0 shares -- capital too small "
            "for this risk_pct/stop combination at this share price. Either "
            "increase risk_pct_per_trade, use a wider stop, or skip this "
            "symbol at this price level."
        )
        return PositionSizeResult(0, 0.0, 0.0, 0.0, capped, reasoning)

    actual_position_value = final_quantity * entry_price
    actual_risk_amount = final_quantity * risk_per_share
    actual_risk_pct = actual_risk_amount / capital

    reasoning.append(
        f"Final: {final_quantity} shares, Rs.{actual_position_value:,.0f} position value, "
        f"actual risk Rs.{actual_risk_amount:,.2f} ({actual_risk_pct:.2%} of capital)"
    )

    return PositionSizeResult(
        quantity=final_quantity,
        position_value=actual_position_value,
        actual_risk_amount=actual_risk_amount,
        actual_risk_pct_of_capital=actual_risk_pct,
        capped_by_exposure_limit=capped,
        reasoning=reasoning,
    )


def size_signal(
    signal: Signal,
    entry_price: float,
    capital: float,
    risk_pct_per_trade: float = 0.01,
    max_capital_pct_per_trade: float = 0.25,
    leverage: float = 1.0,
) -> PositionSizeResult | None:
    """Convenience wrapper: size a Signal object directly (from any of the
    detector strategies) rather than pulling out stop_loss manually."""
    return calculate_position_size(
        capital=capital,
        entry_price=entry_price,
        stop_loss_price=signal.stop_loss,
        risk_pct_per_trade=risk_pct_per_trade,
        max_capital_pct_per_trade=max_capital_pct_per_trade,
        leverage=leverage,
    )
