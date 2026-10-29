"""
Pydantic models for the API layer -- request validation and response
shapes. Kept separate from main.py so the API contract is easy to read
in one place without wading through endpoint logic.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StrategyName(str, Enum):
    OPENING_RANGE_BREAKOUT = "opening_range_breakout"
    ROLLING_RANGE_BREAKOUT = "rolling_range_breakout"
    VALIDATED_BREAKOUT = "validated_breakout"
    TREND_CONTINUATION = "trend_continuation"
    REVERSAL_DETECTION = "reversal_detection"
    VWAP_RECLAIM = "vwap_reclaim"
    SMART_MONEY = "smart_money"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    MOMENTUM_ENTRY = "momentum_entry"


class ScanRequest(BaseModel):
    symbols: list[str] = Field(..., description="Symbols to scan, must already have cached data")
    min_score: float = Field(0.5, ge=0.0, le=1.0)


class ScanResultResponse(BaseModel):
    symbol: str
    score: float
    setup_type: str
    direction: str
    reasoning: list[str]


class DailyPickResponse(BaseModel):
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    target_price: float
    score: float
    setup_type: str
    reasoning: list[str]
    composite_score: Optional[float] = None
    composite_reasoning: Optional[list[str]] = None


class WatchlistItemResponse(BaseModel):
    symbol: str
    direction: str
    category: str  # "confirmed_pick" | "near_level" -- see paper_trading/watchlist.py's WatchlistEntry.reason
    detail: str
    score: Optional[float] = None  # confirmed_pick only
    distance_atr: Optional[float] = None  # near_level only
    # confirmed_pick only: real entry/stop-loss plus a display-only target
    # (see detectors.scanner.display_target_price) -- None for near_level,
    # which has no confirmed setup to price yet.
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    # near_level only: the real numbers behind `detail`'s free text.
    current_price: Optional[float] = None
    level_price: Optional[float] = None
    # confirmed_pick only: multi-factor re-ranking score (see detectors/
    # composite_score.py) and the human-readable breakdown behind it.
    composite_score: Optional[float] = None
    composite_reasoning: Optional[list[str]] = None


class WatchlistResponse(BaseModel):
    status: str = "ok"  # "ok" | "stale_previous_day" -- see api/main.py's get_watchlist()
    message: str = ""  # human-readable explanation, set whenever status != "ok"
    generated_at: str
    items: list[WatchlistItemResponse] = []


class OpenPositionResponse(BaseModel):
    symbol: str
    strategy_name: str
    direction: str
    entry_time: str
    entry_price: float
    stop_loss: float
    target: float
    reasoning: list[str] = []
    # Both None when paper_trading/live_poller.py's latest_prices.json
    # isn't available/fresh enough (e.g. the live session predates this
    # field, or hasn't polled this symbol yet this cycle) -- see
    # api/main.py's get_live_activity() for the exact staleness rule.
    # NOT a live quote fetched by the API itself -- see live_poller.py's
    # LATEST_PRICES_PATH docstring for why.
    current_price: Optional[float] = None
    status_indicator: Optional[str] = None  # "near_target" | "near_stop" | "near_entry" | None


class ClosedTradeResponse(BaseModel):
    symbol: str
    strategy_name: str
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    # Specific plain-language reason(s) behind an early-exit-warning exit
    # (exit_reason in "early_warning_momentum_reversal" |
    # "early_warning_setup_invalidated") -- empty for every other
    # exit_reason, which are already self-explanatory from price alone.
    exit_reasoning: list[str] = []
    pnl_points: float
    r_multiple: float


class LiveActivityResponse(BaseModel):
    open_positions: list[OpenPositionResponse]
    closed_trades: list[ClosedTradeResponse]


class StockMomentumResponse(BaseModel):
    symbol: str
    return_pct: float  # % change since today's market open -- see detectors/sector_momentum.py
    magnitude: str = "unknown"  # "small" | "moderate" | "strong" | "unknown" -- vs this stock's own typical daily range


class SectorMomentumResponse(BaseModel):
    sector: str
    return_pct: float  # % change since today's market open, equal-weighted across constituents
    num_constituents: int
    top_stocks: list[StockMomentumResponse]
    magnitude: str = "unknown"


class SectorsResponse(BaseModel):
    status: str = "ok"  # "ok" | "stale_previous_day" | "insufficient_data" -- see api/main.py's get_sectors()
    message: str = ""  # human-readable explanation, set whenever status != "ok"
    generated_at: str
    as_of: Optional[str] = None  # None when status != "ok"
    sector_coverage_pct: float  # % of the cached universe that has a sector label -- honesty context
    sectors: list[SectorMomentumResponse] = []  # sorted by |return_pct| descending, most-trending first


class MarketBreadthResponse(BaseModel):
    status: str = "ok"  # "ok" | "stale_previous_day" -- see api/main.py's get_market_breadth()
    message: str = ""  # human-readable explanation, set whenever status != "ok"
    as_of: Optional[str] = None  # None when status != "ok"
    advancing: int = 0
    declining: int = 0
    flat: int = 0
    total: int = 0
    pct_advancing: float = 0.0
    pct_declining: float = 0.0
    pct_flat: float = 0.0
    interpretation: str = ""  # "broadly bullish" | "broadly bearish" | "mixed/neutral"


class SectorMomentumPicksResponse(BaseModel):
    status: str  # "ok" | "market_closed" | "insufficient_data"
    message: str  # human-readable explanation, always set (empty-ish for "ok")
    generated_at: str  # when this response was built
    data_as_of: str | None = None  # timestamp of the live data actually used, None if status != "ok"
    lookback_minutes: int
    up_sectors: list[SectorMomentumResponse] = []
    down_sectors: list[SectorMomentumResponse] = []
    disclaimer: str


class DailyPicksResponse(BaseModel):
    buy: list[DailyPickResponse]
    sell: list[DailyPickResponse]


class StrongTrendStockResponse(BaseModel):
    symbol: str
    direction: str  # "long" | "short"
    current_price: float
    move_pct_from_open: float
    trending_since: str  # ISO timestamp -- frontend computes the display duration, same pattern as timeAgo()
    volume_status: str  # human-readable, see detectors/strong_trend.py's check_volume_consistency()
    sector: Optional[str] = None
    sector_aligned: Optional[bool] = None  # None if the symbol has no sector mapping


class StrongTrendResponse(BaseModel):
    status: str = "ok"  # "ok" | "stale_previous_day" -- see api/main.py's get_strong_trend_stocks()
    message: str = ""
    generated_at: str
    as_of: Optional[str] = None
    longs: list[StrongTrendStockResponse] = []
    shorts: list[StrongTrendStockResponse] = []
    disclaimer: str


class TopMoverResponse(BaseModel):
    symbol: str
    move_pct_from_open: float
    current_price: float
    volume_zscore: Optional[float] = None  # None if not enough bars yet today to judge
    volume_label: str  # "high" | "normal" | "low" | "unknown"


class TopMoversResponse(BaseModel):
    status: str = "ok"  # "ok" | "stale_previous_day" -- see api/main.py's get_top_movers()
    message: str = ""
    generated_at: str
    as_of: Optional[str] = None
    gainers: list[TopMoverResponse] = []
    losers: list[TopMoverResponse] = []
    disclaimer: str


class BestPickResponse(BaseModel):
    symbol: str
    direction: str  # "long" | "short"
    current_price: float
    entry_price: float  # same value as current_price -- both exposed since callers ask for both by name
    move_pct_from_open: float
    volume_zscore: float
    volume_label: str  # "high" | "normal" | "low" -- see detectors/top_movers.py's classify_volume_zscore()
    stop_loss: float
    risk_per_share: float  # |entry_price - stop_loss|
    target_1: float  # 1:2 risk-reward -- partial-exit level
    target_2: float  # 1:3 risk-reward -- hold-remainder-to-EOD level
    rank_score: float  # |move%| x volume z-score
    reasoning: list[str]


class BestPicksResponse(BaseModel):
    status: str = "ok"  # "ok" | "stale_previous_day" -- see api/main.py's get_best_picks()
    message: str = ""
    generated_at: str
    as_of: Optional[str] = None
    picks: list[BestPickResponse] = []
    disclaimer: str


class EarlyMomentumPickResponse(BaseModel):
    symbol: str
    direction: str  # "long" | "short"
    current_price: float
    entry_price: float  # same value as current_price -- both exposed since callers ask for both by name
    move_pct_from_open: float
    volume_zscore: float
    sector: Optional[str] = None
    sector_aligned: Optional[bool] = None  # None if the symbol has no sector mapping (vacuous gate pass)
    stop_loss: float
    risk_per_share: float  # |entry_price - stop_loss|
    target_1: float  # 1:2 risk-reward -- partial-exit level
    target_2: float  # 1:3 risk-reward -- hold-remainder level
    rank_score: float  # volume z-score x sector alignment score
    reasoning: list[str]


class EarlyMomentumResponse(BaseModel):
    status: str = "ok"  # "ok" | "stale_previous_day" -- see api/main.py's get_early_momentum()
    message: str = ""
    generated_at: str
    as_of: Optional[str] = None
    picks: list[EarlyMomentumPickResponse] = []
    disclaimer: str


class PartialExitRequest(BaseModel):
    symbol: str
    direction: str  # "long" | "short"
    total_quantity: int = Field(..., gt=0)
    exit_quantity: int = Field(..., gt=0)
    exit_price: float = Field(..., gt=0)
    remaining_quantity: int = Field(..., ge=0)
    new_stop_loss: float = Field(..., gt=0)


class PartialExitResponse(BaseModel):
    id: str
    symbol: str
    direction: str
    total_quantity: int
    exit_quantity: int
    exit_price: float
    remaining_quantity: int
    new_stop_loss: float
    created_at: str
    # Reconciled from paper_trading/journal.jsonl's open positions and
    # paper_trading/live_poller.py's latest_prices.json (same sources
    # /live-activity uses) -- all three are None when the originating
    # position can no longer be found open, or no fresh live price is
    # available. See api/main.py's get_partial_exits().
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    pnl_status: Optional[str] = None  # "profit" | "breakeven" | "loss" -- None if unrealized_pnl is None


class PartialExitsResponse(BaseModel):
    exits: list[PartialExitResponse] = []


class CrashWatchHitResponse(BaseModel):
    symbol: str
    trigger_time: str  # when the decline first qualified today
    decline_pct: float  # negative -- trailing-15min return at trigger_time, see detectors/crash_watch.py
    volume_zscore: float
    current_price: float
    avg_traded_value: float  # liquidity proxy, Rs./min -- see detectors/liquidity_analysis.py


class CrashWatchResponse(BaseModel):
    status: str = "ok"  # "ok" | "stale_previous_day" | "market_closed"
    message: str = ""  # human-readable explanation, set whenever status != "ok"
    generated_at: str
    as_of: Optional[str] = None  # None when status != "ok"
    hits: list[CrashWatchHitResponse] = []  # sorted deepest decline first
    disclaimer: str


class LiveSessionStatusResponse(BaseModel):
    status: str  # "alive" | "stale" | "not_running"
    message: str
    last_heartbeat: str | None = None
    age_seconds: float | None = None
    open_positions: int | None = None
    symbols_polled: int | None = None


class GroupBreakdownResponse(BaseModel):
    group_name: str
    trade_count: int
    win_rate: float
    total_r: float
    avg_r: float
    reliable: bool


class EquityCurvePointResponse(BaseModel):
    time: str
    cumulative_r: float


class PortfolioReportResponse(BaseModel):
    has_data: bool
    message: str
    total_trades: int = 0
    total_r: float = 0.0
    overall_win_rate: float = 0.0
    max_drawdown_r: float = 0.0
    equity_curve: list[EquityCurvePointResponse] = []
    by_symbol: list[GroupBreakdownResponse] = []
    by_strategy: list[GroupBreakdownResponse] = []
    reasoning: list[str] = []


class CalibrationBinResponse(BaseModel):
    confidence_range_low: float
    confidence_range_high: float
    trade_count: int
    actual_win_rate: float
    avg_confidence: float
    reliable: bool


class CalibrationReportResponse(BaseModel):
    has_data: bool
    message: str
    generated_at: str | None = None
    strategy: str | None = None
    symbols_tested: int | None = None
    brier_score: float | None = None
    overall_trade_count: int | None = None
    bins: list[CalibrationBinResponse] = []
    reasoning: list[str] = []


class BacktestRequest(BaseModel):
    strategy: StrategyName
    symbols: list[str]
    start_date: date
    end_date: date


class TradeResponse(BaseModel):
    symbol: str
    direction: str
    entry_time: str
    entry_price: float
    stop_loss: float
    target: float
    exit_time: Optional[str]
    exit_price: Optional[float]
    exit_reason: Optional[str]
    pnl_points: float
    r_multiple: float
    confidence: float
    reasoning: list[str]


class PerformanceReportResponse(BaseModel):
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_points: float
    sharpe_ratio: float
    expectancy_points: float
    avg_r_multiple: float
    avg_win_points: float
    avg_loss_points: float


class BacktestResponse(BaseModel):
    trades: list[TradeResponse]
    performance: PerformanceReportResponse


class PositionSizeRequest(BaseModel):
    capital: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    stop_loss_price: float = Field(..., gt=0)
    risk_pct_per_trade: float = Field(0.01, gt=0, le=0.1)
    max_capital_pct_per_trade: float = Field(0.25, gt=0, le=1.0)
    leverage: float = Field(1.0, ge=1.0, le=20.0)


class ExplainScanRequest(BaseModel):
    symbol: str
    setup_type: str
    direction: str
    score: float
    reasoning: list[str]


class ExplainBacktestRequest(BaseModel):
    strategy: str
    symbols: list[str]
    performance: dict
    trade_count: int


class ExplainPositionSizeRequest(BaseModel):
    quantity: int
    position_value: float
    actual_risk_pct: float
    capped: bool
    reasoning: list[str]


class ExplainResponse(BaseModel):
    explanation: str


class PositionSizeResponse(BaseModel):
    quantity: int
    position_value: float
    actual_risk_amount: float
    actual_risk_pct_of_capital: float
    capped_by_exposure_limit: bool
    reasoning: list[str]
