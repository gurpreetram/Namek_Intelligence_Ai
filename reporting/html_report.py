"""
Renders backtest and scanner results as a single static HTML file --
no server, no Flask, just open the file in a browser. Simpler to read
than scrolling a terminal, and easy to screenshot/share.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backtest.engine import Trade
from backtest.metrics import PerformanceReport
from detectors.scanner import ScanResult


@dataclass
class SymbolBreakdown:
    """One strategy's aggregate performance on one symbol -- lets you see
    which specific stocks a strategy actually has edge on, rather than only
    a blended universe-wide number that can hide per-symbol variance."""
    symbol: str
    trades: int
    win_rate: float
    total_r: float
    avg_r: float

_STYLE = """
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; background: #0f1117; color: #e6e6e6; margin: 0; padding: 24px; }
  h1 { color: #fff; border-bottom: 2px solid #333; padding-bottom: 12px; }
  h2 { color: #7dd3fc; margin-top: 36px; }
  h3 { color: #a5b4fc; margin-top: 24px; font-size: 15px; text-transform: uppercase; letter-spacing: 0.03em; }
  .caveat { background: #3a2e17; border-left: 4px solid #d9a441; padding: 12px 16px; margin: 12px 0; border-radius: 4px; color: #f0d8a8; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #2a2d36; }
  th { background: #1a1d27; color: #9ca3af; font-size: 13px; text-transform: uppercase; }
  tr:hover { background: #1a1d27; }
  .win { color: #4ade80; }
  .loss { color: #f87171; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }
  .metric-card { background: #1a1d27; border-radius: 8px; padding: 16px; }
  .metric-label { color: #9ca3af; font-size: 12px; text-transform: uppercase; }
  .metric-value { font-size: 24px; font-weight: 600; margin-top: 4px; }
  .reasoning { color: #9ca3af; font-size: 13px; margin: 2px 0 2px 20px; }
  .no-trades { color: #9ca3af; font-style: italic; padding: 12px; }
  .score-high { color: #4ade80; font-weight: 600; }
  .score-mid { color: #facc15; }
</style>
"""


def _trade_rows(trades: list[Trade]) -> str:
    if not trades:
        return '<div class="no-trades">No trades generated.</div>'
    rows = []
    for t in trades:
        pnl_class = "win" if t.pnl_points > 0 else "loss"
        exit_price_str = f"{t.exit_price:.2f}" if t.exit_price is not None else "-"
        rows.append(f"""
        <tr>
          <td>{t.symbol}</td>
          <td>{t.direction.value}</td>
          <td>{t.entry_time}</td>
          <td>{t.entry_price:.2f}</td>
          <td>{t.exit_reason or '-'}</td>
          <td>{exit_price_str}</td>
          <td class="{pnl_class}">{t.pnl_points:+.2f}</td>
          <td class="{pnl_class}">{t.r_multiple:+.2f}R</td>
        </tr>""")
    return f"""
    <table>
      <tr><th>Symbol</th><th>Dir</th><th>Entry Time</th><th>Entry</th>
          <th>Exit Reason</th><th>Exit</th><th>PnL</th><th>R-Multiple</th></tr>
      {''.join(rows)}
    </table>"""


def _metrics_grid(report: PerformanceReport) -> str:
    def card(label: str, value: str) -> str:
        return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'

    return f"""
    <div class="metrics-grid">
      {card("Trades", str(report.total_trades))}
      {card("Win Rate", f"{report.win_rate:.1%}")}
      {card("Profit Factor", f"{report.profit_factor:.2f}")}
      {card("Max Drawdown", f"{report.max_drawdown_points:.2f} pts")}
      {card("Sharpe", f"{report.sharpe_ratio:.2f}")}
      {card("Expectancy", f"{report.expectancy_points:.3f} pts")}
      {card("Avg R-Multiple", f"{report.avg_r_multiple:.2f}")}
    </div>"""


def _symbol_breakdown_rows(breakdown: list[SymbolBreakdown]) -> str:
    if not breakdown:
        return ""
    ranked = sorted(breakdown, key=lambda b: b.total_r, reverse=True)
    rows = []
    for b in ranked:
        r_class = "win" if b.total_r > 0 else "loss"
        rows.append(f"""
        <tr>
          <td>{b.symbol}</td>
          <td>{b.trades}</td>
          <td>{b.win_rate:.1%}</td>
          <td class="{r_class}">{b.total_r:+.2f}R</td>
          <td class="{r_class}">{b.avg_r:+.2f}R</td>
        </tr>""")
    return f"""
    <h3>Per-Symbol Breakdown</h3>
    <table>
      <tr><th>Symbol</th><th>Trades</th><th>Win Rate</th><th>Total R</th><th>Avg R</th></tr>
      {''.join(rows)}
    </table>"""


def _scanner_rows(results: list[ScanResult]) -> str:
    if not results:
        return '<div class="no-trades">No qualifying setups found.</div>'
    rows = []
    for r in results:
        score_class = "score-high" if r.score >= 0.75 else "score-mid"
        reasoning_html = "".join(f'<div class="reasoning">- {reason}</div>' for reason in r.reasoning)
        rows.append(f"""
        <tr>
          <td>{r.symbol}</td>
          <td class="{score_class}">{r.score:.2f}</td>
          <td>{r.setup_type}</td>
          <td>{r.direction}</td>
        </tr>
        <tr><td colspan="4">{reasoning_html}</td></tr>""")
    return f"""
    <table>
      <tr><th>Symbol</th><th>Score</th><th>Setup</th><th>Direction</th></tr>
      {''.join(rows)}
    </table>"""


def generate_report(
    output_path: Path,
    strategy_results: dict[str, tuple[list[Trade], PerformanceReport]],
    scanner_results: list[ScanResult],
    data_source_note: str = "Free yfinance data (~7 days, 1-min bars) -- sample size too small for real conclusions, mechanics check only.",
    per_symbol_breakdown: dict[str, list[SymbolBreakdown]] | None = None,
) -> None:
    per_symbol_breakdown = per_symbol_breakdown or {}
    sections = []
    for name, (trades, report) in strategy_results.items():
        sections.append(f"""
        <h2>{name}</h2>
        {_metrics_grid(report)}
        {_trade_rows(trades)}
        {_symbol_breakdown_rows(per_symbol_breakdown.get(name, []))}
        """)

    scanner_section = f"""
    <h2>Scanner (live-style scan)</h2>
    {_scanner_rows(scanner_results)}
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NAMEK Intelligence -- Backtest Report</title>{_STYLE}</head>
<body>
  <h1>NAMEK Intelligence -- Backtest Report</h1>
  <div class="caveat">⚠ {data_source_note}<br>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
  {''.join(sections)}
  {scanner_section}
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
