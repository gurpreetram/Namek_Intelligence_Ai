import { useEffect, useState } from 'react'
import { api } from '../api'
import ExplainPanel from './ExplainPanel'

// Order matters here: STRATEGIES[1] is the dropdown's default selection
// (see useState(STRATEGIES[1].value) below) -- new entries are appended
// at the end so that default (validated_breakout) doesn't silently shift.
const STRATEGIES = [
  { value: 'opening_range_breakout', label: 'Opening Range Breakout' },
  { value: 'validated_breakout', label: 'Validated Breakout' },
  { value: 'trend_continuation', label: 'Trend Continuation' },
  { value: 'reversal_detection', label: 'Reversal Detection' },
  { value: 'vwap_reclaim', label: 'VWAP Reclaim' },
  { value: 'smart_money', label: 'Smart Money Concepts' },
  { value: 'fibonacci_retracement', label: 'Fibonacci Retracement' },
  { value: 'rolling_range_breakout', label: 'Rolling Range Breakout' },
  { value: 'momentum_entry', label: 'Momentum Entry (experimental)' },
]

function MetricCard({ label, value }) {
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', padding: '12px 16px' }}>
      <div className="text-muted" style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
      <div className="mono" style={{ fontSize: 20, fontWeight: 600, marginTop: 4 }}>{value}</div>
    </div>
  )
}

export default function BacktestTab() {
  const [availableSymbols, setAvailableSymbols] = useState([])
  const [selected, setSelected] = useState([])
  const [strategy, setStrategy] = useState(STRATEGIES[1].value)
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.symbols().then(setAvailableSymbols).catch(() => {})
    const today = new Date()
    const tenDaysAgo = new Date(today.getTime() - 10 * 24 * 60 * 60 * 1000)
    setEndDate(today.toISOString().slice(0, 10))
    setStartDate(tenDaysAgo.toISOString().slice(0, 10))
  }, [])

  const toggleSymbol = (sym) => {
    setSelected((prev) => prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym])
  }

  const runBacktest = () => {
    if (selected.length === 0) return
    setLoading(true)
    setError(null)
    api.backtest(strategy, selected, startDate, endDate)
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 4 }}>Backtest</h2>
      <p className="text-muted" style={{ marginTop: 0, marginBottom: 20 }}>
        Runs a strategy against cached historical data and reports real performance metrics.
      </p>

      {availableSymbols.length === 0 && (
        <div style={{ padding: 16, border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text-muted)', marginBottom: 20 }}>
          No symbols in local cache yet. Fetch data first, then reload this page.
        </div>
      )}

      {availableSymbols.length > 0 && (
        <>
          <div style={{ marginBottom: 16 }}>
            <div className="text-muted" style={{ fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Symbols</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {availableSymbols.map((sym) => (
                <button key={sym} onClick={() => toggleSymbol(sym)} style={{
                  borderColor: selected.includes(sym) ? 'var(--accent)' : 'var(--border)',
                  background: selected.includes(sym) ? 'var(--accent-dim)' : 'var(--surface-raised)',
                }}>{sym}</button>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', marginBottom: 24, flexWrap: 'wrap' }}>
            <label>
              <div className="text-muted" style={{ fontSize: 12, marginBottom: 6 }}>Strategy</div>
              <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                {STRATEGIES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </label>
            <label>
              <div className="text-muted" style={{ fontSize: 12, marginBottom: 6 }}>Start date</div>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
            </label>
            <label>
              <div className="text-muted" style={{ fontSize: 12, marginBottom: 6 }}>End date</div>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
            </label>
            <button onClick={runBacktest} disabled={selected.length === 0 || loading}
                    style={{ borderColor: 'var(--accent)', color: 'var(--accent)' }}>
              {loading ? 'Running...' : 'Run backtest'}
            </button>
          </div>
        </>
      )}

      {error && <div className="text-loss" style={{ marginBottom: 16 }}>{error}</div>}

      {result && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, marginBottom: 24 }}>
            <MetricCard label="Trades" value={result.performance.total_trades} />
            <MetricCard label="Win rate" value={`${(result.performance.win_rate * 100).toFixed(1)}%`} />
            <MetricCard label="Profit factor" value={result.performance.profit_factor.toFixed(2)} />
            <MetricCard label="Max drawdown" value={`${result.performance.max_drawdown_points.toFixed(2)} pts`} />
            <MetricCard label="Avg R-multiple" value={result.performance.avg_r_multiple.toFixed(2)} />
            <MetricCard label="Sharpe" value={result.performance.sharpe_ratio.toFixed(2)} />
          </div>

          <ExplainPanel fetchExplanation={() =>
            api.explainBacktest(strategy, selected, result.performance, result.performance.total_trades)
          } />

          <div style={{ height: 20 }} />

          {result.trades.length === 0 ? (
            <div className="text-muted">No trades generated in this range.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>
                  <th>Reason</th><th>PnL (pts)</th><th>R</th>
                </tr>
              </thead>
              <tbody>
                {result.trades.map((t, i) => (
                  <tr key={i}>
                    <td>{t.symbol}</td>
                    <td>{t.direction}</td>
                    <td>{t.entry_price.toFixed(2)}</td>
                    <td>{t.exit_price?.toFixed(2) ?? '-'}</td>
                    <td className="text-muted">{t.exit_reason ?? '-'}</td>
                    <td className={t.pnl_points > 0 ? 'text-profit' : 'text-loss'}>
                      {t.pnl_points > 0 ? '+' : ''}{t.pnl_points.toFixed(2)}
                    </td>
                    <td className={t.r_multiple > 0 ? 'text-profit' : 'text-loss'}>
                      {t.r_multiple > 0 ? '+' : ''}{t.r_multiple.toFixed(2)}R
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  )
}
