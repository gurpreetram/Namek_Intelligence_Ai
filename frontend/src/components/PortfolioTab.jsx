import { useEffect, useState } from 'react'
import { api } from '../api'

function EquityCurveChart({ points }) {
  const width = 800
  const height = 220
  const padding = 32

  if (points.length < 2) {
    return <div className="text-dim" style={{ fontSize: 12 }}>Not enough closed trades yet for a curve.</div>
  }

  const values = points.map((p) => p.cumulative_r)
  const minR = Math.min(0, ...values)
  const maxR = Math.max(0, ...values)
  const range = maxR - minR || 1

  const x = (i) => padding + (i / (points.length - 1)) * (width - 2 * padding)
  const y = (v) => height - padding - ((v - minR) / range) * (height - 2 * padding)
  const zeroY = y(0)

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(p.cumulative_r).toFixed(1)}`).join(' ')
  const final = values[values.length - 1]
  const lineColor = final >= 0 ? 'var(--profit)' : 'var(--loss)'

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      {/* zero reference line */}
      <line x1={padding} y1={zeroY} x2={width - padding} y2={zeroY}
            stroke="var(--border)" strokeWidth="1" strokeDasharray="4 3" />
      <text x={padding} y={zeroY - 4} className="mono" fontSize="10" fill="var(--text-dim)">0R</text>
      <text x={padding} y={y(maxR) + 10} className="mono" fontSize="10" fill="var(--text-dim)">{maxR.toFixed(1)}R</text>
      <text x={padding} y={y(minR) - 2} className="mono" fontSize="10" fill="var(--text-dim)">{minR.toFixed(1)}R</text>
      <path d={pathD} fill="none" stroke={lineColor} strokeWidth="2" />
      <circle cx={x(points.length - 1)} cy={y(final)} r="3.5" fill={lineColor} />
    </svg>
  )
}

function BreakdownTable({ title, rows }) {
  return (
    <div style={{ flex: 1, minWidth: 320 }}>
      <div className="text-muted" style={{
        fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em',
      }}>
        {title}
      </div>
      {rows.length === 0 ? (
        <div className="text-dim" style={{ fontSize: 12 }}>No data.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th style={{ textAlign: 'right' }}>Trades</th>
              <th style={{ textAlign: 'right' }}>Win rate</th>
              <th style={{ textAlign: 'right' }}>Total R</th>
              <th style={{ textAlign: 'right' }}>Avg R</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((g) => (
              <tr key={g.group_name}>
                <td className="mono">
                  {g.group_name}
                  {!g.reliable && (
                    <span className="text-dim" style={{ fontSize: 10, marginLeft: 6 }}>[LOW SAMPLE]</span>
                  )}
                </td>
                <td className="mono" style={{ textAlign: 'right' }}>{g.trade_count}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{(g.win_rate * 100).toFixed(1)}%</td>
                <td className="mono" style={{ textAlign: 'right', color: g.total_r >= 0 ? 'var(--profit)' : 'var(--loss)' }}>
                  {g.total_r >= 0 ? '+' : ''}{g.total_r.toFixed(2)}R
                </td>
                <td className="mono" style={{ textAlign: 'right', color: g.avg_r >= 0 ? 'var(--profit)' : 'var(--loss)' }}>
                  {g.avg_r >= 0 ? '+' : ''}{g.avg_r.toFixed(2)}R
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function CalibrationSection() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.calibrationReport()
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  return (
    <section style={{ marginTop: 32 }}>
      <h3 style={{ fontSize: 15, marginBottom: 8 }}>Confidence Calibration -- Validated Breakout</h3>
      <p className="text-dim" style={{ marginTop: 0, marginBottom: 12, fontSize: 12 }}>
        Does a higher confidence score actually mean a higher win rate, or is the number arbitrary?
        Brier score: 0 = perfectly calibrated, 0.25 = no better than a coin flip, closer to 0 is better.
      </p>

      {error && <div className="text-loss" style={{ fontSize: 12 }}>{error}</div>}
      {!data && !error && <div className="text-muted" style={{ fontSize: 12 }}>Loading...</div>}

      {data && !data.has_data && (
        <div className="text-muted" style={{
          border: '1px solid var(--border)', background: 'var(--surface)', padding: '12px 16px', fontSize: 12,
        }}>
          {data.message}
        </div>
      )}

      {data && data.has_data && (
        <div style={{ border: '1px solid var(--border)', background: 'var(--surface)', padding: '14px 18px' }}>
          <div className="mono" style={{ fontSize: 13, marginBottom: 10 }}>
            {data.overall_trade_count} trades &middot; {data.symbols_tested} symbols &middot;
            {' '}Brier score <b style={{ color: data.brier_score < 0.2 ? 'var(--profit)' : 'var(--loss)' }}>
              {data.brier_score.toFixed(3)}
            </b>
          </div>
          <table>
            <thead>
              <tr>
                <th>Confidence bin</th>
                <th style={{ textAlign: 'right' }}>Trades</th>
                <th style={{ textAlign: 'right' }}>Avg confidence</th>
                <th style={{ textAlign: 'right' }}>Actual win rate</th>
              </tr>
            </thead>
            <tbody>
              {data.bins.map((b, i) => (
                <tr key={i}>
                  <td className="mono">
                    {b.confidence_range_low.toFixed(2)}-{b.confidence_range_high.toFixed(2)}
                    {!b.reliable && <span className="text-dim" style={{ fontSize: 10, marginLeft: 6 }}>[LOW SAMPLE]</span>}
                  </td>
                  <td className="mono" style={{ textAlign: 'right' }}>{b.trade_count}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>{b.avg_confidence.toFixed(2)}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>{(b.actual_win_rate * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.reasoning.map((r, i) => (
            <div key={i} className="text-dim" style={{ fontSize: 11, marginTop: 8 }}>- {r}</div>
          ))}
        </div>
      )}
    </section>
  )
}

export default function PortfolioTab() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    api.portfolioReport()
      .then((resp) => { setData(resp); setError(null) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 4 }}>Portfolio</h2>
      <p className="text-muted" style={{ marginTop: 0, marginBottom: 24 }}>
        Aggregate performance across the real paper-trading journal -- equity curve, per-strategy
        and per-symbol breakdowns.
      </p>

      {loading && <div className="text-muted">Loading portfolio report...</div>}
      {error && <div className="text-loss">{error}</div>}

      {data && !data.has_data && (
        <div className="text-muted" style={{
          border: '1px solid var(--border)', background: 'var(--surface-raised)',
          padding: '14px 18px', fontSize: 13,
        }}>
          {data.message}
        </div>
      )}

      {data && data.has_data && (
        <>
          <div className="mono" style={{ display: 'flex', gap: 28, marginBottom: 20, fontSize: 13 }}>
            <span>{data.total_trades} trades</span>
            <span>win rate {(data.overall_win_rate * 100).toFixed(1)}%</span>
            <span style={{ color: data.total_r >= 0 ? 'var(--profit)' : 'var(--loss)', fontWeight: 600 }}>
              {data.total_r >= 0 ? '+' : ''}{data.total_r.toFixed(2)}R total
            </span>
            <span className="text-loss">max drawdown {data.max_drawdown_r.toFixed(2)}R</span>
          </div>

          <div style={{ border: '1px solid var(--border)', background: 'var(--surface)', padding: '16px 20px', marginBottom: 24 }}>
            <div className="text-muted" style={{ fontSize: 12, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Equity Curve (cumulative R)
            </div>
            <EquityCurveChart points={data.equity_curve} />
          </div>

          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 8 }}>
            <BreakdownTable title="By Strategy" rows={data.by_strategy} />
            <BreakdownTable title="By Symbol" rows={data.by_symbol} />
          </div>

          {data.reasoning.map((r, i) => (
            <div key={i} className="text-dim" style={{ fontSize: 11, marginTop: 8 }}>- {r}</div>
          ))}
        </>
      )}

      <CalibrationSection />
    </div>
  )
}
