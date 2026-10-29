import { useState } from 'react'
import { api } from '../api'
import ExplainPanel from './ExplainPanel'

export default function PositionSizeTab() {
  const [capital, setCapital] = useState(100000)
  const [entryPrice, setEntryPrice] = useState(1300)
  const [stopLoss, setStopLoss] = useState(1290)
  const [riskPct, setRiskPct] = useState(1)
  const [leverage, setLeverage] = useState(1)
  const [maxCapitalPct, setMaxCapitalPct] = useState(25)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const calculate = () => {
    setLoading(true)
    setError(null)
    api.positionSize({
      capital,
      entry_price: entryPrice,
      stop_loss_price: stopLoss,
      risk_pct_per_trade: riskPct / 100,
      max_capital_pct_per_trade: maxCapitalPct / 100,
      leverage,
    })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  const Field = ({ label, value, onChange, step = 'any' }) => (
    <label style={{ display: 'block', marginBottom: 16 }}>
      <div className="text-muted" style={{ fontSize: 12, marginBottom: 6 }}>{label}</div>
      <input type="number" step={step} value={value}
             onChange={(e) => onChange(parseFloat(e.target.value))}
             style={{ width: '100%' }} />
    </label>
  )

  return (
    <div style={{ padding: 24, maxWidth: 640 }}>
      <h2 style={{ marginBottom: 4 }}>Position Sizing</h2>
      <p className="text-muted" style={{ marginTop: 0, marginBottom: 20 }}>
        How many shares to buy, based on how much of your capital you're willing to risk on this one trade.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Field label="Capital (Rs.)" value={capital} onChange={setCapital} step="1000" />
        <Field label="Entry price" value={entryPrice} onChange={setEntryPrice} />
        <Field label="Stop-loss price" value={stopLoss} onChange={setStopLoss} />
        <Field label="Risk per trade (%)" value={riskPct} onChange={setRiskPct} step="0.1" />
        <Field label="Leverage (broker MIS multiplier)" value={leverage} onChange={setLeverage} step="0.5" />
        <Field label="Max capital per trade (%)" value={maxCapitalPct} onChange={setMaxCapitalPct} step="1" />
      </div>

      <button onClick={calculate} disabled={loading}
              style={{ borderColor: 'var(--accent)', color: 'var(--accent)', marginBottom: 24 }}>
        {loading ? 'Calculating...' : 'Calculate position size'}
      </button>

      {error && <div className="text-loss" style={{ marginBottom: 16 }}>{error}</div>}

      {result && (
        <div style={{ border: '1px solid var(--border)', background: 'var(--surface)', padding: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
            <div>
              <div className="text-muted" style={{ fontSize: 12, textTransform: 'uppercase' }}>Quantity</div>
              <div className="mono" style={{ fontSize: 32, fontWeight: 600, color: 'var(--accent)' }}>
                {result.quantity} <span style={{ fontSize: 16, color: 'var(--text-muted)' }}>shares</span>
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div className="text-muted" style={{ fontSize: 12, textTransform: 'uppercase' }}>Position value</div>
              <div className="mono" style={{ fontSize: 20, marginTop: 4 }}>
                Rs.{result.position_value.toLocaleString('en-IN')}
              </div>
            </div>
          </div>

          {result.capped_by_exposure_limit && (
            <div style={{
              background: 'var(--accent-dim)', color: 'var(--accent)', padding: '8px 12px',
              fontSize: 12, marginBottom: 16, borderLeft: '3px solid var(--accent)',
            }}>
              Capped by exposure limit -- actual risk is below your target %, which is the safe direction to be wrong in.
            </div>
          )}

          <div className="mono" style={{ fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.8 }}>
            {result.reasoning.map((r, i) => <div key={i}>- {r}</div>)}
          </div>

          <ExplainPanel fetchExplanation={() => api.explainPositionSize({
            quantity: result.quantity,
            position_value: result.position_value,
            actual_risk_pct: result.actual_risk_pct_of_capital,
            capped: result.capped_by_exposure_limit,
            reasoning: result.reasoning,
          })} />
        </div>
      )}
    </div>
  )
}
