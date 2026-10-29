import { useEffect, useState } from 'react'
import { api } from '../api'
import ConfidenceWarning from './ConfidenceWarning'
import ExplainPanel from './ExplainPanel'

export default function ScannerTab() {
  const [availableSymbols, setAvailableSymbols] = useState([])
  const [selected, setSelected] = useState([])
  const [minScore, setMinScore] = useState(0.5)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.symbols().then(setAvailableSymbols).catch(() => {})
  }, [])

  const toggleSymbol = (sym) => {
    setSelected((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    )
  }

  const runScan = () => {
    if (selected.length === 0) return
    setLoading(true)
    setError(null)
    api.scan(selected, minScore)
      .then(setResults)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 4 }}>Scanner</h2>
      <p className="text-muted" style={{ marginTop: 0, marginBottom: 20 }}>
        Finds which cached symbols currently show a qualifying setup, ranked by score.
      </p>

      {availableSymbols.length === 0 && (
        <div style={{
          padding: 16, border: '1px solid var(--border)', background: 'var(--surface)',
          color: 'var(--text-muted)', marginBottom: 20,
        }}>
          No symbols in local cache yet. Fetch data first via
          {' '}<code style={{ color: 'var(--accent)' }}>data/fetcher_yfinance.py</code>
          {' '}or Upstox, then reload this page.
        </div>
      )}

      {availableSymbols.length > 0 && (
        <>
          <div style={{ marginBottom: 16 }}>
            <div className="text-muted" style={{ fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Symbols to scan
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {availableSymbols.map((sym) => (
                <button
                  key={sym}
                  onClick={() => toggleSymbol(sym)}
                  style={{
                    borderColor: selected.includes(sym) ? 'var(--accent)' : 'var(--border)',
                    background: selected.includes(sym) ? 'var(--accent-dim)' : 'var(--surface-raised)',
                  }}
                >
                  {sym}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', marginBottom: 24 }}>
            <label>
              <div className="text-muted" style={{ fontSize: 12, marginBottom: 6 }}>Min score</div>
              <input
                type="number" min="0" max="1" step="0.05" value={minScore}
                onChange={(e) => setMinScore(parseFloat(e.target.value))}
                style={{ width: 80 }}
              />
            </label>
            <button onClick={runScan} disabled={selected.length === 0 || loading}
                    style={{ borderColor: 'var(--accent)', color: 'var(--accent)' }}>
              {loading ? 'Scanning...' : 'Run scan'}
            </button>
          </div>
        </>
      )}

      {error && <div className="text-loss" style={{ marginBottom: 16 }}>{error}</div>}

      {results && (
        results.length === 0 ? (
          <div className="text-muted">No qualifying setups right now at this score threshold.</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {results.map((r, i) => (
              <div key={i} style={{
                border: '1px solid var(--border)', background: 'var(--surface)', padding: 16,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: 15 }}>
                    {r.symbol}
                  </span>
                  <span className="mono" style={{
                    color: r.score >= 0.75 ? 'var(--profit)' : 'var(--accent)',
                    fontWeight: 600,
                  }}>
                    {r.score.toFixed(2)}
                  </span>
                </div>
                <div className="text-muted" style={{ marginBottom: 8, fontSize: 13 }}>
                  {r.setup_type} &middot; {r.direction}
                </div>
                {/* "none"/"illiquid"/"error" are not a real strategy decision --
                    no confidence score was actually produced, so there is
                    nothing to warn about (and warning anyway reads as if
                    "none" were a strategy name, which is wrong). */}
                {!['none', 'illiquid', 'error'].includes(r.setup_type) && (
                  <ConfidenceWarning strategyName={r.setup_type} style={{ marginBottom: 10 }} />
                )}
                {r.reasoning.map((reason, j) => (
                  <div key={j} className="text-dim mono" style={{ fontSize: 12, marginBottom: 2 }}>
                    - {reason}
                  </div>
                ))}
                <ExplainPanel fetchExplanation={() => api.explainScan(r)} />
              </div>
            ))}
          </div>
        )
      )}
    </div>
  )
}
