import { useState } from 'react'

export default function ExplainPanel({ fetchExplanation }) {
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const run = () => {
    setLoading(true)
    setError(null)
    fetchExplanation()
      .then((r) => setExplanation(r.explanation))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }

  return (
    <div style={{ marginTop: 10 }}>
      {!explanation && (
        <button onClick={run} disabled={loading} style={{
          fontSize: 12, padding: '5px 12px',
          borderColor: 'var(--accent-dim)', color: 'var(--accent)',
        }}>
          {loading ? 'Asking AI...' : '✨ Explain this'}
        </button>
      )}
      {error && (
        <div className="text-loss" style={{ fontSize: 12, marginTop: 6 }}>{error}</div>
      )}
      {explanation && (
        <div style={{
          marginTop: 8, padding: 12, background: 'var(--surface-raised)',
          borderLeft: '2px solid var(--accent)', fontSize: 13, lineHeight: 1.6,
        }}>
          {explanation}
        </div>
      )}
    </div>
  )
}
