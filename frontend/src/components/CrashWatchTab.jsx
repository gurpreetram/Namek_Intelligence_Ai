import { useEffect, useState } from 'react'
import { api } from '../api'

const CRASH_WATCH_REFRESH_MS = 30000

function formatTime(isoLike) {
  return new Date(isoLike).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function CrashHitRow({ hit }) {
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: '3px solid var(--loss)',
      background: 'var(--surface)', padding: '12px 18px',
      display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <span className="mono" style={{ fontWeight: 600, fontSize: 15, minWidth: 120 }}>{hit.symbol}</span>
        <span className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--loss)', minWidth: 70 }}>
          {hit.decline_pct.toFixed(2)}%
        </span>
        <span className="text-dim mono" style={{ fontSize: 11 }}>
          dropped at {formatTime(hit.trigger_time)}
        </span>
      </div>
      <div className="mono" style={{ display: 'flex', gap: 16, fontSize: 12, alignItems: 'center' }}>
        <span className="text-muted">
          vol z-score <b style={{ color: 'var(--accent)' }}>{hit.volume_zscore >= 0 ? '+' : ''}{hit.volume_zscore.toFixed(2)}</b>
        </span>
        <span className="text-muted">
          current <b style={{ color: 'var(--text-primary)' }}>{hit.current_price.toFixed(2)}</b>
        </span>
        <span className="text-dim" title="avg traded value/min over the prior day -- liquidity proxy">
          liquidity Rs.{(hit.avg_traded_value / 100000).toFixed(1)}L/min
        </span>
      </div>
    </div>
  )
}

export default function CrashWatchTab() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [lastPolled, setLastPolled] = useState(null)

  useEffect(() => {
    let cancelled = false
    const refresh = () => {
      api.crashWatch()
        .then((resp) => {
          if (cancelled) return
          setData(resp)
          setError(null)
          setLastPolled(new Date())
        })
        .catch((e) => { if (!cancelled) setError(e.message) })
        .finally(() => { if (!cancelled) setLoading(false) })
    }
    refresh()
    const interval = setInterval(refresh, CRASH_WATCH_REFRESH_MS)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 4 }}>Crash Watch</h2>
      <p className="text-muted" style={{ marginTop: 0, marginBottom: 20 }}>
        Stocks currently showing a rapid, large intraday decline.
      </p>

      {/* Disclaimer stays visible in every state, not just on first load -- same pattern as
          SectorsTab's Live Momentum Picks disclaimer (api/main.py's _CRASH_WATCH_DISCLAIMER). */}
      <div style={{
        border: '1px solid var(--border)', background: 'var(--surface-raised)',
        padding: '10px 14px', marginBottom: 24, fontSize: 12, color: 'var(--text-muted)',
      }}>
        {data?.disclaimer || (
          <>
            Informational only -- <b style={{ color: 'var(--accent)' }}>NOT a trade signal</b>. We tested
            both continuation (shorting the crash) and reversal (buying the dip) across 420+ historical
            events, multiple severity tiers, and time-of-day variants -- none showed a real edge. This tab
            shows WHERE a crash occurred, not what to do about it.
          </>
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
        <h3 style={{ fontSize: 16, margin: 0 }}>
          Crashing Today {data?.status === 'ok' && (
            <span className="mono text-dim" style={{ fontSize: 13, fontWeight: 400 }}>({data.hits.length})</span>
          )}
        </h3>
        <span className="text-dim mono" style={{ fontSize: 11 }}>
          {lastPolled ? `last checked ${lastPolled.toLocaleTimeString()} · auto-refresh 30s` : 'loading...'}
        </span>
      </div>

      {loading && <div className="text-muted">Loading crash watch...</div>}
      {error && <div className="text-loss">{error}</div>}

      {data && data.status !== 'ok' && (
        <div className="text-muted" style={{
          border: '1px solid var(--border)', background: 'var(--surface)', padding: '14px 18px',
        }}>
          {data.message}
        </div>
      )}

      {data && data.status === 'ok' && (
        <>
          <div className="text-dim mono" style={{ fontSize: 11, marginBottom: 16 }}>
            data as of {new Date(data.as_of).toLocaleString()}
          </div>

          {data.hits.length === 0 ? (
            <div className="text-muted" style={{
              border: '1px solid var(--border)', background: 'var(--surface)', padding: '14px 18px',
            }}>
              No crashes detected right now.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {data.hits.map((hit) => <CrashHitRow key={hit.symbol} hit={hit} />)}
            </div>
          )}
        </>
      )}
    </div>
  )
}
