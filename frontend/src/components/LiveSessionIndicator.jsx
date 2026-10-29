import { useEffect, useState } from 'react'
import { api } from '../api'

const POLL_MS = 15000

// Mirrors StatusStrip.jsx's poll pattern, but reports whether a live
// polling SESSION is actually alive (paper_trading/heartbeat.json), not
// whether the API server itself is reachable -- the two can disagree
// (API up, no live session running) and that gap is exactly what this
// exists to surface, so "Active Now" in Daily Picks isn't trusted blind.
export default function LiveSessionIndicator() {
  const [data, setData] = useState(null)
  const [reachable, setReachable] = useState(true)

  useEffect(() => {
    let cancelled = false
    const check = () => {
      api.liveSessionStatus()
        .then((resp) => { if (!cancelled) { setData(resp); setReachable(true) } })
        .catch(() => { if (!cancelled) setReachable(false) })
    }
    check()
    const interval = setInterval(check, POLL_MS)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  const status = !reachable ? 'unreachable' : (data?.status ?? 'checking')

  const config = {
    checking: { color: 'var(--text-dim)', label: 'CHECKING SESSION' },
    unreachable: { color: 'var(--text-dim)', label: 'SESSION STATUS UNKNOWN' },
    alive: { color: 'var(--profit)', label: 'LIVE SESSION ACTIVE' },
    stale: { color: 'var(--loss)', label: 'SESSION STALE' },
    not_running: { color: 'var(--text-muted)', label: 'NO LIVE SESSION' },
  }[status]

  return (
    <div
      title={data?.message}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '6px 16px', borderBottom: '1px solid var(--border)',
        background: 'var(--surface)', fontFamily: 'var(--font-mono)',
        fontSize: 11, letterSpacing: '0.06em',
      }}
    >
      <span style={{
        width: 7, height: 7, borderRadius: '50%', background: config.color,
        boxShadow: status === 'alive' ? `0 0 6px ${config.color}` : 'none',
        flexShrink: 0,
      }} />
      <span style={{ color: config.color }}>{config.label}</span>

      {(status === 'alive' || status === 'stale') && data?.last_heartbeat && (
        <span style={{ color: 'var(--text-dim)' }}>
          &middot; last heartbeat {new Date(data.last_heartbeat).toLocaleTimeString()}
          {typeof data.age_seconds === 'number' && ` (${data.age_seconds.toFixed(0)}s ago)`}
        </span>
      )}
      {(status === 'alive' || status === 'stale') && typeof data?.open_positions === 'number' && (
        <span style={{ color: 'var(--text-dim)' }}>
          &middot; {data.open_positions} open position{data.open_positions === 1 ? '' : 's'}
        </span>
      )}
      {status === 'not_running' && (
        <span style={{ color: 'var(--text-dim)' }}>
          &middot; run <code style={{ color: 'var(--accent)' }}>examples/run_live_session.py</code> to start polling
        </span>
      )}
    </div>
  )
}
