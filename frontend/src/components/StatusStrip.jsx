import { useEffect, useState } from 'react'
import { api } from '../api'

export default function StatusStrip() {
  const [status, setStatus] = useState('checking') // checking | live | offline

  useEffect(() => {
    let cancelled = false
    const check = () => {
      api.health()
        .then(() => { if (!cancelled) setStatus('live') })
        .catch(() => { if (!cancelled) setStatus('offline') })
    }
    check()
    const interval = setInterval(check, 8000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  const config = {
    checking: { color: 'var(--text-dim)', label: 'CONNECTING' },
    live: { color: 'var(--profit)', label: 'API LIVE' },
    offline: { color: 'var(--loss)', label: 'API OFFLINE' },
  }[status]

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '6px 16px', borderBottom: '1px solid var(--border)',
      background: 'var(--surface)', fontFamily: 'var(--font-mono)',
      fontSize: 11, letterSpacing: '0.06em',
    }}>
      <span style={{
        width: 7, height: 7, borderRadius: '50%', background: config.color,
        boxShadow: status === 'live' ? `0 0 6px ${config.color}` : 'none',
        flexShrink: 0,
      }} />
      <span style={{ color: config.color }}>{config.label}</span>
      <span style={{ color: 'var(--text-dim)' }}>
        &middot; localhost:8000
      </span>
      {status === 'offline' && (
        <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
          Run <code style={{ color: 'var(--accent)' }}>run_backend_dev.bat</code> (development) or{' '}
          <code style={{ color: 'var(--accent)' }}>run_backend_live.bat</code> (live trading days, no --reload)
          in the project folder
        </span>
      )}
    </div>
  )
}
