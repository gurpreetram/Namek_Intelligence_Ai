import { useEffect, useState } from 'react'
import { api } from '../api'

const LIVE_PICKS_REFRESH_MS = 30000

// Matches detectors/sector_momentum.py's classify_move_magnitude() exactly
// -- how big today's-so-far move is relative to THIS stock/sector's own
// typical full daily range, not a fixed %-of-price threshold.
const MAGNITUDE_CONFIG = {
  small: { label: 'small move', color: 'var(--text-dim)' },
  moderate: { label: 'moderate move', color: 'var(--text-muted)' },
  strong: { label: 'strong move', color: 'var(--accent)' },
  unknown: { label: 'magnitude unknown', color: 'var(--text-dim)' },
}

function MagnitudeTag({ magnitude }) {
  const cfg = MAGNITUDE_CONFIG[magnitude] || MAGNITUDE_CONFIG.unknown
  return (
    <span className="mono" style={{
      fontSize: 10, color: cfg.color, border: `1px solid ${cfg.color}`,
      borderRadius: 2, padding: '0px 5px', textTransform: 'uppercase', letterSpacing: '0.02em',
    }}>
      {cfg.label}
    </span>
  )
}

function SectorCard({ sector, referenceLabel = '% change since market open' }) {
  const positive = sector.return_pct >= 0
  const color = positive ? 'var(--profit)' : 'var(--loss)'
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: `3px solid ${color}`,
      background: 'var(--surface)', padding: '14px 18px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
        <span className="mono" style={{ fontWeight: 600, fontSize: 15, letterSpacing: '0.02em' }}>
          {sector.sector.replace(/_/g, ' ')}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <MagnitudeTag magnitude={sector.magnitude} />
          <span className="mono" style={{ fontSize: 16, fontWeight: 700, color }}>
            {positive ? '+' : ''}{sector.return_pct.toFixed(2)}%
          </span>
        </div>
      </div>
      <div className="text-dim" style={{ fontSize: 11, marginBottom: 10 }}>
        {sector.num_constituents} constituents tracked &middot; {referenceLabel}, vs this sector's own typical daily range
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {sector.top_stocks.map((s) => {
          const stockPositive = s.return_pct >= 0
          const stockCfg = MAGNITUDE_CONFIG[s.magnitude] || MAGNITUDE_CONFIG.unknown
          return (
            <div key={s.symbol} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
              <span className="mono">{s.symbol}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="text-dim mono" style={{ fontSize: 10 }}>{stockCfg.label}</span>
                <span className="mono" style={{ color: stockPositive ? 'var(--profit)' : 'var(--loss)', minWidth: 56, textAlign: 'right' }}>
                  {stockPositive ? '+' : ''}{s.return_pct.toFixed(2)}%
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function MarketBreadthSummary() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.marketBreadth().then(setData).catch((e) => setError(e.message))
  }, [])

  // Informational context, same as the rest of this tab -- fail quietly
  // rather than block the page if this one reading isn't available.
  if (error) return null
  if (!data) {
    return <div className="text-muted" style={{ marginBottom: 16, fontSize: 12 }}>Loading market breadth...</div>
  }

  if (data.status !== 'ok') {
    return (
      <div style={{
        marginBottom: 24, padding: '10px 16px', border: '1px solid var(--border)',
        background: 'var(--surface)', borderLeft: '3px solid var(--text-muted)', fontSize: 12,
      }}>
        <span className="mono text-muted" style={{ fontWeight: 600 }}>Market Breadth unavailable: </span>
        <span className="text-dim">{data.message}</span>
      </div>
    )
  }

  const color = data.interpretation === 'broadly bullish' ? 'var(--profit)'
    : data.interpretation === 'broadly bearish' ? 'var(--loss)' : 'var(--text-muted)'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 14, marginBottom: 24,
      padding: '10px 16px', border: '1px solid var(--border)', background: 'var(--surface)',
      borderLeft: `3px solid ${color}`,
    }}>
      <span className="mono" style={{ fontSize: 12, fontWeight: 600, color, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
        Market Breadth: {data.interpretation}
      </span>
      <span className="mono" style={{ fontSize: 12, color: 'var(--profit)' }}>
        {data.advancing} up ({data.pct_advancing.toFixed(0)}%)
      </span>
      <span className="text-dim">/</span>
      <span className="mono" style={{ fontSize: 12, color: 'var(--loss)' }}>
        {data.declining} down ({data.pct_declining.toFixed(0)}%)
      </span>
      <span className="text-dim mono" style={{ fontSize: 11 }}>
        {data.flat} flat
      </span>
      <span className="text-dim mono" style={{ fontSize: 11, marginLeft: 'auto' }}>
        {data.total} symbols &middot; as of {new Date(data.as_of).toLocaleString()}
      </span>
    </div>
  )
}

function LiveMomentumPicks() {
  const [picks, setPicks] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [lastPolled, setLastPolled] = useState(null)

  useEffect(() => {
    let cancelled = false
    const refresh = () => {
      api.sectorMomentumPicks()
        .then((resp) => {
          if (cancelled) return
          setPicks(resp)
          setError(null)
          setLastPolled(new Date())
        })
        .catch((e) => { if (!cancelled) setError(e.message) })
        .finally(() => { if (!cancelled) setLoading(false) })
    }
    refresh()
    const interval = setInterval(refresh, LIVE_PICKS_REFRESH_MS)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  return (
    <section style={{ marginBottom: 32 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <h3 style={{ fontSize: 16 }}>Live Momentum Picks</h3>
        <span className="text-dim mono" style={{ fontSize: 11 }}>
          {lastPolled ? `last updated ${lastPolled.toLocaleTimeString()} · auto-refresh 30s` : 'loading...'}
        </span>
      </div>

      {/* Disclaimer stays visible in every state, not just on first load -- see api/main.py's
          _SECTOR_PICKS_DISCLAIMER for why this is exploratory, not a validated signal. */}
      <div style={{
        border: '1px solid var(--border)', background: 'var(--surface-raised)',
        padding: '10px 14px', marginBottom: 16, fontSize: 12, color: 'var(--text-muted)',
      }}>
        {picks?.disclaimer || (
          <>
            Exploratory / weak signal, <b style={{ color: 'var(--accent)' }}>not a validated edge</b>.
            Backtested against 21 days of cached history -- every lookback/forward combination tested
            fell far short of a real, tradeable edge over random selection. Shown for exploration only.
          </>
        )}
      </div>

      {loading && <div className="text-muted">Loading live momentum picks...</div>}
      {error && <div className="text-loss">{error}</div>}

      {picks && picks.status === 'market_closed' && (
        <div className="text-muted" style={{
          border: '1px solid var(--border)', background: 'var(--surface)', padding: '14px 18px',
        }}>
          {picks.message}
        </div>
      )}

      {picks && picks.status === 'insufficient_data' && (
        <div className="text-muted" style={{
          border: '1px solid var(--border)', background: 'var(--surface)', padding: '14px 18px',
        }}>
          {picks.message}
        </div>
      )}

      {picks && picks.status === 'ok' && (
        <>
          <div className="text-dim mono" style={{ fontSize: 11, marginBottom: 12 }}>
            data as of {new Date(picks.data_as_of).toLocaleTimeString()} &middot;
            {' '}{picks.lookback_minutes}-min rolling lookback
          </div>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 320 }}>
              <div className="text-muted" style={{
                fontSize: 12, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.04em',
              }}>
                Up Right Now <span className="mono text-dim">({picks.up_sectors.length})</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {picks.up_sectors.length === 0
                  ? <div className="text-dim" style={{ fontSize: 12 }}>None right now.</div>
                  : picks.up_sectors.map((s) => (
                      <SectorCard key={s.sector} sector={s} referenceLabel={`% change over last ${picks.lookback_minutes} min`} />
                    ))}
              </div>
            </div>
            <div style={{ flex: 1, minWidth: 320 }}>
              <div className="text-muted" style={{
                fontSize: 12, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.04em',
              }}>
                Down Right Now <span className="mono text-dim">({picks.down_sectors.length})</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {picks.down_sectors.length === 0
                  ? <div className="text-dim" style={{ fontSize: 12 }}>None right now.</div>
                  : picks.down_sectors.map((s) => (
                      <SectorCard key={s.sector} sector={s} referenceLabel={`% change over last ${picks.lookback_minutes} min`} />
                    ))}
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  )
}

export default function SectorsTab() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    api.sectors()
      .then((resp) => { setData(resp); setError(null) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const up = data ? data.sectors.filter((s) => s.return_pct >= 0).sort((a, b) => b.return_pct - a.return_pct) : []
  const down = data ? data.sectors.filter((s) => s.return_pct < 0).sort((a, b) => a.return_pct - b.return_pct) : []

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 4 }}>Sectors</h2>
      <p className="text-muted" style={{ marginTop: 0, marginBottom: 24 }}>
        Which NSE sectors are trending today, and the stocks driving each move.
      </p>

      <MarketBreadthSummary />

      <LiveMomentumPicks />

      <h3 style={{ fontSize: 16, marginBottom: 8 }}>Today So Far</h3>
      <div style={{
        border: '1px solid var(--border)', background: 'var(--surface-raised)',
        padding: '10px 14px', marginBottom: 24, fontSize: 12, color: 'var(--text-muted)',
      }}>
        Informational / contextual only -- <b style={{ color: 'var(--accent)' }}>not a trade signal</b>.
        No entry, target, or stop-loss here; this is for identifying WHERE to look, not a validated strategy
        like Daily Picks. Coverage is limited to symbols with an official NSE sectoral-index classification
        {data && <> (<span className="mono">{data.sector_coverage_pct.toFixed(1)}%</span> of the cached universe)</>} --
        most small/mid-caps have no official sector label, which is a real, permanent limitation.
      </div>

      {loading && <div className="text-muted">Loading sector data...</div>}
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
            as of {new Date(data.as_of).toLocaleString()}
          </div>

          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 320 }}>
              <div className="text-muted" style={{
                fontSize: 12, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.04em',
              }}>
                Trending Up <span className="mono text-dim">({up.length})</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {up.length === 0
                  ? <div className="text-dim" style={{ fontSize: 12 }}>No sectors up today.</div>
                  : up.map((s) => <SectorCard key={s.sector} sector={s} />)}
              </div>
            </div>

            <div style={{ flex: 1, minWidth: 320 }}>
              <div className="text-muted" style={{
                fontSize: 12, marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.04em',
              }}>
                Trending Down <span className="mono text-dim">({down.length})</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {down.length === 0
                  ? <div className="text-dim" style={{ fontSize: 12 }}>No sectors down today.</div>
                  : down.map((s) => <SectorCard key={s.sector} sector={s} />)}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
