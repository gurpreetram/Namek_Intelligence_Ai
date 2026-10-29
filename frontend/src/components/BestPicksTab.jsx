import { useEffect, useState } from 'react'
import { api } from '../api'
import TopMoverCard, { volumeBadgeColor } from './TopMoverCard'

const BEST_PICKS_REFRESH_MS = 60000
const EARLY_MOMENTUM_REFRESH_MS = 60000 // matches /early-momentum's own 60s server-side cache TTL
const TOP_MOVERS_REFRESH_MS = 60000 // matches /top-movers' own 60s server-side cache TTL -- no point polling faster

function EarlyMomentumCard({ pick }) {
  const dirColor = pick.direction === 'long' ? 'var(--profit)' : 'var(--loss)'
  const accent = 'var(--early-momentum)'
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: `5px solid ${accent}`,
      background: 'var(--surface-raised)', padding: '22px 26px', flex: '1 1 340px', minWidth: 300,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
        <div>
          <div className="mono" style={{ fontWeight: 700, fontSize: 24 }}>{pick.symbol}</div>
          <span className="mono" style={{
            display: 'inline-block', marginTop: 4, fontSize: 11, fontWeight: 700, letterSpacing: '0.04em',
            color: dirColor, border: `1px solid ${dirColor}`, borderRadius: 2, padding: '2px 8px',
          }}>
            {pick.direction.toUpperCase()}
          </span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="mono" style={{ fontSize: 20, fontWeight: 700 }}>{pick.current_price.toFixed(2)}</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: dirColor }}>
            {pick.move_pct_from_open >= 0 ? '+' : ''}{pick.move_pct_from_open.toFixed(2)}%
          </div>
        </div>
      </div>

      <div style={{ marginBottom: 14 }}>
        <span className="mono" style={{
          fontSize: 11, fontWeight: 700, color: accent, border: `1px solid ${accent}`, borderRadius: 2, padding: '2px 8px',
        }}>
          VOL Z-SCORE {pick.volume_zscore.toFixed(2)}
        </span>
        {pick.sector && (
          <span className="text-dim mono" style={{ fontSize: 11, marginLeft: 8 }}>
            {pick.sector} {pick.sector_aligned ? '(aligned)' : ''}
          </span>
        )}
      </div>

      <div className="mono" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 13, marginBottom: 14 }}>
        <div>
          <div className="text-dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Entry</div>
          <div style={{ fontWeight: 600 }}>{pick.entry_price.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Stop Loss</div>
          <div style={{ fontWeight: 600, color: 'var(--loss)' }}>
            {pick.stop_loss.toFixed(2)} <span className="text-dim" style={{ fontSize: 11 }}>(risk {pick.risk_per_share.toFixed(2)}/share)</span>
          </div>
        </div>
        <div>
          <div className="text-dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Partial Exit (1:2)</div>
          <div style={{ fontWeight: 600, color: 'var(--profit)' }}>{pick.target_1.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Hold Rest (1:3)</div>
          <div style={{ fontWeight: 600, color: 'var(--profit)' }}>{pick.target_2.toFixed(2)}</div>
        </div>
      </div>

      <div style={{ marginBottom: 14 }}>
        {pick.reasoning.map((r, i) => (
          <div key={i} className="text-dim mono" style={{ fontSize: 11 }}>- {r}</div>
        ))}
      </div>

      <div className="text-muted" style={{
        fontSize: 12, borderTop: '1px solid var(--border)', paddingTop: 10,
      }}>
        Place order manually in Kotak Neo -- this system does not execute trades.
      </div>
    </div>
  )
}

function BestPickCard({ pick }) {
  const dirColor = pick.direction === 'long' ? 'var(--profit)' : 'var(--loss)'
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: `5px solid ${dirColor}`,
      background: 'var(--surface-raised)', padding: '22px 26px', flex: '1 1 380px', minWidth: 320,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
        <div>
          <div className="mono" style={{ fontWeight: 700, fontSize: 26 }}>{pick.symbol}</div>
          <span className="mono" style={{
            display: 'inline-block', marginTop: 4, fontSize: 11, fontWeight: 700, letterSpacing: '0.04em',
            color: dirColor, border: `1px solid ${dirColor}`, borderRadius: 2, padding: '2px 8px',
          }}>
            {pick.direction.toUpperCase()}
          </span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="mono" style={{ fontSize: 22, fontWeight: 700 }}>{pick.current_price.toFixed(2)}</div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: dirColor }}>
            {pick.move_pct_from_open >= 0 ? '+' : ''}{pick.move_pct_from_open.toFixed(2)}%
          </div>
        </div>
      </div>

      <div style={{ marginBottom: 14 }}>
        <span className="mono" style={{
          fontSize: 11, fontWeight: 700, color: volumeBadgeColor(pick.volume_label),
          border: `1px solid ${volumeBadgeColor(pick.volume_label)}`, borderRadius: 2, padding: '2px 8px',
        }}>
          VOLUME: {pick.volume_label.toUpperCase()}
        </span>
        <span className="text-dim mono" style={{ fontSize: 11, marginLeft: 8 }}>
          z-score {pick.volume_zscore.toFixed(2)}
        </span>
      </div>

      <div className="mono" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 13, marginBottom: 14 }}>
        <div>
          <div className="text-dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Entry</div>
          <div style={{ fontWeight: 600 }}>{pick.entry_price.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Stop Loss</div>
          <div style={{ fontWeight: 600, color: 'var(--loss)' }}>
            {pick.stop_loss.toFixed(2)} <span className="text-dim" style={{ fontSize: 11 }}>(risk {pick.risk_per_share.toFixed(2)}/share)</span>
          </div>
        </div>
        <div>
          <div className="text-dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Partial Exit (1:2)</div>
          <div style={{ fontWeight: 600, color: 'var(--profit)' }}>{pick.target_1.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Hold Rest (1:3)</div>
          <div style={{ fontWeight: 600, color: 'var(--profit)' }}>{pick.target_2.toFixed(2)}</div>
        </div>
      </div>

      <div style={{ marginBottom: 14 }}>
        {pick.reasoning.map((r, i) => (
          <div key={i} className="text-dim mono" style={{ fontSize: 11 }}>- {r}</div>
        ))}
      </div>

      <div className="text-muted" style={{
        fontSize: 12, borderTop: '1px solid var(--border)', paddingTop: 10,
      }}>
        Place order manually in Kotak Neo -- this system does not execute trades.
      </div>
    </div>
  )
}

export default function BestPicksTab() {
  const [bestPicks, setBestPicks] = useState(null)
  const [bestPicksError, setBestPicksError] = useState(null)
  const [earlyMomentum, setEarlyMomentum] = useState(null)
  const [earlyMomentumError, setEarlyMomentumError] = useState(null)
  const [topMovers, setTopMovers] = useState(null)
  const [topMoversError, setTopMoversError] = useState(null)

  useEffect(() => {
    let cancelled = false
    const refresh = () => {
      api.bestPicks()
        .then((resp) => { if (!cancelled) { setBestPicks(resp); setBestPicksError(null) } })
        .catch((e) => { if (!cancelled) setBestPicksError(e.message) })
    }
    refresh()
    const interval = setInterval(refresh, BEST_PICKS_REFRESH_MS)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  useEffect(() => {
    let cancelled = false
    const refresh = () => {
      api.earlyMomentum()
        .then((resp) => { if (!cancelled) { setEarlyMomentum(resp); setEarlyMomentumError(null) } })
        .catch((e) => { if (!cancelled) setEarlyMomentumError(e.message) })
    }
    refresh()
    const interval = setInterval(refresh, EARLY_MOMENTUM_REFRESH_MS)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  useEffect(() => {
    let cancelled = false
    const refresh = () => {
      api.topMovers()
        .then((resp) => { if (!cancelled) { setTopMovers(resp); setTopMoversError(null) } })
        .catch((e) => { if (!cancelled) setTopMoversError(e.message) })
    }
    refresh()
    const interval = setInterval(refresh, TOP_MOVERS_REFRESH_MS)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 4 }}>Best Picks</h2>
      <div style={{
        border: '1px solid var(--accent-dim)', background: 'var(--surface)', color: 'var(--text-muted)',
        padding: '10px 16px', marginBottom: 28, fontSize: 12,
      }}>
        {bestPicks?.disclaimer
          ?? 'Momentum-based picks -- not backtested. Observation only. Place orders manually in Kotak Neo. Past moves do not guarantee future performance.'}
      </div>

      {/* EARLY MOMENTUM SECTION: catch the move before it's already happened -- above Best 2 Picks */}
      <section style={{ marginBottom: 40 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
          <h3 style={{ fontSize: 19, color: 'var(--early-momentum)' }}>Early Momentum (Catch Before The Move)</h3>
          {earlyMomentum && (
            <span className="text-dim mono" style={{ fontSize: 11 }}>
              checked {new Date(earlyMomentum.generated_at).toLocaleTimeString()} &middot; auto-refresh 60s
            </span>
          )}
        </div>
        <div style={{
          border: '1px solid var(--early-momentum-dim)', background: 'var(--surface)', color: 'var(--text-muted)',
          padding: '10px 16px', marginBottom: 14, fontSize: 12,
        }}>
          {earlyMomentum?.disclaimer
            ?? 'Early entry signal -- move is just starting. Higher risk than confirmed picks. Not backtested yet.'}
        </div>

        {earlyMomentumError && <div className="text-loss">{earlyMomentumError}</div>}
        {earlyMomentum && earlyMomentum.status !== 'ok' && (
          <div className="text-muted">{earlyMomentum.message}</div>
        )}
        {earlyMomentum && earlyMomentum.status === 'ok' && earlyMomentum.picks.length === 0 && (
          <div className="text-muted">No stocks currently meeting all early-momentum criteria.</div>
        )}
        {earlyMomentum && earlyMomentum.status === 'ok' && earlyMomentum.picks.length > 0 && (
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            {earlyMomentum.picks.map((pick) => <EarlyMomentumCard key={pick.symbol} pick={pick} />)}
          </div>
        )}
      </section>

      {/* TOP SECTION: Today's Best 2 Picks */}
      <section style={{ marginBottom: 40 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 14 }}>
          <h3 style={{ fontSize: 19 }}>Today&apos;s Best 2 Picks</h3>
          {bestPicks && (
            <span className="text-dim mono" style={{ fontSize: 11 }}>
              checked {new Date(bestPicks.generated_at).toLocaleTimeString()} &middot; auto-refresh 60s
            </span>
          )}
        </div>

        {bestPicksError && <div className="text-loss">{bestPicksError}</div>}
        {bestPicks && bestPicks.status !== 'ok' && (
          <div className="text-muted">{bestPicks.message}</div>
        )}
        {bestPicks && bestPicks.status === 'ok' && bestPicks.picks.length === 0 && (
          <div className="text-muted">No stocks currently meeting all criteria.</div>
        )}
        {bestPicks && bestPicks.status === 'ok' && bestPicks.picks.length > 0 && (
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            {bestPicks.picks.map((pick) => <BestPickCard key={pick.symbol} pick={pick} />)}
          </div>
        )}
      </section>

      {/* BOTTOM SECTION: Top Movers Today (same data/design as the F4 Daily Picks section) */}
      <section>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
          <h3 style={{ fontSize: 15 }}>Top Movers Today</h3>
          {topMovers && (
            <span className="text-dim mono" style={{ fontSize: 11 }}>
              checked {new Date(topMovers.generated_at).toLocaleTimeString()}
            </span>
          )}
        </div>
        <p className="text-dim" style={{ marginTop: 0, marginBottom: 12, fontSize: 12 }}>
          Price move from today's open. No strategy signal -- for awareness only. Check volume before acting.
        </p>

        {topMoversError && <div className="text-loss">{topMoversError}</div>}
        {topMovers && topMovers.status !== 'ok' && (
          <div className="text-muted">{topMovers.message}</div>
        )}
        {topMovers && topMovers.status === 'ok'
          && topMovers.gainers.length === 0 && topMovers.losers.length === 0 && (
          <div className="text-muted">No movers data available right now.</div>
        )}
        {topMovers && topMovers.status === 'ok'
          && (topMovers.gainers.length > 0 || topMovers.losers.length > 0) && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <div className="text-profit mono" style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                GAINERS
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {topMovers.gainers.map((m) => <TopMoverCard key={m.symbol} mover={m} positive />)}
              </div>
            </div>
            <div>
              <div className="text-loss mono" style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                LOSERS
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {topMovers.losers.map((m) => <TopMoverCard key={m.symbol} mover={m} positive={false} />)}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
