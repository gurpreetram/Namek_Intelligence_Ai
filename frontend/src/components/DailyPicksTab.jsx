import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import TopMoverCard from './TopMoverCard'

const LIVE_ACTIVITY_REFRESH_MS = 30000
// Display cap only -- the backend still scans/ranks the full ~75-symbol
// watchlist (see paper_trading/watchlist.py); this just limits what's
// shown to the top 5 buy + top 5 sell per section, per the original design.
const TOP_N_PER_SIDE = 5

function parseTimestamp(s) {
  // Backend sends "2026-07-10 13:57:00+05:30" (str(pandas.Timestamp)) --
  // swap the space for 'T' so it's unambiguous ISO-8601 for Date parsing.
  return new Date(s.replace(' ', 'T'))
}

function timeAgo(isoLike) {
  const ms = Date.now() - parseTimestamp(isoLike).getTime()
  const mins = Math.round(ms / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min ago`
  const hours = Math.floor(mins / 60)
  const rem = mins % 60
  return `${hours}h ${rem}m ago`
}

function formatTime(isoLike) {
  return parseTimestamp(isoLike).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// item.trending_since is a plain ISO timestamp (see detectors/strong_trend.py) --
// no space-vs-'T' ambiguity to fix here like parseTimestamp() handles above.
function trendDuration(isoTimestamp) {
  const mins = Math.max(0, Math.round((Date.now() - new Date(isoTimestamp).getTime()) / 60000))
  const hours = Math.floor(mins / 60)
  const rem = mins % 60
  return hours > 0 ? `trending ${hours}h ${rem}m` : `trending ${rem}m`
}

const STATUS_CONFIG = {
  near_target: { label: 'trending toward target', color: 'var(--profit)' },
  near_stop: { label: 'trending toward stop', color: 'var(--loss)' },
  near_entry: { label: 'near entry, no clear lean', color: 'var(--text-muted)' },
}

// Matches paper_trading/engine.py's exit_reason values exactly
// ("target"/"stop_loss"/"eod_square_off"/"shutdown_close") -- see
// PaperTradingEngine.check_exit()/force_close().
const EXIT_REASON_CONFIG = {
  target: { label: 'TARGET HIT', color: 'var(--profit)' },
  stop_loss: { label: 'STOPPED OUT', color: 'var(--loss)' },
  eod_square_off: { label: 'EOD SQUARE-OFF', color: 'var(--text-muted)' },
  shutdown_close: { label: 'MANUAL SHUTDOWN', color: 'var(--text-muted)' },
  early_warning_momentum_reversal: { label: 'EARLY EXIT: MOMENTUM REVERSED', color: 'var(--accent)' },
  early_warning_setup_invalidated: { label: 'EARLY EXIT: SETUP INVALIDATED', color: 'var(--accent)' },
}

function PartialExitForm({ pos, onSubmit, onCancel }) {
  const [totalQty, setTotalQty] = useState('')
  const [exitQty, setExitQty] = useState('')
  const [exitPrice, setExitPrice] = useState(pos.current_price != null ? pos.current_price.toFixed(2) : '')
  const [remainingQty, setRemainingQty] = useState('')
  const [newStop, setNewStop] = useState(pos.entry_price.toFixed(2))
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  // Auto-fills remaining = total - exit whenever either changes, but stays
  // editable afterward -- a user may have already partially exited this
  // same position earlier via Kotak Neo without logging it here yet.
  const recomputeRemaining = (total, exit) => {
    const t = parseInt(total, 10)
    const e = parseInt(exit, 10)
    if (!isNaN(t) && !isNaN(e) && t >= e) setRemainingQty(String(t - e))
  }

  const submit = async () => {
    setError(null)
    const total = parseInt(totalQty, 10)
    const exit = parseInt(exitQty, 10)
    const remaining = parseInt(remainingQty, 10)
    const price = parseFloat(exitPrice)
    const stop = parseFloat(newStop)
    if (!total || !exit || isNaN(remaining) || !price || !stop) {
      setError('Fill in all fields with valid numbers.')
      return
    }
    if (exit > total) {
      setError('Exit quantity cannot exceed total quantity.')
      return
    }
    setSubmitting(true)
    try {
      await onSubmit({
        symbol: pos.symbol, direction: pos.direction,
        total_quantity: total, exit_quantity: exit, exit_price: price,
        remaining_quantity: remaining, new_stop_loss: stop,
      })
    } catch (e) {
      setError(e.message)
      setSubmitting(false)
    }
  }

  const fieldStyle = { display: 'flex', flexDirection: 'column', gap: 2 }
  const inputStyle = { width: 90, padding: '5px 8px', fontSize: 12 }

  return (
    <div style={{
      marginTop: 10, padding: '10px 12px', border: '1px dashed var(--border)',
      background: 'var(--surface)', display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      <div className="mono text-muted" style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 11 }}>
        <label style={fieldStyle}>
          total shares held
          <input type="number" min="1" value={totalQty} style={inputStyle}
            onChange={(e) => { setTotalQty(e.target.value); recomputeRemaining(e.target.value, exitQty) }} />
        </label>
        <label style={fieldStyle}>
          exited qty
          <input type="number" min="1" value={exitQty} style={inputStyle}
            onChange={(e) => { setExitQty(e.target.value); recomputeRemaining(totalQty, e.target.value) }} />
        </label>
        <label style={fieldStyle}>
          exit price
          <input type="number" step="0.05" value={exitPrice} style={inputStyle}
            onChange={(e) => setExitPrice(e.target.value)} />
        </label>
        <label style={fieldStyle}>
          remaining qty
          <input type="number" min="0" value={remainingQty} style={inputStyle}
            onChange={(e) => setRemainingQty(e.target.value)} />
        </label>
        <label style={fieldStyle}>
          new stop (breakeven+)
          <input type="number" step="0.05" value={newStop} style={inputStyle}
            onChange={(e) => setNewStop(e.target.value)} />
        </label>
      </div>
      {error && <div className="text-loss" style={{ fontSize: 11 }}>{error}</div>}
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={submit} disabled={submitting} style={{ fontSize: 11, padding: '5px 12px' }}>
          {submitting ? 'saving...' : 'save partial exit'}
        </button>
        <button onClick={onCancel} disabled={submitting} style={{ fontSize: 11, padding: '5px 12px' }}>
          cancel
        </button>
      </div>
    </div>
  )
}

function OpenPositionCard({ pos, onPartialExit }) {
  const status = pos.status_indicator ? STATUS_CONFIG[pos.status_indicator] : null
  const [showExitForm, setShowExitForm] = useState(false)
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: `3px solid var(--accent)`,
      background: 'var(--surface-raised)', padding: '14px 18px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%', background: 'var(--accent)',
            boxShadow: '0 0 6px var(--accent)', flexShrink: 0,
          }} />
          <span className="mono" style={{ fontWeight: 600, fontSize: 16 }}>{pos.symbol}</span>
          <span className="text-muted" style={{ fontSize: 12 }}>
            {pos.direction} &middot; {pos.strategy_name}
          </span>
        </div>
        <span className="text-dim mono" style={{ fontSize: 11 }}>
          entered {timeAgo(pos.entry_time)} ({formatTime(pos.entry_time)})
        </span>
      </div>

      <div className="mono" style={{ display: 'flex', gap: 20, fontSize: 13, marginBottom: 8 }}>
        <span>entry <b style={{ color: 'var(--text-primary)' }}>{pos.entry_price.toFixed(2)}</b></span>
        <span>target <b style={{ color: 'var(--profit)' }}>{pos.target.toFixed(2)}</b></span>
        <span>stop <b style={{ color: 'var(--loss)' }}>{pos.stop_loss.toFixed(2)}</b></span>
      </div>

      <div className="mono" style={{ fontSize: 12, marginBottom: pos.reasoning?.length ? 8 : 0 }}>
        {status ? (
          <span style={{ color: status.color }}>&#9679; {status.label} (last price {pos.current_price.toFixed(2)})</span>
        ) : (
          <span className="text-dim">no fresh live price available for a directional read yet</span>
        )}
      </div>

      {pos.reasoning && pos.reasoning.length > 0 && (
        <div>
          {pos.reasoning.map((r, i) => (
            <div key={i} className="text-dim mono" style={{ fontSize: 11 }}>- {r}</div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 10 }}>
        <button onClick={() => setShowExitForm((v) => !v)} style={{ fontSize: 11, padding: '5px 12px' }}>
          {showExitForm ? 'cancel' : 'Partial Exit'}
        </button>
      </div>
      {showExitForm && (
        <PartialExitForm
          pos={pos}
          onCancel={() => setShowExitForm(false)}
          onSubmit={async (payload) => { await onPartialExit(payload); setShowExitForm(false) }}
        />
      )}
    </div>
  )
}

function PartialExitRow({ ex }) {
  let color = 'var(--text-muted)'
  if (ex.pnl_status === 'profit') color = 'var(--profit)'
  else if (ex.pnl_status === 'loss') color = 'var(--loss)'
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: `3px solid ${color}`,
      background: 'var(--surface)', padding: '10px 16px',
      display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className="mono" style={{ fontWeight: 600, fontSize: 14, minWidth: 110 }}>{ex.symbol}</span>
        <span className="text-muted mono" style={{ fontSize: 12 }}>{ex.direction}</span>
        <span className="text-dim mono" style={{ fontSize: 11 }}>
          exited {ex.exit_quantity} @ {ex.exit_price.toFixed(2)}
        </span>
      </div>
      <div className="mono" style={{ display: 'flex', gap: 16, fontSize: 12, alignItems: 'center' }}>
        <span className="text-muted">remaining <b style={{ color: 'var(--text-primary)' }}>{ex.remaining_quantity}</b></span>
        <span className="text-muted">stop <b style={{ color: 'var(--accent)' }}>{ex.new_stop_loss.toFixed(2)}</b></span>
        {ex.unrealized_pnl != null ? (
          <span style={{ color, fontWeight: 600 }}>
            {ex.unrealized_pnl >= 0 ? '+' : ''}{ex.unrealized_pnl.toFixed(2)} unrealized
          </span>
        ) : (
          <span className="text-dim">P&amp;L unavailable</span>
        )}
      </div>
    </div>
  )
}

function StrongTrendRow({ item }) {
  const dirColor = item.direction === 'long' ? 'var(--profit)' : 'var(--loss)'
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: `3px solid ${dirColor}`,
      background: 'var(--surface)', padding: '10px 16px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="mono" style={{ fontWeight: 600, fontSize: 14, minWidth: 110 }}>{item.symbol}</span>
          <span className="text-muted mono" style={{ fontSize: 12 }}>{item.direction}</span>
          <span className="text-dim mono" style={{ fontSize: 11 }}>{trendDuration(item.trending_since)}</span>
        </div>
        <div className="mono" style={{ display: 'flex', gap: 16, fontSize: 12, alignItems: 'center' }}>
          <span className="text-muted">price <b style={{ color: 'var(--text-primary)' }}>{item.current_price.toFixed(2)}</b></span>
          <span style={{ color: dirColor, fontWeight: 600 }}>
            {item.move_pct_from_open >= 0 ? '+' : ''}{item.move_pct_from_open.toFixed(2)}%
          </span>
        </div>
      </div>
      <div className="text-dim mono" style={{ fontSize: 11, marginTop: 6 }}>
        {item.volume_status}
        {item.sector && ` · sector: ${item.sector}${item.sector_aligned ? ' (aligned)' : ''}`}
      </div>
    </div>
  )
}

function NearLevelRow({ item }) {
  return (
    <div style={{
      border: '1px solid var(--border)', background: 'var(--surface)',
      padding: '10px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className="mono" style={{ fontWeight: 600, fontSize: 14, minWidth: 110 }}>{item.symbol}</span>
        <span className="text-muted mono" style={{ fontSize: 12 }}>{item.direction}</span>
      </div>
      <div className="mono" style={{ display: 'flex', gap: 16, fontSize: 12, alignItems: 'center' }}>
        {item.current_price != null && item.level_price != null ? (
          <>
            <span className="text-muted">current <b style={{ color: 'var(--text-primary)' }}>{item.current_price.toFixed(2)}</b></span>
            <span className="text-muted">watch level <b style={{ color: 'var(--accent)' }}>{item.level_price.toFixed(2)}</b></span>
          </>
        ) : (
          <span className="text-dim">{item.detail}</span>
        )}
        <span style={{
          color: item.distance_atr < 0.02 ? 'var(--accent)' : 'var(--text-muted)',
          fontWeight: item.distance_atr < 0.02 ? 600 : 400,
        }}>
          {item.distance_atr.toFixed(3)}x ATR
        </span>
      </div>
    </div>
  )
}

function ConfirmedPickRow({ item }) {
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: `3px solid ${item.direction === 'long' ? 'var(--profit)' : 'var(--loss)'}`,
      background: 'var(--surface)', padding: '10px 16px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="mono" style={{ fontWeight: 600, fontSize: 14, minWidth: 110 }}>{item.symbol}</span>
          <span className="text-muted mono" style={{ fontSize: 12 }}>{item.direction} &middot; {item.detail}</span>
          {item.composite_score != null && (
            <span className="mono" style={{
              fontSize: 11, fontWeight: 700, color: 'var(--accent)',
              border: '1px solid var(--accent)', borderRadius: 2, padding: '1px 6px',
            }}>
              composite {item.composite_score.toFixed(3)}
            </span>
          )}
        </div>
        {item.entry_price != null && (
          <div className="mono" style={{ display: 'flex', gap: 16, fontSize: 12 }}>
            <span className="text-muted">entry <b style={{ color: 'var(--text-primary)' }}>{item.entry_price.toFixed(2)}</b></span>
            <span className="text-muted">target <b style={{ color: 'var(--profit)' }}>{item.target_price.toFixed(2)}</b></span>
            <span className="text-muted">stop <b style={{ color: 'var(--loss)' }}>{item.stop_loss.toFixed(2)}</b></span>
          </div>
        )}
      </div>
      {item.composite_reasoning && item.composite_reasoning.length > 0 && (
        <div style={{ marginTop: 6 }}>
          {item.composite_reasoning.map((r, i) => (
            <div key={i} className="text-dim mono" style={{ fontSize: 11 }}>- {r}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function ClosedTradeRow({ trade }) {
  const positive = trade.pnl_points > 0
  const color = positive ? 'var(--profit)' : 'var(--loss)'
  const reasonCfg = EXIT_REASON_CONFIG[trade.exit_reason] || { label: trade.exit_reason, color: 'var(--text-muted)' }
  return (
    <div style={{
      border: '1px solid var(--border-soft)', borderLeft: `2px solid ${color}`,
      background: 'var(--surface)', padding: '8px 14px', fontSize: 12,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="mono" style={{ fontWeight: 600 }}>{trade.symbol}</span>
          <span className="text-muted">{trade.direction} &middot; {trade.strategy_name}</span>
          <span className="mono" style={{
            color: reasonCfg.color, fontWeight: 700, fontSize: 11,
            border: `1px solid ${reasonCfg.color}`, borderRadius: 2, padding: '1px 6px',
          }}>
            {reasonCfg.label}
          </span>
        </div>
        <span className="text-dim mono">{formatTime(trade.exit_time)}</span>
      </div>
      <div className="mono" style={{ display: 'flex', gap: 14, marginTop: 2 }}>
        <span className="text-muted">entry {trade.entry_price.toFixed(2)}</span>
        <span className="text-muted">exit {trade.exit_price.toFixed(2)}</span>
        <span style={{ color, fontWeight: 600 }}>
          {trade.pnl_points >= 0 ? '+' : ''}{trade.pnl_points.toFixed(2)} pts ({trade.r_multiple >= 0 ? '+' : ''}{trade.r_multiple.toFixed(2)}R)
        </span>
      </div>
      {trade.exit_reasoning && trade.exit_reasoning.length > 0 && (
        <div style={{ marginTop: 4 }}>
          {trade.exit_reasoning.map((r, i) => (
            <div key={i} className="text-dim mono" style={{ fontSize: 11 }}>- {r}</div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function DailyPicksTab() {
  const [watchlist, setWatchlist] = useState(null)
  const [watchlistError, setWatchlistError] = useState(null)
  const [watchlistLoading, setWatchlistLoading] = useState(true)

  const [openPositions, setOpenPositions] = useState([])
  const [closedTrades, setClosedTrades] = useState([])
  const [activityError, setActivityError] = useState(null)
  const [lastRefreshed, setLastRefreshed] = useState(null)
  const [showCompleted, setShowCompleted] = useState(false)
  const hasAutoExpanded = useRef(false)

  const [strongTrend, setStrongTrend] = useState(null)
  const [strongTrendError, setStrongTrendError] = useState(null)

  const [topMovers, setTopMovers] = useState(null)
  const [topMoversError, setTopMoversError] = useState(null)

  const [partialExits, setPartialExits] = useState([])
  const [partialExitsError, setPartialExitsError] = useState(null)

  useEffect(() => {
    setWatchlistLoading(true)
    api.watchlist()
      .then((resp) => { setWatchlist(resp); setWatchlistError(null) })
      .catch((e) => setWatchlistError(e.message))
      .finally(() => setWatchlistLoading(false))
  }, [])

  useEffect(() => {
    let cancelled = false
    const refresh = () => {
      api.liveActivity(200)
        .then((resp) => {
          if (cancelled) return
          setOpenPositions(resp.open_positions)
          setClosedTrades(resp.closed_trades)
          setActivityError(null)
          setLastRefreshed(new Date())
          // Auto-expand "Completed Today" the first time an exit shows up,
          // so a reversal's exit_reason is visible without a manual click
          // -- but only once, so it doesn't fight a user who re-collapses
          // it afterward.
          if (resp.closed_trades.length > 0 && !hasAutoExpanded.current) {
            hasAutoExpanded.current = true
            setShowCompleted(true)
          }
        })
        .catch((e) => { if (!cancelled) setActivityError(e.message) })

      // Same 30s cadence as Active Now/Completed Today above -- both new
      // sections are cheap enough (server-side cached, see api/main.py)
      // to just ride along on this existing interval rather than adding
      // separate timers.
      api.strongTrendStocks()
        .then((resp) => { if (!cancelled) { setStrongTrend(resp); setStrongTrendError(null) } })
        .catch((e) => { if (!cancelled) setStrongTrendError(e.message) })

      api.topMovers()
        .then((resp) => { if (!cancelled) { setTopMovers(resp); setTopMoversError(null) } })
        .catch((e) => { if (!cancelled) setTopMoversError(e.message) })

      api.partialExits()
        .then((resp) => { if (!cancelled) { setPartialExits(resp.exits); setPartialExitsError(null) } })
        .catch((e) => { if (!cancelled) setPartialExitsError(e.message) })
    }
    refresh()
    const interval = setInterval(refresh, LIVE_ACTIVITY_REFRESH_MS)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  const refreshPartialExits = () => {
    api.partialExits()
      .then((resp) => { setPartialExits(resp.exits); setPartialExitsError(null) })
      .catch((e) => setPartialExitsError(e.message))
  }

  const handlePartialExit = async (payload) => {
    await api.submitPartialExit(payload)
    refreshPartialExits()
  }

  const allItems = watchlist ? watchlist.items : []
  // confirmed_pick items already arrive score-sorted (descending) within
  // each direction -- see paper_trading/watchlist.py's select_watchlist().
  const confirmedBuys = allItems
    .filter((i) => i.category === 'confirmed_pick' && i.direction === 'long')
    .slice(0, TOP_N_PER_SIDE)
  const confirmedSells = allItems
    .filter((i) => i.category === 'confirmed_pick' && i.direction === 'short')
    .slice(0, TOP_N_PER_SIDE)
  const nearLevelBuys = allItems
    .filter((i) => i.category === 'near_level' && i.direction === 'long')
    .sort((a, b) => a.distance_atr - b.distance_atr)
    .slice(0, TOP_N_PER_SIDE)
  const nearLevelSells = allItems
    .filter((i) => i.category === 'near_level' && i.direction === 'short')
    .sort((a, b) => a.distance_atr - b.distance_atr)
    .slice(0, TOP_N_PER_SIDE)

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 4 }}>Daily Picks</h2>
      <p className="text-muted" style={{ marginTop: 0, marginBottom: 28 }}>
        Observation mode only -- no real execution. Ranked by what you can act on right now.
      </p>

      {/* SECTION 1: Active Now */}
      <section style={{ marginBottom: 36 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
          <h3 style={{ fontSize: 17 }}>
            Active Now <span className="mono text-dim" style={{ fontSize: 13, fontWeight: 400 }}>({openPositions.length} open)</span>
          </h3>
          <span className="text-dim mono" style={{ fontSize: 11 }}>
            {lastRefreshed ? `updated ${lastRefreshed.toLocaleTimeString()} · auto-refresh 30s` : 'loading...'}
          </span>
        </div>

        {activityError && <div className="text-loss" style={{ marginBottom: 12 }}>{activityError}</div>}
        {!activityError && openPositions.length === 0 && (
          <div className="text-muted">No open positions right now.</div>
        )}
        {openPositions.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {openPositions.map((pos, i) => (
              <OpenPositionCard
                key={`${pos.symbol}-${pos.strategy_name}-${pos.entry_time}-${i}`}
                pos={pos}
                onPartialExit={handlePartialExit}
              />
            ))}
          </div>
        )}
      </section>

      {/* SECTION: Partial Exits Active */}
      <section style={{ marginBottom: 36 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
          <h3 style={{ fontSize: 15 }}>
            Partial Exits Active <span className="mono text-dim" style={{ fontSize: 12, fontWeight: 400 }}>
              ({partialExits.length})
            </span>
          </h3>
        </div>
        <p className="text-dim" style={{ marginTop: 0, marginBottom: 12, fontSize: 12 }}>
          Local tracking only -- no live execution. You place the order in Kotak Neo; this just remembers what's left and at what stop.
        </p>
        {partialExitsError && <div className="text-loss">{partialExitsError}</div>}
        {!partialExitsError && partialExits.length === 0 && (
          <div className="text-muted">No partial exits logged today.</div>
        )}
        {partialExits.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {partialExits.map((ex) => <PartialExitRow key={ex.id} ex={ex} />)}
          </div>
        )}
      </section>

      {/* SECTION: Top Movers Today (top 10 gainers + top 10 losers) */}
      <section style={{ marginBottom: 36 }}>
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

      {/* SECTION: Strong Trend (top 5 long + top 5 short) */}
      <section style={{ marginBottom: 36 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
          <h3 style={{ fontSize: 15 }}>
            Strong Trend <span className="mono text-dim" style={{ fontSize: 12, fontWeight: 400 }}>
              (top {strongTrend?.longs?.length ?? 0} long + top {strongTrend?.shorts?.length ?? 0} short)
            </span>
          </h3>
          {strongTrend && (
            <span className="text-dim mono" style={{ fontSize: 11 }}>
              checked {new Date(strongTrend.generated_at).toLocaleTimeString()}
            </span>
          )}
        </div>
        <p className="text-dim" style={{ marginTop: 0, marginBottom: 12, fontSize: 12 }}>
          Momentum-based informational list -- not a backtested signal. For extended hold/partial-exit candidates only.
        </p>

        {strongTrendError && <div className="text-loss">{strongTrendError}</div>}
        {strongTrend && strongTrend.status !== 'ok' && (
          <div className="text-muted">{strongTrend.message}</div>
        )}
        {strongTrend && strongTrend.status === 'ok'
          && strongTrend.longs.length === 0 && strongTrend.shorts.length === 0 && (
          <div className="text-muted">No stocks currently meeting all strong-trend criteria.</div>
        )}
        {strongTrend && strongTrend.status === 'ok'
          && (strongTrend.longs.length > 0 || strongTrend.shorts.length > 0) && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {strongTrend.longs.map((item) => <StrongTrendRow key={item.symbol} item={item} />)}
            {strongTrend.shorts.map((item) => <StrongTrendRow key={item.symbol} item={item} />)}
          </div>
        )}
      </section>

      {watchlist && watchlist.status !== 'ok' && (
        <div className="text-muted" style={{
          border: '1px solid var(--border)', background: 'var(--surface)', padding: '14px 18px',
          marginBottom: 24,
        }}>
          {watchlist.message}
        </div>
      )}

      {/* SECTION 2: Confirmed (top 5 buy + top 5 sell) */}
      <section style={{ marginBottom: 36 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
          <h3 style={{ fontSize: 15 }}>
            Confirmed <span className="mono text-dim" style={{ fontSize: 12, fontWeight: 400 }}>
              (top {confirmedBuys.length} buy + top {confirmedSells.length} sell)
            </span>
          </h3>
          {watchlist && (
            <span className="text-dim mono" style={{ fontSize: 11 }}>
              last checked {new Date(watchlist.generated_at).toLocaleTimeString()}
            </span>
          )}
        </div>
        <p className="text-dim" style={{ marginTop: 0, marginBottom: 12, fontSize: 12 }}>
          Already fired -- a strategy confirmed a setup on its own most recent bar. Target is a display-only 2.5% width, not a backtested figure; stop-loss is the real, backtested one.
        </p>

        {watchlistLoading && <div className="text-muted">Loading watchlist (full-universe scan, ~15-20s)...</div>}
        {watchlistError && <div className="text-loss">{watchlistError}</div>}
        {watchlist && watchlist.status === 'ok' && confirmedBuys.length === 0 && confirmedSells.length === 0 && (
          <div className="text-muted">No confirmed picks right now.</div>
        )}
        {watchlist && (confirmedBuys.length > 0 || confirmedSells.length > 0) && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {confirmedBuys.map((item) => <ConfirmedPickRow key={item.symbol} item={item} />)}
            {confirmedSells.map((item) => <ConfirmedPickRow key={item.symbol} item={item} />)}
          </div>
        )}
      </section>

      {/* SECTION 3: Close to Confirming (top 5 buy + top 5 sell by proximity) */}
      <section style={{ marginBottom: 36 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
          <h3 style={{ fontSize: 15 }}>
            Close to Confirming <span className="mono text-dim" style={{ fontSize: 12, fontWeight: 400 }}>
              (top {nearLevelBuys.length} buy + top {nearLevelSells.length} sell, closest to level first)
            </span>
          </h3>
          {watchlist && (
            <span className="text-dim mono" style={{ fontSize: 11 }}>
              last checked {new Date(watchlist.generated_at).toLocaleTimeString()}
            </span>
          )}
        </div>
        <p className="text-dim" style={{ marginTop: 0, marginBottom: 12, fontSize: 12 }}>
          Not yet confirmed -- price sitting near a known support/resistance level, worth watching for a possible breakout/reversal.
        </p>

        {watchlistLoading && <div className="text-muted">Loading watchlist (full-universe scan, ~15-20s)...</div>}
        {watchlistError && <div className="text-loss">{watchlistError}</div>}
        {watchlist && watchlist.status === 'ok' && nearLevelBuys.length === 0 && nearLevelSells.length === 0 && (
          <div className="text-muted">Nothing near a level right now.</div>
        )}
        {watchlist && (nearLevelBuys.length > 0 || nearLevelSells.length > 0) && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {nearLevelBuys.map((item) => <NearLevelRow key={item.symbol} item={item} />)}
            {nearLevelSells.map((item) => <NearLevelRow key={item.symbol} item={item} />)}
          </div>
        )}
      </section>

      {/* SECTION 3: Completed Today */}
      <section>
        <button
          onClick={() => setShowCompleted((v) => !v)}
          style={{
            display: 'flex', alignItems: 'center', gap: 8, background: 'transparent',
            border: 'none', padding: 0, marginBottom: showCompleted ? 12 : 0, cursor: 'pointer',
          }}
        >
          <span className="text-dim mono" style={{ fontSize: 12 }}>{showCompleted ? '▾' : '▸'}</span>
          <span className="text-muted" style={{ fontSize: 13 }}>
            Completed Today <span className="mono text-dim">({closedTrades.length})</span>
          </span>
        </button>

        {showCompleted && (
          closedTrades.length === 0 ? (
            <div className="text-dim" style={{ fontSize: 12 }}>No closed trades yet today.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {closedTrades.map((trade, i) => (
                <ClosedTradeRow key={`${trade.symbol}-${trade.strategy_name}-${trade.entry_time}-${i}`} trade={trade} />
              ))}
            </div>
          )
        )}
      </section>
    </div>
  )
}
