// Shared between DailyPicksTab.jsx (F4) and BestPicksTab.jsx (F8) -- both
// render the /top-movers gainers/losers grid, and must not silently drift
// into two different card designs for the same data.
export function volumeBadgeColor(label) {
  if (label === 'high') return 'var(--profit)'
  if (label === 'low') return 'var(--text-dim)'
  return 'var(--text-muted)'
}

export default function TopMoverCard({ mover, positive }) {
  const color = positive ? 'var(--profit)' : 'var(--loss)'
  return (
    <div style={{
      border: '1px solid var(--border)', borderLeft: `3px solid ${color}`,
      background: 'var(--surface)', padding: '10px 14px',
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <span className="mono" style={{ fontWeight: 600, fontSize: 14 }}>{mover.symbol}</span>
        <span className="text-muted mono" style={{ fontSize: 11 }}>{mover.current_price.toFixed(2)}</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
        <span className="mono" style={{ fontWeight: 700, fontSize: 18, color }}>
          {mover.move_pct_from_open >= 0 ? '+' : ''}{mover.move_pct_from_open.toFixed(2)}%
        </span>
        <span className="mono" style={{ fontSize: 10, color: volumeBadgeColor(mover.volume_label) }}>
          volume: {mover.volume_label}
        </span>
      </div>
    </div>
  )
}
