import { useState } from 'react'
import StatusStrip from './components/StatusStrip'
import LiveSessionIndicator from './components/LiveSessionIndicator'
import ScannerTab from './components/ScannerTab'
import BacktestTab from './components/BacktestTab'
import PositionSizeTab from './components/PositionSizeTab'
import DailyPicksTab from './components/DailyPicksTab'
import SectorsTab from './components/SectorsTab'
import PortfolioTab from './components/PortfolioTab'
import CrashWatchTab from './components/CrashWatchTab'
import BestPicksTab from './components/BestPicksTab'

const TABS = [
  { key: 'scanner', label: 'Scanner', hint: 'F1' },
  { key: 'backtest', label: 'Backtest', hint: 'F2' },
  { key: 'sizing', label: 'Position Sizing', hint: 'F3' },
  { key: 'daily-picks', label: 'Daily Picks', hint: 'F4' },
  { key: 'sectors', label: 'Sectors', hint: 'F5' },
  { key: 'portfolio', label: 'Portfolio', hint: 'F6' },
  { key: 'crash-watch', label: 'Crash Watch', hint: 'F7' },
  { key: 'best-picks', label: 'Best Picks', hint: 'F8' },
]

function App() {
  const [activeTab, setActiveTab] = useState('scanner')

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <StatusStrip />
      <LiveSessionIndicator />

      <header style={{
        padding: '20px 24px 0 24px', borderBottom: '1px solid var(--border)',
      }}>
        <h1 style={{ fontSize: 18, marginBottom: 16, letterSpacing: '0.01em' }}>
          NAMEK <span style={{ color: 'var(--accent)' }}>Intelligence</span>
        </h1>
        <nav style={{ display: 'flex', gap: 4 }}>
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                borderRadius: '2px 2px 0 0',
                borderBottom: 'none',
                background: activeTab === tab.key ? 'var(--bg)' : 'transparent',
                borderColor: activeTab === tab.key ? 'var(--border)' : 'transparent',
                color: activeTab === tab.key ? 'var(--text-primary)' : 'var(--text-muted)',
                display: 'flex', alignItems: 'center', gap: 8,
                paddingBottom: 10,
              }}
            >
              <span className="mono text-dim" style={{ fontSize: 10 }}>{tab.hint}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main style={{ flex: 1 }}>
        {/* All tabs stay mounted at all times -- only visibility toggles via CSS.
            Conditionally rendering ({activeTab === 'x' && <Tab/>}) would unmount
            the inactive tab, destroying its fetched state and any live-refresh
            polling, so switching back would blank out and re-fetch from scratch. */}
        <div style={{ display: activeTab === 'scanner' ? 'block' : 'none' }}><ScannerTab /></div>
        <div style={{ display: activeTab === 'backtest' ? 'block' : 'none' }}><BacktestTab /></div>
        <div style={{ display: activeTab === 'sizing' ? 'block' : 'none' }}><PositionSizeTab /></div>
        <div style={{ display: activeTab === 'daily-picks' ? 'block' : 'none' }}><DailyPicksTab /></div>
        <div style={{ display: activeTab === 'sectors' ? 'block' : 'none' }}><SectorsTab /></div>
        <div style={{ display: activeTab === 'portfolio' ? 'block' : 'none' }}><PortfolioTab /></div>
        <div style={{ display: activeTab === 'crash-watch' ? 'block' : 'none' }}><CrashWatchTab /></div>
        <div style={{ display: activeTab === 'best-picks' ? 'block' : 'none' }}><BestPicksTab /></div>
      </main>
    </div>
  )
}

export default App
