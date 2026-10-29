const BASE_URL = 'http://localhost:8000'

async function request(path, options = {}) {
  const resp = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed: ${resp.status}`)
  }
  return resp.json()
}

export const api = {
  health: () => request('/health'),
  symbols: () => request('/symbols'),
  watchlist: () => request('/watchlist'),
  liveActivity: (limit) => request(`/live-activity?limit=${limit ?? 100}`),
  strongTrendStocks: () => request('/strong-trend-stocks'),
  topMovers: () => request('/top-movers'),
  bestPicks: () => request('/best-picks'),
  earlyMomentum: () => request('/early-momentum'),
  partialExits: () => request('/partial-exits'),
  submitPartialExit: (payload) =>
    request('/partial-exit', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  liveSessionStatus: () => request('/live-session-status'),
  sectors: () => request('/sectors'),
  marketBreadth: () => request('/market-breadth'),
  sectorMomentumPicks: () => request('/sector-momentum-picks'),
  crashWatch: () => request('/crash-watch'),
  portfolioReport: () => request('/portfolio-report'),
  calibrationReport: () => request('/calibration-report'),
  scan: (symbols, minScore) =>
    request('/scan', {
      method: 'POST',
      body: JSON.stringify({ symbols, min_score: minScore }),
    }),
  backtest: (strategy, symbols, startDate, endDate) =>
    request('/backtest', {
      method: 'POST',
      body: JSON.stringify({
        strategy, symbols, start_date: startDate, end_date: endDate,
      }),
    }),
  positionSize: (params) =>
    request('/position-size', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  explainScan: (result) =>
    request('/explain/scan', {
      method: 'POST',
      body: JSON.stringify(result),
    }),
  explainBacktest: (strategy, symbols, performance, tradeCount) =>
    request('/explain/backtest', {
      method: 'POST',
      body: JSON.stringify({ strategy, symbols, performance, trade_count: tradeCount }),
    }),
  explainPositionSize: (result) =>
    request('/explain/position-size', {
      method: 'POST',
      body: JSON.stringify(result),
    }),
}
