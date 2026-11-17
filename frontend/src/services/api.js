const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

async function request(path, options = {}) {
  const isGet = !options.method || options.method.toUpperCase() === 'GET'
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { Accept: 'application/json', ...options.headers },
    cache: isGet ? 'no-store' : options.cache,
    ...options,
  })
  if (!response.ok) {
    let detail = 'The SilverAI API could not complete this request.'
    try {
      const payload = await response.json()
      detail = payload.detail || detail
    } catch {
      // Keep the public error stable when the server does not return JSON.
    }
    throw new Error(detail)
  }
  return response.json()
}

export const api = {
  health: () => request('/api/health'),
  models: () => request('/api/models'),
  latest: () => request('/api/predictions/latest'),
  candles: (limit = 288, timeframe = null) => request(`/api/market/candles?limit=${limit}${timeframe ? `&timeframe_minutes=${timeframe}` : ''}`),
  marketStatus: () => request('/api/market/status'),
  liveStatus: () => request('/api/live/status'),
  marketInsights: () => request('/api/market/insights'),
  predictionHistory: (limit = 100) => request(`/api/predictions?limit=${limit}`),
  paperTrading: () => request('/api/paper-trading'),
  startPaperTrading: () => request('/api/paper-trading/start', { method: 'POST' }),
  stopPaperTrading: () => request('/api/paper-trading/stop', { method: 'POST' }),
  resetPaperTrading: () => request('/api/paper-trading/reset', { method: 'POST' }),
  clearTestingData: () => request('/api/testing/clear', { method: 'POST' }),
  evaluationMetrics: () => request('/api/evaluation/metrics'),
  diagnostics: () => request('/api/diagnostics'),
  frozenEvaluation: () => request('/api/evaluation/frozen'),
  predict: (features) => request('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  }),
}

export { API_BASE_URL }
