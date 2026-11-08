import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { api, API_BASE_URL } from './services/api'
import './styles.css'

const directionCopy = {
  UP: { label: 'UPWARD BIAS', tone: 'positive', icon: '↗' },
  DOWN: { label: 'DOWNWARD BIAS', tone: 'negative', icon: '↘' },
  NEUTRAL: { label: 'NEUTRAL BIAS', tone: 'neutral', icon: '→' },
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

function formatDate(value) {
  if (!value) return 'Awaiting prediction'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
}

function StatusDot({ active = false }) {
  return <span className={`status-dot ${active ? 'active' : ''}`} aria-hidden="true" />
}

function Skeleton({ className = '' }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />
}

function App() {
  const [health, setHealth] = useState(null)
  const [models, setModels] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState('')
  const [theme, setTheme] = useState(() => localStorage.getItem('silverai-theme') || 'dark')

  const loadDashboard = async (isRefresh = false) => {
    setError('')
    isRefresh ? setRefreshing(true) : setLoading(true)
    const results = await Promise.allSettled([api.health(), api.models(), api.latest()])
    const [healthResult, modelsResult, latestResult] = results
    if (healthResult.status === 'fulfilled') setHealth(healthResult.value)
    if (modelsResult.status === 'fulfilled') setModels(modelsResult.value)
    if (latestResult.status === 'fulfilled') setPrediction(latestResult.value)
    if (latestResult.status === 'rejected' && latestResult.reason?.message !== 'No prediction available.') {
      setError(latestResult.reason?.message || 'Unable to load the latest prediction.')
    }
    if (healthResult.status === 'rejected' && modelsResult.status === 'rejected') {
      setError('The prediction backend is unavailable. Check that the API is running and try again.')
    }
    isRefresh ? setRefreshing(false) : setLoading(false)
  }

  useEffect(() => { loadDashboard() }, [])
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('silverai-theme', theme)
  }, [theme])

  const latest = prediction?.predictions?.at(-1) || prediction
  const signal = latest?.signal
  const scenarios = latest?.scenarios || {}
  const direction = directionCopy[latest?.predicted_direction_name] || directionCopy.NEUTRAL
  const probabilities = latest?.class_probabilities || {}
  const reliability = latest?.reliability
  const scenarioList = useMemo(() => [
    ['BULLISH', scenarios.Bullish],
    ['BEARISH', scenarios.Bearish],
    ['NEUTRAL', scenarios.Neutral],
  ], [scenarios])

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">S</div>
          <div><div className="brand-name">SilverAI</div><div className="brand-subtitle">MARKET INTELLIGENCE</div></div>
        </div>
        <div className="topbar-actions">
          <div className="api-status"><StatusDot active={health?.models_available} /> <span>{health?.models_available ? 'Models online' : 'Checking API'}</span></div>
          <button className="icon-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme">{theme === 'dark' ? '☼' : '☾'}</button>
          <button className="refresh-button" onClick={() => loadDashboard(true)} disabled={refreshing}>{refreshing ? 'Refreshing…' : 'Refresh latest'} <span>↻</span></button>
        </div>
      </header>

      <main className="dashboard">
        <section className="page-heading">
          <div>
            <div className="eyebrow">COMEX · SI FUTURES</div>
            <h1>Silver market command center</h1>
            <p>Short-horizon intelligence for the next 15 minutes, built from the selected SilverAI models.</p>
          </div>
          <div className="last-updated"><span className="eyebrow">LAST PREDICTION</span><strong>{formatDate(latest?.prediction_timestamp)}</strong></div>
        </section>

        {error && <div className="alert error"><span>!</span>{error}<button onClick={() => loadDashboard(true)}>Retry</button></div>}

        <section className="overview-grid">
          <article className="card instrument-card">
            <div className="card-heading"><span className="eyebrow">INSTRUMENT</span><span className="exchange-tag">COMEX</span></div>
            <div className="instrument-title"><span className="metal-orb">Ag</span><div><h2>Silver Futures</h2><p>SI · 15-minute forecast horizon</p></div></div>
            <div className="unavailable-metric"><span className="metric-value">—</span><span>Live price unavailable</span></div>
            <div className="card-footnote">Price data is not supplied by the current prediction API.</div>
          </article>
          <article className="card forecast-card">
            <div className="card-heading"><span className="eyebrow">MODEL FORECAST</span><span className={`pill ${direction.tone}`}>{direction.icon} {direction.label}</span></div>
            {loading ? <><Skeleton className="forecast-skeleton" /><Skeleton className="small-skeleton" /></> : latest ? <><div className={`forecast-value ${direction.tone}`}>{latest.predicted_direction_name || '—'}</div><div className="forecast-return">{formatPercent(latest.predicted_return_15m)} <span>predicted return</span></div></> : <EmptyState compact title="No prediction loaded" detail="Submit engineered features to the prediction API, then refresh this view." />}
          </article>
          <article className="card status-card">
            <div className="card-heading"><span className="eyebrow">SYSTEM STATUS</span><StatusDot active={health?.models_available} /></div>
            <div className="status-line"><span>Backend</span><strong className={health?.status === 'ok' ? 'text-positive' : ''}>{health?.status === 'ok' ? 'Operational' : health?.status || 'Checking'}</strong></div>
            <div className="status-line"><span>Selected models</span><strong>{health?.models_available ? '2 loaded' : 'Unavailable'}</strong></div>
            <div className="status-line"><span>API endpoint</span><strong className="mono">{API_BASE_URL}</strong></div>
          </article>
        </section>

        <div className="section-label"><span>01</span><h2>Prediction overview</h2><div /></div>
        <section className="content-grid">
          <article className="card probability-card">
            <div className="card-heading"><div><span className="eyebrow">CLASS PROBABILITIES</span><h3>Directional conviction</h3></div><span className="data-label">CLASSIFIER OUTPUT</span></div>
            {latest ? <ProbabilityBars probabilities={probabilities} /> : <EmptyState title="Probability data unavailable" detail="Probabilities will appear when a prediction has been generated." />}
            <div className="legend-note">Probabilities describe this prediction only; they are not historical model reliability.</div>
          </article>
          <article className="card signal-card">
            <div className="card-heading"><div><span className="eyebrow">ANALYTICAL SIGNAL</span><h3>Decision context</h3></div><span className={`signal-badge ${signal?.signal?.toLowerCase() || 'empty'}`}>{signal?.signal || 'AWAITING DATA'}</span></div>
            {signal ? <><p className="signal-explanation">{signal.explanation || 'No explanation supplied.'}</p><EvidenceList title="Supporting evidence" items={signal.supporting_evidence} tone="positive" /><EvidenceList title="Conflicting evidence" items={signal.conflicting_evidence} tone="negative" /></> : <EmptyState title="No signal yet" detail="The signal is derived after the selected classifier and regressor run." />}
          </article>
        </section>

        <div className="section-label"><span>02</span><h2>Scenario analysis</h2><div /></div>
        <section className="scenario-grid">
          {scenarioList.map(([name, scenario]) => <ScenarioCard key={name} name={name} scenario={scenario} />)}
        </section>

        <div className="section-label"><span>03</span><h2>Reliability & model context</h2><div /></div>
        <section className="content-grid reliability-grid">
          <article className="card reliability-card">
            <div className="card-heading"><div><span className="eyebrow">RELIABILITY ASSESSMENT</span><h3>What the model knows</h3></div><span className={`pill ${reliability?.status === 'available' ? 'positive' : 'neutral'}`}>{reliability?.status || 'unavailable'}</span></div>
            {reliability ? <><div className="reliability-stats"><div><strong>{reliability.probabilities_available ? 'Available' : 'Unavailable'}</strong><span>Probability quality</span></div><div><strong>{reliability.models_agree ? 'Aligned' : 'Mixed'}</strong><span>Model agreement</span></div></div><EvidenceList title="Limitations" items={reliability.limitations} tone="neutral" /></> : <EmptyState title="Reliability unavailable" detail="No prediction has been returned yet." />}
          </article>
          <article className="card metadata-card">
            <div className="card-heading"><div><span className="eyebrow">MODEL METADATA</span><h3>Selected production artifacts</h3></div></div>
            <MetadataRow label="Classifier" value={models?.classifier_name} /><MetadataRow label="Regressor" value={models?.regressor_name} /><MetadataRow label="Horizon" value={models ? `${models.prediction_horizon_minutes} minutes / ${models.prediction_horizon_bars} bars` : null} /><MetadataRow label="Regression target" value={models?.regression_target} />
          </article>
        </section>

        <div className="section-label"><span>04</span><h2>Historical analysis</h2><div /></div>
        <section className="card historical-card"><div className="empty-chart"><div className="chart-grid-lines" /><div className="chart-empty-icon">⌁</div><h3>Historical series not available</h3><p>The current API exposes the latest prediction only. Forecast-versus-actual charts will appear when historical prediction records are persisted.</p></div></section>
        <footer><span>SilverAI / Analytical interface</span><span>Data and forecasts are informational only.</span></footer>
      </main>
    </div>
  )
}

function ProbabilityBars({ probabilities }) {
  const rows = [['UP', probabilities.up, 'positive'], ['NEUTRAL', probabilities.neutral, 'neutral'], ['DOWN', probabilities.down, 'negative']]
  return <div className="probability-list">{rows.map(([label, value, tone]) => <div className="probability-row" key={label}><div className="probability-label"><span className={`direction-dot ${tone}`} />{label}<strong>{formatPercent(value, 1)}</strong></div><div className="bar-track"><div className={`bar-fill ${tone}`} style={{ width: `${Math.max(0, Math.min(100, Number(value || 0) * 100))}%` }} /></div></div>)}</div>
}

function ScenarioCard({ name, scenario }) {
  const tone = name === 'BULLISH' ? 'positive' : name === 'BEARISH' ? 'negative' : 'neutral'
  return <article className={`card scenario-card ${tone}`}><div className="scenario-top"><span className="scenario-icon">{name === 'BULLISH' ? '↗' : name === 'BEARISH' ? '↘' : '→'}</span><div><span className="eyebrow">SCENARIO</span><h3>{name}</h3></div><span className="scenario-probability">{formatPercent(scenario?.probability, 1)}</span></div>{scenario ? <><p className="scenario-summary">{scenario.expected_direction_name || scenario.expected_direction || 'Direction not specified.'}</p><EvidenceList title="Supporting" items={scenario.supporting_evidence} tone="positive" /><EvidenceList title="Contradicting" items={scenario.contradicting_evidence} tone="negative" />{scenario.reliability_limitations?.length > 0 && <div className="limitation">{scenario.reliability_limitations[0]}</div>}</> : <EmptyState compact title="Awaiting scenario" detail="Scenario evidence is returned with the latest prediction." />}</article>
}

function EvidenceList({ title, items, tone }) {
  const list = Array.isArray(items) ? items.filter(Boolean) : []
  return <div className="evidence"><span className="evidence-title">{title}</span>{list.length ? <ul>{list.slice(0, 3).map((item, index) => <li className={tone} key={`${item}-${index}`}>{item}</li>)}</ul> : <p className="muted">None supplied</p>}</div>
}

function MetadataRow({ label, value }) { return <div className="metadata-row"><span>{label}</span><strong className={value?.includes?.('_') ? 'mono' : ''}>{value || '—'}</strong></div> }
function EmptyState({ title, detail, compact = false }) { return <div className={`empty-state ${compact ? 'compact' : ''}`}><div className="empty-mark">—</div><div><strong>{title}</strong><p>{detail}</p></div></div> }

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>)
