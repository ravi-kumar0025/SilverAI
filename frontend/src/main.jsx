import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { api } from './services/api'
import FinalSignalChartPanel from './ChartPanel.jsx'
import { normalizeFinalSignal } from './chartSignals.mjs'
import './styles.css'

const NAV = [
  ['dashboard', 'Dashboard'],
  ['predictions', 'Predictions'],
  ['analysis', 'Analysis'],
  ['evaluation', 'Evaluation'],
  ['paper', 'Paper Trading'],
  ['model', 'Model'],
  ['data', 'Diagnostics'],
  ['settings', 'Settings']
]

const money = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(3)
const percent = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(2)}%`
const date = value => value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—'
const signalName = value => ({
  BULLISH: 'UP', BEARISH: 'DOWN', UP: 'UP', DOWN: 'DOWN',
  NEUTRAL: 'NO CLEAR SIGNAL', MIXED: 'NO CLEAR SIGNAL', INSUFFICIENT_DATA: 'NO CLEAR SIGNAL'
}[value] || 'NO CLEAR SIGNAL')

function App() {
  const [route, setRoute] = useState(location.hash.slice(1) || 'dashboard')
  const [theme, setTheme] = useState(() => localStorage.getItem('silverai-theme') || 'light')
  const [state, setState] = useState({
    market: null, insights: null, status: null, models: null, prediction: null,
    history: [], metrics: null, paper: null, frozen: null, diagnostics: null
  })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const load = async () => {
    setBusy(true)
    setError('')
    const results = await Promise.allSettled([
      api.candles(1000), api.marketInsights(), api.marketStatus(), api.models(),
      api.latest().catch(() => null), api.predictionHistory(200),
      api.evaluationMetrics(), api.paperTrading(), api.frozenEvaluation(), api.diagnostics()
    ])
    const value = index => results[index].status === 'fulfilled' ? results[index].value : null
    const failures = results.filter(r => r.status === 'rejected').map(r => r.reason?.message).filter(Boolean)
    setState({
      market: value(0), insights: value(1), status: value(2), models: value(3), prediction: value(4),
      history: value(5)?.predictions || [], metrics: value(6), paper: value(7),
      frozen: value(8), diagnostics: value(9)
    })
    if (failures.length) setError(failures.join(' · '))
    setBusy(false)
  }

  useEffect(() => {
    load()
    const onHash = () => setRoute(location.hash.slice(1) || 'dashboard')
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('silverai-theme', theme)
  }, [theme])

  const latest = state.prediction?.predictions?.at(-1) || state.prediction

  const pages = {
    dashboard: <Dashboard state={state} latest={latest} busy={busy} />,
    predictions: <Predictions state={state} latest={latest} />,
    analysis: <Analysis models={state.models} latest={latest} />,
    evaluation: <Evaluation metrics={state.metrics} />,
    paper: <Paper paper={state.paper} />,
    model: <ModelInfo models={state.models} frozen={state.frozen} />,
    data: <Diagnostics state={state} />,
    settings: <Settings models={state.models} status={state.status} />
  }

  return (
    <div className="terminal">
      <header className="topbar">
        <a className="brand" href="#dashboard">
          <span className="brand-mark">S</span>
          <span>
            <b>SilverAI</b>
            <small>Market Intelligence</small>
          </span>
        </a>

        <nav>
          {NAV.map(([id, label]) => (
            <a key={id} href={`#${id}`} className={route === id ? 'active' : ''}>
              {label}
            </a>
          ))}
        </nav>

        <div className="top-tools">
          <span className="feed">
            <i className={state.status?.is_live ? 'live' : ''} />
            {state.status?.is_live ? 'Live' : 'Historical'}
          </span>
          <button className="theme" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? '☀' : '☾'}
          </button>
          <button className="refresh" onClick={load} disabled={busy}>
            {busy ? 'Updating…' : 'Refresh'}
          </button>
        </div>
      </header>

      <main>
        {error && (
          <div className="alert">
            {error}
            <button onClick={load}>Retry</button>
          </div>
        )}
        {pages[route] || pages.dashboard}
        <footer>
          <span>XAG/USD · Dukascopy spot</span>
          <span>{state.status?.is_live ? 'LIVE' : 'HISTORICAL'} · Paper trading only</span>
        </footer>
      </main>
    </div>
  )
}

function PageHead({ kicker, title, text }) {
  return (
    <div className="page-head">
      <div>
        <span className="kicker">{kicker}</span>
        <h1>{title}</h1>
        {text && <p>{text}</p>}
      </div>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <small>{label}</small>
      <b>{value ?? '—'}</b>
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="row">
      <span>{label}</span>
      <b>{value ?? '—'}</b>
    </div>
  )
}

function Panel({ title, children, className = '' }) {
  return (
    <section className={`intel-panel ${className}`}>
      <div className="panel-top">
        <span className="kicker">{title}</span>
      </div>
      {children}
    </section>
  )
}

function Empty({ children }) {
  return (
    <div className="empty">
      <span>ⓘ</span>
      <span>{children}</span>
    </div>
  )
}

function Dashboard({ state, latest, busy }) {
  return (
    <>
      <PageHead
        kicker="DASHBOARD"
        title="XAG/USD spot intelligence"
        text="XAG/USD spot with Dukascopy candles, model signals, and evaluation diagnostics."
      />
      <MarketOverview market={state.market} status={state.status} models={state.models} busy={busy} />
      <FinalSignalChartPanel market={state.market} predictions={state.history} models={state.models} />
      <MarketInsightPanel insights={state.insights} />
      <div className="below-grid">
        <SignalPanel prediction={latest} models={state.models} />
        <ScenarioPanel prediction={latest} />
      </div>
    </>
  )
}

function MarketInsightPanel({ insights }) {
  const cross = insights?.cross_market
  return (
    <Panel title="TRADER CONTEXT" className="insight-panel">
      <div className="rows">
        <Row label="Trend" value={insights?.trend?.state || 'Unavailable'} />
        <Row label="Bias" value={insights?.trend?.bias || 'No signal'} />
        <Row label="Volatility" value={insights?.volatility?.state || 'Unavailable'} />
        <Row label="Cross-market coverage" value={cross ? `${(Number(cross.coverage_ratio) * 100).toFixed(1)}%` : 'Unavailable'} />
      </div>
      <p className="panel-note">Context is derived from closed UTC candles. It is not an order recommendation.</p>
    </Panel>
  )
}

function MarketOverview({ market, status, models, busy }) {
  const s = market?.stats
  const changePositive = s?.change >= 0

  return (
    <section className="instrument-row">
      <div className="instrument-card primary">
        <div className="instrument-card-top">
          <div className="instrument-logo">Ag</div>
          <div>
            <div className="instrument-name">XAG/USD Spot</div>
            <div className="instrument-symbol">XAG/USD · {models?.configuration?.candle_minutes || 5}m</div>
          </div>
        </div>
        <div className="instrument-price">{money(s?.close)}</div>
        <div className={`instrument-change ${changePositive ? 'up' : 'down'}`}>
          {s ? `${changePositive ? '+' : ''}${s.change.toFixed(3)} (${percent(s.change_pct)})` : '—'}
        </div>
        <div className="instrument-meta">
          {status?.is_live ? 'Live data' : 'Historical'} · {busy ? 'Updating…' : date(market?.timestamp)}
        </div>
      </div>

      <div className="instrument-card">
        <div className="instrument-label">Session Open</div>
        <div className="instrument-value">{money(s?.open)}</div>
      </div>
      <div className="instrument-card">
        <div className="instrument-label">Session High</div>
        <div className="instrument-value">{money(s?.high)}</div>
      </div>
      <div className="instrument-card">
        <div className="instrument-label">Session Low</div>
        <div className="instrument-value">{money(s?.low)}</div>
      </div>
    </section>
  )
}

function SignalPanel({ prediction, models }) {
  const signal = signalName(prediction?.signal?.signal || prediction?.final_signal)
  const cls = signal === 'UP' ? 'up' : signal === 'DOWN' ? 'down' : 'neutral'
  const probabilities = prediction?.calibrated_probabilities || prediction?.class_probabilities || {}

  const expected = prediction?.predicted_return_15m
  const confidence = prediction?.confidence
  return (
    <Panel title="FINAL ANALYTICAL SIGNAL" className={`monitor ${cls}`}>
      <div className="signal">
        <div>
          <small>Primary model output</small>
          <strong>{signal}</strong>
          <span>{prediction ? expected != null ? `${percent(expected)} expected return` : `${percent(confidence)} calibrated confidence` : 'No prediction yet'}</span>
        </div>
        <span className="signal-badge">
          {signal === 'UP' ? '↑' : signal === 'DOWN' ? '↓' : '—'}
        </span>
      </div>

      <div className="rows">
        <Row label="Prediction time" value={date(prediction?.prediction_timestamp || prediction?.candle_timestamp)} />
        <Row label="Forecast horizon" value={`${models?.prediction_horizon_minutes || 15} minutes`} />
        <Row label="Calibration" value={models?.calibration_status === 'available' ? 'Calibrated' : 'Raw'} />
        <Row label="Model" value={models?.classifier_name} />
      </div>

      <div className="probabilities">
        {[['UP', probabilities.up, 'up'], ['DOWN', probabilities.down, 'down']].map(([label, value, color]) => (
          <div key={label}>
            <span>{label}</span>
            <b>{percent(value)}</b>
            <em>
              <u className={color} style={{ width: `${Math.max(0, Number(value || 0) * 100)}%` }} />
            </em>
          </div>
        ))}
      </div>
    </Panel>
  )
}

function ScenarioPanel({ prediction }) {
  const scenarios = prediction?.scenarios
  if (!scenarios) {
    const probabilities = prediction?.calibrated_probabilities || prediction?.class_probabilities || {}
    return (
      <Panel title="BINARY SPOT MODEL">
        <div className="rows">
          <Row label="Up probability" value={percent(probabilities.up)} />
          <Row label="Down probability" value={percent(probabilities.down)} />
          <Row label="Confidence" value={percent(prediction?.confidence)} />
          <Row label="Decision" value={prediction ? signalName(prediction.signal?.signal) : 'No signal'} />
        </div>
      </Panel>
    )
  }
  return (
    <Panel title="SCENARIO ANALYSIS">
      <div className="scenario-grid">
        {['Bullish', 'Bearish', 'Neutral'].map(name => {
          const item = scenarios?.[name]
          return (
            <article key={name}>
              <b>{name}</b>
              <p>{item?.expected_behavior || item?.description || 'Unavailable until a prediction is generated.'}</p>
              <small>{item?.probability == null ? 'Probability unavailable' : percent(item.probability)}</small>
              <div className="rows">
                <Row label="Supports" value={item?.supporting_evidence?.[0] || '—'} />
                <Row label="Invalidates" value={item?.invalidation_conditions?.[0] || '—'} />
              </div>
            </article>
          )
        })}
      </div>
    </Panel>
  )
}

function Predictions({ state, latest }) {
  return (
    <>
      <PageHead
        kicker="PREDICTIONS"
        title="Prediction history"
        text="Original forecasts are persisted once and evaluated after the horizon completes."
      />
      <SignalPanel prediction={latest} models={state.models} />
      <Panel title="PERSISTED FORECASTS">
        <div className="journal">
          {state.history.length ? state.history.map(row => {
            const finalSignal = normalizeFinalSignal(row)
            return (
              <div className="journal-row" key={row.prediction_id}>
                <span>{finalSignal ? signalName(finalSignal.signal) : row.candle_finalized ? 'UNAVAILABLE' : 'PENDING'}</span>
                <b>{date(row.candle_timestamp)}</b>
                <em>{row.outcome_status}{row.actual_direction ? ` · actual ${row.actual_direction}` : ''}</em>
              </div>
            )
          }) : <Empty>No persisted predictions are available.</Empty>}
        </div>
      </Panel>
    </>
  )
}

function CalibrationComparison({ prediction, models }) {
  const raw = prediction?.class_probabilities || {
    down: prediction?.probability_down,
    neutral: prediction?.probability_neutral,
    up: prediction?.probability_up
  }
  const calibrated = prediction?.calibrated_probabilities || {
    down: prediction?.calibrated_probability_down,
    neutral: prediction?.calibrated_probability_neutral,
    up: prediction?.calibrated_probability_up
  }
  const hasCalibrated = Object.values(calibrated).every(value => value != null && Number.isFinite(Number(value)))
  const rows = [['DOWN', 'down'], ['NEUTRAL', 'neutral'], ['UP', 'up']]

  return (
    <Panel title="PROBABILITY PROVENANCE">
      <div className="rows">
        <Row
          label="Calibration status"
          value={models?.calibration_status === 'available' && hasCalibrated ? 'Calibrated artifact applied' : 'Calibration unavailable; raw probabilities only'}
        />
        {rows.map(([label, key]) => (
          <Row key={key} label={`${label} raw`} value={percent(raw?.[key])} />
        ))}
        {hasCalibrated && rows.map(([label, key]) => (
          <Row key={`calibrated-${key}`} label={`${label} calibrated`} value={percent(calibrated[key])} />
        ))}
      </div>
    </Panel>
  )
}

function Analysis({ models, latest }) {
  const features = models?.feature_names || []
  return (
    <>
      <PageHead
        kicker="ANALYSIS"
        title="Technical analysis"
        text="Model-backed evidence and conditional scenarios."
      />
      <div className="analysis-hero">
        <div>
          <span className="kicker">ACTIVE CONFIGURATION</span>
          <h2>{models?.configuration?.candle_minutes || 5}-minute context / {models?.prediction_horizon_minutes || 15}-minute forecast</h2>
          <p>Feature generation uses the approved causal schema. Calibration: {models?.calibration_status || 'unavailable'}.</p>
        </div>
        <div className="analysis-count">
          <strong>{features.length}</strong>
          <span>MODEL FEATURES</span>
        </div>
      </div>
      <CalibrationComparison prediction={latest} models={models} />
      <ScenarioPanel prediction={latest} />
      <Panel title="FEATURE SCHEMA">
        <div className="feature-grid">
          {features.map(feature => <code key={feature}>{feature}</code>)}
        </div>
      </Panel>
    </>
  )
}

function MetricTable({ rows, columns }) {
  return rows?.length ? (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>{columns.map(c => <th key={c}>{c.replaceAll('_', ' ')}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.model}-${row.dataset}-${index}`}>
              {columns.map(c => (
                <td key={c}>{typeof row[c] === 'number' ? row[c].toFixed(4) : row[c] ?? '—'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ) : (
    <Empty>Evaluation reports are unavailable. Run the reproducible evaluation command to populate metrics.</Empty>
  )
}

function Evaluation({ metrics }) {
  return (
    <>
      <PageHead
        kicker="EVALUATION"
        title="Model evaluation"
        text="Metrics come from backend reports and historical records."
      />
      <div className="stats-grid">
        <Stat label="PREDICTIONS" value={metrics?.predictions_total} />
        <Stat label="COMPLETED" value={metrics?.completed_outcomes} />
        <Stat label="PENDING" value={metrics?.pending_outcomes} />
        <Stat label="ACCURACY" value={metrics?.directional_accuracy == null ? '—' : percent(metrics.directional_accuracy)} />
      </div>
      <Panel title="CLASSIFICATION AND BASELINES">
        <MetricTable
          rows={metrics?.classification_metrics}
          columns={['dataset', 'model', 'accuracy', 'balanced_accuracy', 'macro_f1', 'signal_frequency', 'neutral_coverage']}
        />
      </Panel>
      <Panel title="REGRESSION">
        <MetricTable
          rows={metrics?.regression_metrics}
          columns={['dataset', 'model', 'mae', 'rmse', 'r2']}
        />
      </Panel>
    </>
  )
}

function Paper({ paper }) {
  return (
    <>
      <PageHead
        kicker="PAPER TRADING"
        title="Paper trading"
        text="No real orders. Results use final signals, outcomes, fees and slippage."
      />
      <div className="stats-grid">
        <Stat label="NET P&L" value={paper?.net_pnl == null ? '—' : paper.net_pnl.toFixed(2)} />
        <Stat label="TRADES" value={paper?.trade_count} />
        <Stat label="WINS" value={paper?.win_count} />
        <Stat label="COSTS" value={paper?.total_costs == null ? '—' : paper.total_costs.toFixed(2)} />
      </div>
      <Panel title="COST SENSITIVITY">
        <MetricTable
          rows={(paper?.cost_sensitivity || []).map(row => ({
            costs: `${row.transaction_cost_bps}bps + ${row.slippage_bps}bps`,
            trades: row.trade_count,
            net_pnl: row.net_pnl
          }))}
          columns={['costs', 'trades', 'net_pnl']}
        />
      </Panel>
      <Panel title="ASSUMPTIONS">
        <Row label="Mode" value={paper?.historical_simulation ? 'Historical simulation' : 'Unavailable'} />
        <Row label="Execution" value="Paper only · no real orders" />
      </Panel>
    </>
  )
}

function ModelInfo({ models, frozen }) {
  return (
    <>
      <PageHead
        kicker="MODEL"
        title="Model and configuration"
        text="Active model contract supplied by the backend."
      />
      <Panel title="MODEL ARTIFACTS">
        <Row label="Classifier" value={models?.classifier_name} />
        <Row label="Regressor" value={models?.regressor_name} />
        <Row label="Config version" value={models?.configuration_version} />
        <Row label="Calibration" value={models?.calibration_status} />
        <Row label="Target" value={models?.classification_target} />
        <Row label="Horizon" value={`${models?.prediction_horizon_minutes || '—'} min / ${models?.prediction_horizon_bars || '—'} bars`} />
      </Panel>
      <Panel title="FROZEN EVALUATION">
        <Row label="Status" value={frozen?.status} />
        <Row label="Evaluation days" value={frozen?.evaluation_trading_days || 5} />
        <Row label="Integrity" value={frozen?.integrity ? (frozen.integrity.valid ? 'Valid' : 'Invalid') : 'Not started'} />
        <Row label="Live provider required" value={frozen?.live_provider_required ? 'Yes' : 'Recorded'} />
      </Panel>
    </>
  )
}

function Diagnostics({ state }) {
  return (
    <>
      <PageHead
        kicker="DIAGNOSTICS"
        title="Data quality and connection"
        text="Historical mode works without credentials."
      />
      <Panel title="PROVIDER STATUS">
        <Row label="Provider" value={state.status?.provider} />
        <Row label="Mode" value={state.status?.data_mode || state.status?.status} />
        <Row label="Instrument" value={state.status?.instrument} />
        <Row label="Last update" value={date(state.status?.last_update)} />
        <Row label="Credentials" value={state.status?.is_live ? 'Configured' : 'Not required for historical'} />
      </Panel>
      <Panel title="DIAGNOSTICS">
        <Row label="Recorded events" value={state.diagnostics?.count || 0} />
        {state.diagnostics?.items?.map((item, index) => (
          <Row key={index} label={item.kind} value={item.error} />
        ))}
      </Panel>
    </>
  )
}

function Settings({ models, status }) {
  return (
    <>
      <PageHead
        kicker="SETTINGS"
        title="Application settings"
        text="Values are read from the backend configuration."
      />
      <Panel title="ACTIVE CONFIGURATION">
        <Row label="Asset" value={models?.configuration?.instrument} />
        <Row label="Symbol" value={models?.configuration?.symbol} />
        <Row label="Candle timeframe" value={`${models?.configuration?.candle_minutes || '—'} minutes`} />
        <Row label="Forecast horizon" value={`${models?.prediction_horizon_minutes || '—'} minutes`} />
        <Row label="Timezone" value={models?.configuration?.timezone} />
        <Row label="Data mode" value={status?.is_live ? 'Live' : 'Historical'} />
      </Panel>
      <Panel title="LIVE SETUP">
        <Empty>Configure the live provider environment variables. Historical chart and evaluation remain available without credentials.</Empty>
      </Panel>
    </>
  )
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
