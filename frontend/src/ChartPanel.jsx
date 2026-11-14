import React, { useMemo, useState } from 'react'
import { signalsForCandle } from './chartSignals.mjs'

const money = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(3)
const date = value => value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—'

function aggregate(candles, minutes) {
  if (!candles?.length || minutes === 5) return candles || []
  const groups = new Map()
  candles.forEach(candle => {
    const key = Math.floor(Date.parse(candle.time) / (minutes * 60000))
    const item = groups.get(key) || { time: candle.time, open: candle.open, high: candle.high, low: candle.low, close: candle.close, volume: 0 }
    item.high = Math.max(item.high, candle.high)
    item.low = Math.min(item.low, candle.low)
    item.close = candle.close
    item.volume += candle.volume
    groups.set(key, item)
  })
  return [...groups.values()]
}

function signalLabel(signal) {
  return signal === 'BULLISH' ? 'Bullish' : signal === 'BEARISH' ? 'Bearish' : 'Neutral'
}

export default function ChartPanel({ market, predictions, models }) {
  const [timeframe, setTimeframe] = useState(Number(models?.configuration?.candle_minutes || 5))
  const [horizon, setHorizon] = useState(Number(models?.prediction_horizon_minutes || 15))
  const [range, setRange] = useState('4H')
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState(0)
  const [hover, setHover] = useState(null)
  const [selected, setSelected] = useState(null)
  const all = useMemo(() => aggregate(market?.candles, timeframe), [market, timeframe])
  const rangeBars = { '1H': Math.max(1, Math.round(60 / timeframe)), '4H': Math.max(1, Math.round(240 / timeframe)), '1D': Math.max(1, Math.round(1440 / timeframe)), ALL: all.length }
  const visibleBars = Math.max(12, Math.min(all.length, Math.round((rangeBars[range] || 48) / zoom)))
  const start = Math.max(0, all.length - visibleBars - offset)
  const shown = all.slice(start, start + visibleBars)
  const w = 1400; const h = 520; const pad = { l: 18, r: 86, t: 22, b: 34 }; const priceBottom = 390; const volumeTop = 420
  const values = shown.flatMap(candle => [candle.high, candle.low])
  const rawMin = values.length ? Math.min(...values) : 0; const rawMax = values.length ? Math.max(...values) : 1
  const margin = Math.max((rawMax - rawMin) * .08, .01); const min = rawMin - margin; const max = rawMax + margin
  const maxVolume = Math.max(...shown.map(candle => candle.volume), 1)
  const sy = value => pad.t + (max - value) / (max - min || 1) * (priceBottom - pad.t)
  const vy = value => volumeTop + (1 - value / maxVolume) * 54
  const step = (w - pad.l - pad.r) / Math.max(shown.length, 1)
  const sx = index => pad.l + index * step + step / 2
  const shift = amount => setOffset(Math.max(0, Math.min(Math.max(0, all.length - visibleBars), offset + amount)))
  const active = selected || hover
  const activeIndex = active ? shown.findIndex(candle => candle.time === active.candle.time) : -1
  const tooltipX = activeIndex < 0 ? 0 : Math.min(sx(activeIndex) + 12, w - 230)

  return <section className="chart-shell">
    <div className="chart-head"><div><span className="kicker">PRICE ACTION / HISTORICAL MODE</span><h2>COMEX Silver Futures <code>SI</code></h2></div><div className="chart-legend"><i className="up"/>Bullish <i className="down"/>Bearish <i className="neutral"/>Neutral</div></div>
    <div className="chart-toolbar"><div className="toolbar-cluster"><label>Timeframe <select value={timeframe} onChange={event => { setTimeframe(Number(event.target.value)); setOffset(0); setSelected(null) }}><option value="5">5m</option><option value="10">10m</option><option value="15">15m</option><option value="30">30m</option></select></label><label>Forecast <select value={horizon} onChange={event => setHorizon(Number(event.target.value))}><option value="10">10m</option><option value="15">15m</option><option value="30">30m</option></select></label><button onClick={() => setZoom(value => Math.min(4, value + .5))}>Zoom +</button><button onClick={() => setZoom(value => Math.max(1, value - .5))}>Zoom −</button></div><div className="toolbar-cluster"><button onClick={() => shift(-Math.max(6, Math.round(visibleBars / 3)))}>← Older</button><button onClick={() => shift(Math.max(6, Math.round(visibleBars / 3)))}>Newer →</button>{Object.keys(rangeBars).map(name => <button key={name} className={range === name ? 'selected' : ''} onClick={() => { setRange(name); setOffset(0) }}>{name}</button>)}</div></div>
    {shown.length ? <div className="chart-wrap"><svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" onMouseLeave={() => setHover(null)}>{[0,.25,.5,.75,1].map(value => <g key={value}><line className="grid" x1={pad.l} x2={w - pad.r} y1={sy(min + (max - min) * value)} y2={sy(min + (max - min) * value)}/><text x={w - pad.r + 8} y={sy(min + (max - min) * value) + 4}>{(min + (max - min) * value).toFixed(3)}</text></g>)}{shown.map((candle, index) => { const candleSignals = signalsForCandle(candle, predictions, timeframe); const bullish = candle.close >= candle.open; const bodyWidth = Math.max(4, step * .66); return <g key={candle.time} onMouseMove={() => setHover({ candle, marker: null })}><line className={bullish ? 'candle-up' : 'candle-down'} x1={sx(index)} x2={sx(index)} y1={sy(candle.high)} y2={sy(candle.low)}/><rect className={bullish ? 'body-up' : 'body-down'} x={sx(index) - bodyWidth / 2} y={Math.min(sy(candle.open), sy(candle.close))} width={bodyWidth} height={Math.max(2, Math.abs(sy(candle.open) - sy(candle.close)))}/><rect className={bullish ? 'volume-up' : 'volume-down'} x={sx(index) - bodyWidth / 2} y={vy(candle.volume)} width={bodyWidth} height={volumeTop + 54 - vy(candle.volume)}/>{candleSignals.map((marker, markerIndex) => { const markerY = marker.signal === 'BEARISH' ? sy(candle.high) - 12 - markerIndex * 13 : marker.signal === 'BULLISH' ? sy(candle.low) + 12 + markerIndex * 13 : sy(candle.close) - markerIndex * 13; return <g key={marker.id} className="signal-marker" role="button" tabIndex="0" onMouseMove={event => { event.stopPropagation(); setHover({ candle, marker }) }} onClick={event => { event.stopPropagation(); setSelected({ candle, marker }) }} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') setSelected({ candle, marker }) }}><title>{`${signalLabel(marker.signal)} · ${date(marker.timestamp)} · reference ${money(marker.referencePrice)}`}</title><circle className={`marker-${marker.signal.toLowerCase().replace('bullish', 'up').replace('bearish', 'down').replace('neutral', 'neutral')}`} cx={sx(index)} cy={markerY} r="6"/><text className="marker-glyph" x={sx(index)} y={markerY + 3}>{marker.signal === 'BULLISH' ? '↑' : marker.signal === 'BEARISH' ? '↓' : '•'}</text></g> })}</g> })}<line className="volume-divider" x1={pad.l} x2={w-pad.r} y1={volumeTop-8} y2={volumeTop-8}/><text className="volume-label" x={pad.l} y={volumeTop+12}>VOLUME</text>{active && activeIndex >= 0 && <g><line className="crosshair" x1={sx(activeIndex)} x2={sx(activeIndex)} y1={pad.t} y2={priceBottom}/><rect className="tooltip" x={tooltipX} y="30" width="220" height={active.marker ? 150 : 92}/><text className="tooltip-text" x={tooltipX + 10} y="52">{date(active.candle.time)}</text><text className="tooltip-text" x={tooltipX + 10} y="73">O {money(active.candle.open)} · H {money(active.candle.high)}</text><text className="tooltip-text" x={tooltipX + 10} y="94">L {money(active.candle.low)} · C {money(active.candle.close)}</text>{active.marker && <><text className="tooltip-text marker-tooltip" x={tooltipX + 10} y="116">{signalLabel(active.marker.signal)} · ref {money(active.marker.referencePrice)}</text><text className="tooltip-text marker-tooltip" x={tooltipX + 10} y="136">{active.marker.confidence == null ? 'Confidence unavailable' : `Confidence ${active.marker.confidenceKind ? `${active.marker.confidenceKind}: ` : ''}${(active.marker.confidence * 100).toFixed(1)}%`}</text><text className="tooltip-text marker-tooltip" x={tooltipX + 10} y="156">{active.marker.supportingEvidence[0] || 'Supporting indicators unavailable'}</text></>}</g>}</svg></div> : <div className="chart-empty">Historical candles are unavailable. Expected data/silver_5m.csv.</div>}
    <div className="chart-foot"><span>{shown.length} actual candles · source {market?.source || 'unavailable'} · {timeframe}m view</span><span>Latest: {date(market?.timestamp)} · forecast horizon {horizon}m</span></div>
  </section>
}
