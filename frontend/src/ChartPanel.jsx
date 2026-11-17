import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createSeriesMarkers } from './chartSignals.mjs'

const money = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(3)
const date = value => value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—'

function aggregate(candles, minutes) {
  if (!candles?.length || minutes === 5) return candles || []
  const groups = new Map()
  candles.forEach(candle => {
    const key = Math.floor(Date.parse(candle.time) / (minutes * 60000))
    const item = groups.get(key) || { time: candle.time, open: candle.open, high: candle.high, low: candle.low, close: candle.close }
    item.high = Math.max(item.high, candle.high)
    item.low = Math.min(item.low, candle.low)
    item.close = candle.close
    groups.set(key, item)
  })
  return [...groups.values()]
}

function signalLabel(signal) {
  return signal === 'BULLISH' ? 'Bullish' : signal === 'BEARISH' ? 'Bearish' : 'Neutral'
}

export default function ChartPanel({ market, predictions, latestPrediction, models, mode = 'historical' }) {
  const [timeframe, setTimeframe] = useState(Number(models?.configuration?.candle_minutes || 0))
  const [range, setRange] = useState('4H')
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState(0)
  const [hover, setHover] = useState(null)
  const [selected, setSelected] = useState(null)
  const dragRef = useRef(null)

  useEffect(() => {
    if (models?.configuration?.candle_minutes) setTimeframe(Number(models.configuration.candle_minutes))
  }, [models])

  const all = useMemo(() => {
    const candles = [...(market?.candles || [])]
    if (mode === 'live' && market?.forming_candle) candles.push(market.forming_candle)
    return aggregate(candles, timeframe)
  }, [market, timeframe, mode])
  const chartPredictions = useMemo(() => {
    const records = [...(predictions || [])]
    if (latestPrediction) records.push(latestPrediction)
    const seen = new Set()
    return records.filter(record => {
      const key = record?.prediction_id || `${record?.candle_timestamp || ''}:${record?.final_signal || ''}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
  }, [predictions, latestPrediction])
  const rangeBars = { '1H': Math.max(1, Math.round(60 / timeframe)), '4H': Math.max(1, Math.round(240 / timeframe)), '1D': Math.max(1, Math.round(1440 / timeframe)), ALL: all.length }
  const visibleBars = Math.max(12, Math.min(all.length, Math.round((rangeBars[range] || 48) / zoom)))
  const start = Math.max(0, all.length - visibleBars - offset)
  const shown = all.slice(start, start + visibleBars)

  const w = 1400
  const h = 520
  const pad = { l: 18, r: 86, t: 22, b: 34 }
  const priceBottom = 470

  const values = shown.flatMap(candle => [candle.high, candle.low])
  const rawMin = values.length ? Math.min(...values) : 0
  const rawMax = values.length ? Math.max(...values) : 1
  const margin = Math.max((rawMax - rawMin) * .08, .01)
  const min = rawMin - margin
  const max = rawMax + margin

  const sy = value => pad.t + (max - value) / (max - min || 1) * (priceBottom - pad.t)
  const step = (w - pad.l - pad.r) / Math.max(shown.length, 1)
  const sx = index => pad.l + index * step + step / 2

  const shift = amount => setOffset(Math.max(0, Math.min(Math.max(0, all.length - visibleBars), offset + amount)))
  const clampOffset = value => Math.max(0, Math.min(Math.max(0, all.length - visibleBars), value))

  const handleWheel = event => {
    if (all.length <= 12) return
    event.preventDefault()
    const rect = event.currentTarget.getBoundingClientRect()
    const pointerRatio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
    const direction = event.deltaY < 0 ? 1 : -1
    setZoom(currentZoom => {
      const nextZoom = Math.max(1, Math.min(12, currentZoom + direction * 0.5))
      const currentVisible = Math.max(12, Math.min(all.length, Math.round((rangeBars[range] || 48) / currentZoom)))
      const nextVisible = Math.max(12, Math.min(all.length, Math.round((rangeBars[range] || 48) / nextZoom)))
      const currentStart = Math.max(0, all.length - currentVisible - offset)
      const anchorIndex = currentStart + pointerRatio * currentVisible
      const nextStart = Math.max(0, Math.min(all.length - nextVisible, Math.round(anchorIndex - pointerRatio * nextVisible)))
      setOffset(Math.max(0, all.length - nextVisible - nextStart))
      return nextZoom
    })
  }

  const handleChartKeyDown = event => {
    if (event.key !== '+' && event.key !== '=' && event.key !== '-' && event.key !== '_') return
    event.preventDefault()
    setZoom(value => Math.max(1, Math.min(12, value + (event.key === '+' || event.key === '=' ? 0.5 : -0.5))))
  }

  const handlePointerDown = event => {
    if (event.button !== 0) return
    event.currentTarget.setPointerCapture?.(event.pointerId)
    dragRef.current = { x: event.clientX, offset }
    setSelected(null)
  }

  const handlePointerMove = event => {
    if (!dragRef.current || !all.length) return
    const rect = event.currentTarget.getBoundingClientRect()
    const barsPerPixel = visibleBars / Math.max(rect.width, 1)
    const deltaBars = (event.clientX - dragRef.current.x) * barsPerPixel
    setOffset(clampOffset(dragRef.current.offset + deltaBars))
  }

  const stopDragging = event => {
    if (!dragRef.current) return
    event.currentTarget.releasePointerCapture?.(event.pointerId)
    dragRef.current = null
  }
  const active = selected || hover
  const activeIndex = active ? shown.findIndex(candle => candle.time === active.candle.time) : -1
  const tooltipX = activeIndex < 0 ? 0 : Math.min(sx(activeIndex) + 12, w - 230)
  const last = shown[shown.length - 1]
  const labelEvery = Math.max(1, Math.ceil(shown.length / 6))

  return (
    <section className="chart-shell">
      <div className="chart-head">
        <div>
          <span className="kicker">PRICE ACTION · {mode === 'live' ? 'LIVE' : 'PAST'}</span>
          <h2>XAG/USD Spot <code>XAG/USD</code></h2>
        </div>
        <div className="chart-legend">
          <span className="legend-item"><i className="up" />Bullish</span>
          <span className="legend-item"><i className="down" />Bearish</span>
          <span className="legend-item"><i className="neutral" />Neutral</span>
        </div>
      </div>

      <div className="chart-toolbar">
        <div className="toolbar-cluster">
          <label>
            Timeframe
            <select
              value={timeframe}
              onChange={event => {
                setTimeframe(Number(event.target.value))
                setOffset(0)
                setSelected(null)
              }}
            >
              {models?.configuration?.candle_minutes != null && (
                <option value={models.configuration.candle_minutes}>{models.configuration.candle_minutes}m</option>
              )}
            </select>
          </label>
        <span className="toolbar-readout">
            Forecast {models?.prediction_horizon_minutes == null ? 'Unavailable' : `${models.prediction_horizon_minutes}m`}
          </span>
          <span className="chart-gesture-hint">Mouse wheel: zoom in/out · drag: pan</span>
          <div className="toolbar-divider" />
          <button type="button" onClick={() => setZoom(value => Math.min(12, value + .5))}>Zoom +</button>
          <button type="button" onClick={() => setZoom(value => Math.max(1, value - .5))}>Zoom −</button>
        </div>

        <div className="toolbar-cluster">
          <button type="button" onClick={() => shift(-Math.max(6, Math.round(visibleBars / 3)))}>← Older</button>
          <button type="button" onClick={() => shift(Math.max(6, Math.round(visibleBars / 3)))}>Newer →</button>
          <div className="toolbar-divider" />
          {Object.keys(rangeBars).map(name => (
            <button
              type="button"
              key={name}
              className={range === name ? 'selected' : ''}
              onClick={() => {
                setRange(name)
                setOffset(0)
              }}
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      {shown.length ? (
        <div
          className={`chart-wrap${dragRef.current ? ' is-dragging' : ''}`}
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopDragging}
          onPointerCancel={stopDragging}
          onKeyDown={handleChartKeyDown}
          tabIndex={0}
          aria-label="Interactive XAG/USD chart. Use plus and minus keys or mouse wheel to zoom."
        >
          <svg
            viewBox={`0 0 ${w} ${h}`}
            preserveAspectRatio="xMidYMid meet"
            onMouseLeave={() => setHover(null)}
          >
            {[0, .25, .5, .75, 1].map(value => (
              <g key={value}>
                <line
                  className="grid"
                  x1={pad.l}
                  x2={w - pad.r}
                  y1={sy(min + (max - min) * value)}
                  y2={sy(min + (max - min) * value)}
                />
                <text className="axis" x={w - pad.r + 8} y={sy(min + (max - min) * value) + 4}>
                  {(min + (max - min) * value).toFixed(3)}
                </text>
              </g>
            ))}

            {shown.map((candle, index) => {
              const candleSignals = createSeriesMarkers(candle, chartPredictions, timeframe)
              const bullish = candle.close >= candle.open
              const bodyWidth = Math.max(4, step * 0.62)
              const timeLabel = index % labelEvery === 0 || index === shown.length - 1

              return (
                <g key={candle.time} onMouseMove={() => setHover({ candle, marker: null })}>
                  <title>{candleSignals[0] ? `${candleSignals[0].direction} · ${candleSignals[0].confidenceLabel}` : 'No clear signal'}</title>
                  {timeLabel && (
                    <text className="axis time-axis" x={sx(index)} y={h - 10} textAnchor="middle">
                      {new Date(candle.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </text>
                  )}

                  <line
                    className={candle.forming ? 'candle-forming' : bullish ? 'candle-up' : 'candle-down'}
                    x1={sx(index)}
                    x2={sx(index)}
                    y1={sy(candle.high)}
                    y2={sy(candle.low)}
                    strokeWidth="1.4"
                  />
                  <rect
                    className={candle.forming ? 'body-forming' : bullish ? 'body-up' : 'body-down'}
                    x={sx(index) - bodyWidth / 2}
                    y={Math.min(sy(candle.open), sy(candle.close))}
                    width={bodyWidth}
                    height={Math.max(1.8, Math.abs(sy(candle.open) - sy(candle.close)))}
                    rx="1"
                  />
                  {!candle.forming && candleSignals.map((marker, markerIndex) => {
                    const markerY =
                      marker.signal === 'BEARISH'
                        ? sy(candle.high) - 14 - markerIndex * 14
                        : marker.signal === 'BULLISH'
                          ? sy(candle.low) + 14 + markerIndex * 14
                          : sy(candle.close) - markerIndex * 14

                    return (
                      <g
                        key={marker.id}
                        className="signal-marker"
                        role="button"
                        tabIndex="0"
                        onMouseMove={event => {
                          event.stopPropagation()
                          setHover({ candle, marker })
                        }
                        }
                        onClick={event => {
                          event.stopPropagation()
                          setSelected({ candle, marker })
                        }}
                        onKeyDown={event => {
                          if (event.key === 'Enter' || event.key === ' ') setSelected({ candle, marker })
                        }}
                      >
                        <title>
                          {`${mode === 'live' ? 'Live' : 'Past'} ${marker.direction} · ${marker.confidenceLabel} · ${date(marker.timestamp)} · reference ${money(marker.referencePrice)}`}
                        </title>
                        <circle
                          className={`marker-${marker.signal.toLowerCase().replace('bullish', 'up').replace('bearish', 'down').replace('neutral', 'neutral')}`}
                          cx={sx(index)}
                          cy={markerY}
                          r="7.5"
                        />
                        <text className="marker-glyph" x={sx(index)} y={markerY + 3.5} textAnchor="middle">
                          {marker.signal === 'BULLISH' ? '↑' : marker.signal === 'BEARISH' ? '↓' : '•'}
                        </text>
                      </g>
                    )
                  })}
                  {candle.forming && <text className="forming-label" x={sx(index)} y={sy(candle.high) - 12} textAnchor="middle">FORMING</text>}
                </g>
              )
            })}

            {last && (
              <text className="last-tag" x={w - pad.r + 8} y={sy(last.close) - 6}>
                {money(last.close)}
              </text>
            )}

            {active && activeIndex >= 0 && (
              <g>
                <line className="crosshair" x1={sx(activeIndex)} x2={sx(activeIndex)} y1={pad.t} y2={priceBottom} />
                <line className="crosshair" x1={pad.l} x2={w - pad.r} y1={sy(active.candle.close)} y2={sy(active.candle.close)} />
                <rect className="tooltip" x={tooltipX} y="28" width="228" height={active.marker ? 158 : 98} rx="10" />
                <text className="tooltip-text" x={tooltipX + 14} y="52">{date(active.candle.time)}</text>
                <text className="tooltip-text" x={tooltipX + 14} y="74">
                  O {money(active.candle.open)} · H {money(active.candle.high)}
                </text>
                <text className="tooltip-text" x={tooltipX + 14} y="96">
                  L {money(active.candle.low)} · C {money(active.candle.close)}
                </text>
                {active.marker && (
                  <>
                    <text className="tooltip-text marker-tooltip" x={tooltipX + 14} y="120">
                      {signalLabel(active.marker.signal)} · ref {money(active.marker.referencePrice)}
                    </text>
                    <text className="tooltip-text marker-tooltip" x={tooltipX + 14} y="140">
                      {active.marker.confidence == null
                        ? 'Confidence unavailable'
                        : `Confidence ${active.marker.confidenceKind ? `${active.marker.confidenceKind}: ` : ''}${(active.marker.confidence * 100).toFixed(1)}%`}
                    </text>
                    <text className="tooltip-text marker-tooltip" x={tooltipX + 14} y="160">
                      {active.marker.supportingEvidence?.[0] || 'Supporting indicators unavailable'}
                    </text>
                  </>
                )}
              </g>
            )}
          </svg>
        </div>
      ) : (
        <div className="chart-empty">Historical XAG/USD candles are unavailable. Expected data/raw/dukascopy/xagusd_1m.csv.</div>
      )}

      <div className="chart-foot">
        <span>{shown.length} candles · {market?.source || 'unavailable'} · {timeframe}m</span>
        <span>Latest {date(market?.timestamp)} · horizon {models?.prediction_horizon_minutes == null ? 'Unavailable' : `${models.prediction_horizon_minutes}m`}</span>
      </div>
    </section>
  )
}
