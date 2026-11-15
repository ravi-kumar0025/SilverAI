const FINAL_SIGNALS = new Set(['BULLISH', 'BEARISH', 'NEUTRAL'])

function finiteNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function parsePayload(value) {
  if (!value) return null
  if (typeof value === 'object') return value
  try { return JSON.parse(value) } catch { return null }
}

export function normalizeFinalSignal(record) {
  if (!record || !(record.candle_finalized === true || Number(record.candle_finalized) === 1)) return null
  const signal = String(record.final_signal || '').toUpperCase()
  const timestamp = record.candle_timestamp
  if (!FINAL_SIGNALS.has(signal) || !timestamp || !Number.isFinite(Date.parse(timestamp))) return null
  const scenarios = parsePayload(record.scenario_payload)
  const scenarioName = signal === 'BULLISH' ? 'Bullish' : signal === 'BEARISH' ? 'Bearish' : 'Neutral'
  const evidence = scenarios?.scenarios?.[scenarioName]?.supporting_evidence
  return {
    id: record.prediction_id || `${signal}-${timestamp}`,
    signal,
    timestamp,
    referencePrice: finiteNumber(record.close_price),
    confidence: finiteNumber(record.confidence),
    confidenceKind: record.confidence_kind || null,
    supportingEvidence: Array.isArray(evidence) ? evidence.filter(Boolean) : [],
  }
}

export function finalizedSignals(records) {
  return (records || []).map(normalizeFinalSignal).filter(Boolean)
}

export function signalsForCandle(candle, records, timeframeMinutes = 5) {
  if (!candle?.time) return []
  const start = Date.parse(candle.time)
  if (!Number.isFinite(start)) return []
  const end = start + timeframeMinutes * 60000
  return finalizedSignals(records).filter(signal => {
    const timestamp = Date.parse(signal.timestamp)
    return timestamp >= start && timestamp < end
  })
}

// Stable marker contract used by the chart renderer. Finalized records are
// read-only; no marker is created for a forming candle or an unfinalized row.
export function createSeriesMarkers(candle, records, timeframeMinutes = 5) {
  return signalsForCandle(candle, records, timeframeMinutes).map(signal => ({
    ...signal,
    direction: signal.signal === 'BULLISH' ? 'Up' : signal.signal === 'BEARISH' ? 'Down' : 'No clear signal',
    confidenceLabel: signal.confidence == null ? 'Unavailable' : `${signal.signal === 'BULLISH' ? 'Up' : signal.signal === 'BEARISH' ? 'Down' : 'No clear signal'}, ${(signal.confidence * 100).toFixed(0)}%`,
  }))
}
