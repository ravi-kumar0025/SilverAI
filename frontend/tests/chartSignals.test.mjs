import test from 'node:test'
import assert from 'node:assert/strict'
import { finalizedSignals, normalizeFinalSignal, signalsForCandle } from '../src/chartSignals.mjs'

const candle = { time: '2026-01-01T09:30:00-05:00', open: 50, high: 51, low: 49, close: 50.5, volume: 10 }

test('normalizes finalized bullish, bearish and neutral records', () => {
  const records = ['BULLISH', 'BEARISH', 'NEUTRAL'].map((signal, index) => ({
    prediction_id: signal,
    final_signal: signal,
    candle_finalized: 1,
    candle_timestamp: `2026-01-01T09:${30 + index * 5}:00-05:00`,
    close_price: 50 + index,
    confidence: index / 10,
    scenario_payload: JSON.stringify({ scenarios: { Bullish: { supporting_evidence: ['RSI'] }, Bearish: { supporting_evidence: ['MACD'] }, Neutral: { supporting_evidence: ['range'] } } }),
  }))
  assert.deepEqual(finalizedSignals(records).map(row => row.signal), ['BULLISH', 'BEARISH', 'NEUTRAL'])
  assert.equal(finalizedSignals(records)[0].supportingEvidence[0], 'RSI')
})

test('rejects raw directions, non-finalized rows and invalid timestamps', () => {
  assert.equal(normalizeFinalSignal({ predicted_direction: 'UP', candle_timestamp: candle.time, candle_finalized: 1 }), null)
  assert.equal(normalizeFinalSignal({ final_signal: 'BULLISH', candle_timestamp: candle.time, candle_finalized: 0 }), null)
  assert.equal(normalizeFinalSignal({ final_signal: 'BULLISH', candle_timestamp: 'not-a-date', candle_finalized: 1 }), null)
})

test('maps a finalized marker to its containing candle and preserves refresh-safe records', () => {
  const record = { prediction_id: 'b1', final_signal: 'BULLISH', candle_finalized: 1, candle_timestamp: candle.time, close_price: 50.5 }
  assert.equal(signalsForCandle(candle, [record], 5)[0].id, 'b1')
  assert.equal(signalsForCandle({ ...candle, time: '2026-01-01T09:35:00-05:00' }, [record], 5).length, 0)
})
