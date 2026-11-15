#!/usr/bin/env node

/*
 * Dukascopy realtime bridge.
 *
 * The owner supplies the installed getRealTimeRates implementation through
 * SILVERAI_DUKASCOPY_MODULE. This file deliberately does not install a
 * package or claim a live connection in an offline checkout.
 */

const API_URL = (process.env.SILVERAI_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const MODULE_NAME = process.env.SILVERAI_DUKASCOPY_MODULE || 'dukascopy-node'
const POLL_MS = Number(process.env.SILVERAI_POLL_MS || 20000)
const SYMBOLS = ['xagusd', 'xauusd', 'eurusd']

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
const minute = value => {
  const date = new Date(value)
  date.setUTCSeconds(0, 0)
  return date
}

function extractQuotes(response) {
  if (Array.isArray(response)) return response
  if (Array.isArray(response?.data)) return response.data
  if (Array.isArray(response?.rates)) return response.rates
  return response ? [response] : []
}

function quotePrice(quote) {
  const bid = quote?.bid
  if (typeof bid === 'number') return bid
  if (typeof bid?.price === 'number') return bid.price
  return Number(quote?.price ?? quote?.close ?? quote?.value)
}

function quoteTime(quote) {
  return quote?.timestamp ?? quote?.time ?? quote?.date ?? quote?.datetime
}

function addQuote(accumulators, symbol, quote) {
  const timestamp = new Date(quoteTime(quote))
  const price = quotePrice(quote)
  if (!Number.isFinite(timestamp.getTime()) || !Number.isFinite(price) || price <= 0) return
  const bucket = minute(timestamp).toISOString()
  const key = `${symbol}:${bucket}`
  const current = accumulators.get(key) || { symbol, timestamp: bucket, open: price, high: price, low: price, close: price }
  current.high = Math.max(current.high, price)
  current.low = Math.min(current.low, price)
  current.close = price
  accumulators.set(key, current)
}

async function post(path, body) {
  const response = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(`${path} returned ${response.status}`)
  return response.json()
}

async function loadRealtimeRates(getRealTimeRates, symbol) {
  return getRealTimeRates({ instrument: symbol, timeframe: 'm1', priceType: 'bid', volumes: false })
}

async function main() {
  const imported = await import(MODULE_NAME)
  const getRealTimeRates = imported.getRealTimeRates || imported.default?.getRealTimeRates || imported.default
  if (typeof getRealTimeRates !== 'function') throw new Error(`No getRealTimeRates export found in ${MODULE_NAME}`)
  const accumulators = new Map()
  let backoff = POLL_MS
  for (;;) {
    try {
      const now = new Date()
      for (const symbol of SYMBOLS) {
        const response = await loadRealtimeRates(getRealTimeRates, symbol)
        for (const quote of extractQuotes(response)) addQuote(accumulators, symbol, quote)
      }
      const currentMinute = minute(now).getTime()
      for (const [key, candle] of [...accumulators]) {
        if (new Date(candle.timestamp).getTime() >= currentMinute) continue
        await post('/api/live/candles', candle)
        accumulators.delete(key)
      }
      await post('/api/live/heartbeat', {})
      backoff = POLL_MS
      await sleep(POLL_MS)
    } catch (error) {
      console.error(`[live-bridge] ${new Date().toISOString()} ${error.message}`)
      try { await post('/api/live/heartbeat', {}) } catch {}
      await sleep(Math.min(backoff, 300000))
      backoff = Math.min(backoff * 2, 300000)
    }
  }
}

main().catch(error => {
  console.error(`[live-bridge] fatal: ${error.message}`)
  process.exitCode = 1
})
