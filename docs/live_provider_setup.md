# Live COMEX SI provider setup

SilverAI follows Option B in `ps.pdf`: the Python model runs behind its own live chart web app. For silver, the official live test lasts five trading days, and the submitted indicator must not be changed during the test.

## Provider decision

The repository does not silently select a live provider. Two practical candidates were reviewed:

- CME Group's official market-data APIs provide real-time futures data through a WebSocket API and cover COMEX products. See <https://www.cmegroup.com/market-data/real-time-futures-and-options-data-api.html>.
- Databento provides historical HTTP APIs and live APIs for CME Globex futures. See <https://www.databento.tech/docs/api-live> and <https://www.databento.tech/docs/api-historical>.

The selected provider must be approved based on licensing, account access, symbol/contract coverage, cost, latency, and whether the supplied historical data uses the same contract and roll convention. Neither provider is considered operational in this repository until credentials are supplied and an actual SI event is received.

## Required configuration

The future adapter should use environment variables, never committed secrets:

```text
SILVERAI_MARKET_DATA_PROVIDER=cme|databento
SILVERAI_MARKET_DATA_API_KEY=...
SILVERAI_MARKET_DATA_SYMBOL=SI
SILVERAI_MARKET_DATA_ENVIRONMENT=certification|production
```

The current default remains `HistoricalCsvProvider`, which reports `is_live=false` and `status=historical`. A provider adapter must only report `is_live=true` after authentication, subscription, symbol resolution, timestamp validation, candle-close confirmation, reconnect handling, and a real end-to-end event have been tested.
