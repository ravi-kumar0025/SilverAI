# Active model methodology

The active model uses complete 5-minute XAG/USD candles and a 15-minute target, equal to three future bars. The classifier labels a row Down, Neutral, or Up using an ATR-scaled dead zone with selected `k=0.5`.

The classifier uses the ordered 34-feature schema covering silver price action, volatility, trend/momentum, XAU/USD and EUR/USD relationships, and UTC session context. Incomplete feature rows are removed before any split so training and production inference share the same valid-input contract.

The classifier uses chronological train, calibration, validation, and untouched-test blocks with purge/embargo rows. The probability calibrator is fitted only from the calibration block. `k` and the neutral confidence threshold are selected only on validation data. The final test block is not used for selection.

The regression component predicts `log(close[t+3]/close[t])` and supplies a mean estimate plus 10th, 50th, and 90th conditional quantiles. The price range is derived from the return estimates; it is not a raw price-level model.

These controls prevent future leakage, but the stored metrics remain research evidence. They do not demonstrate a reliable trading edge or authorize live orders.
