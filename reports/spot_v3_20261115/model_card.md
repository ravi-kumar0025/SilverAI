# XAG/USD spot v3 model card

Development-only artifact. No holdout was evaluated.
## Regression model

Trained regression models predict the 1-hour return, converted to a price forecast with a 10-90% range; estimates only. The return is converted to price using `close_now * exp(predicted_return)`; raw price levels are not modeled.
