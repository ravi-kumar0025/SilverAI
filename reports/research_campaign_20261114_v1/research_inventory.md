# SilverAI Comprehensive Predictive Performance Research Campaign

## Campaign status

This is the initial evidence inventory for the staged research campaign. The official production target and previously evaluated historical test partition remain frozen. No production model, metadata, calibration artifact, dataset, or prior report was changed by this inventory.

Completed in the repository before this campaign:

- Frozen original-target holdout evaluation: 24,086 rows, raw accuracy 38.40%.
- Validation-only classification comparison: logistic regression, Random Forest, Extra Trees, HistGradientBoosting variants, causal extended features, dummy baselines, and a train-fold-only decision-rule study.
- Calibration experiment and final holdout comparison.
- Descriptive error/regime analysis.
- Timeframe/horizon coverage experiment without predictive model comparison.

## Data inventory

| Artifact | Role | SHA-256 |
|---|---|---|
| `data/SI_1m_1098d.csv` | Raw one-minute intraday OHLCV | `B4B278EEF0A48E53DAD1ADCFEAEC18E40A9FA1BA84CB89209B4733E08EEE65A2` |
| `data/silver_5m.csv` | Five-minute OHLCV source | `4C664BD6C5D9DAE0BE52C010BE20C9D10CD454C0B0073D227195EDF56B03A397` |
| `data/silver_features.csv` | Five-minute feature dataset | `566DEE8A29537074E27338D014775698BD6D42988BE07CDF5A3DD94FEA27D613` |
| `data/silver_modeling_data.csv` | Feature dataset plus official 15-minute targets | `FA5F5A9270324D10EFC7ADD52964ADE00F908507137EA3B4EE0F0D64109879A5` |
| `data/SI_daily_roll-adjusted.csv` | Daily roll-adjusted OHLCV | `B75259BA2862202F465CC2921470C16F08DCADB3A631F4C8D15E590D5276B720` |

The modeling contract contains 160,573 validated rows. The official split is 112,398 train, 24,083 validation, and 24,086 test rows, chronological with three purged rows at train/validation boundaries. The training-only threshold is `0.0776397515528126` percent. Labels are DOWN / NEUTRAL / UP with class order `[-1, 0, 1]`; horizon is 15 minutes over three 5-minute bars.

## Pipeline inventory

- Loading and timestamp validation: `src/data/loader.py`, `src/data/preprocessing.py`.
- Causal approved features: `src/features/technical_indicators.py`; exactly 17 production features.
- Timestamp-aware target generation: `src/targets.py`.
- Official target preparation, thresholding and chronological split: `src/models/train.py`.
- Calibration: `src/models/calibration.py`, `src/experiments/calibration_experiment.py`.
- Frozen holdout evaluation: `src/experiments/final_holdout_evaluation.py`.
- Error/regime analysis: `src/experiments/error_analysis.py`.
- Validation-only permutation feature diagnostic: `src/experiments/feature_ablation.py`.
- Timeframe/horizon coverage only: `src/experiments/timeframe_horizon.py`.
- Classification candidate study: `src/experiments/classification_improvement.py`.

## Evidence-backed limitations

1. The official target is a fixed absolute-return neutral band. Existing evidence does not establish that another label definition improves predictive performance.
2. The approved feature set is small and contains price, candle, volume, time and rolling-volatility information, but no cross-asset or order-book information.
3. The timeframe/horizon experiment measures target coverage and class balance; it does not compare predictive models for alternative horizons.
4. The existing feature-ablation module is validation-only permutation diagnostics, not causal feature selection or a completed campaign-wide stability study.
5. LightGBM is installed and a bounded validation supplement was executed. XGBoost is unavailable; CatBoost and deep-learning frameworks were not found in the installed dependency contract and were not run.
6. No reliable external GC, DXY, rates, order-book, spread, trade-flow, or announcement-aligned dataset is present in the repository.
7. The historical test partition is permanently excluded from development and selection. Validation results must not be presented as future or live performance.

## External-data feasibility inventory

The repository currently contains no provider credentials, acquisition adapter, or aligned external datasets. CME documents real-time futures/options data through its WebSocket API and historical data through CME DataMine; official COMEX silver documentation confirms the product and historical-data route. CME DataMine catalog entries show paid access and licensing/subscription requirements. These are feasibility findings, not acquired data or completed integrations.

Potential sources and requirements:

| Information | Current local availability | Feasible source class | Main requirement |
|---|---|---|---|
| COMEX SI/GC/Copper futures | SI only, historical local file | CME DataMine or licensed distributor | Account, product/contract mapping, licensing, timestamp alignment |
| Silver volume/open interest | Volume only in local OHLCV | CME historical futures data | Contract-specific open-interest fields and release timing |
| DXY and rates/real yields | Not present | FRED/Treasury or licensed market vendor | Publication timestamps, release-vintage control, granularity alignment |
| Bid/ask/order book/trade flow | Not present | CME MDP or licensed tick/order-book vendor | Entitlement, high-resolution storage, contract roll and clock alignment |
| Economic announcements | Not present | Official agency calendars or licensed calendar | Actual publication timestamp and revision/vintage handling |

References: CME Market Data APIs (`https://www.cmegroup.com/market-data/market-data-api.html`), CME DataMine silver catalog (`https://datamine.new.cmegroup.com/catalog/dataset/1c8f57ad6de1cab16a052c3dab2f16be`), CME futures/options data catalog (`https://www.cmegroup.com/market-data/browse-data/catalog/futures-and-options-data.html`), and FRED Treasury rate categories (`https://fred.stlouisfed.org/categories/115?t=rate`).

## Campaign stage and next step

Stage 1 is complete. The next justified stage is alternative-target research on train/validation data only: rebuild timestamp-valid labels for a small predefined set of horizons and volatility-adjusted thresholds, regenerate causal features per candidate, and run the same chronological candidate protocol. Triple-barrier, trend-scanning, two-stage models, SHAP, SVM, CatBoost, deep learning, ensembles and external-data integration remain unexecuted.

The stretch goal of 60% accuracy is not supported by current evidence; the strongest completed validation accuracy is approximately 42.91% on the original target, and the frozen historical test accuracy is 38.40%.
