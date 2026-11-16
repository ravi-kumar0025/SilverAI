"""Train the new 15-minute XAG/USD Down/Neutral/Up model from current data.

pred_1 is the fitted multiclass classifier. pred_2 is the same classifier plus
the train-only probability calibrator and validation-selected emission policy.
The existing binary study is not modified.
"""
from __future__ import annotations

import hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib, numpy as np, pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from src.data.market_candle_builder import build_candles
from src.features.market_features import SPOT_FEATURE_COLUMNS, generate_spot_features
from src.models.model_artifact_contract import dump_versioned_artifact, runtime_versions
from src.models.probability_calibration import ChronologicalProbabilityCalibrator, calibration_metrics
from src.directional_labels import make_spot_multiclass_labels

OUT = ROOT / "reports" / "spot_15m_20261116_multiclass"
SYMBOLS = ("xagusd", "xauusd", "eurusd")
LABELS = {0: "down", 1: "neutral", 2: "up"}

def complete_feature_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows the production inference validator can accept."""
    complete = frame["cross_market_complete"].fillna(False).astype(bool)
    complete &= frame[list(SPOT_FEATURE_COLUMNS)].notna().all(axis=1)
    return frame.loc[complete].reset_index(drop=True)

def read_csv(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path); x["timestamp"] = pd.to_datetime(x.timestamp, utc=True)
    return x[["timestamp", "open", "high", "low", "close"]]

def read_live(symbol: str) -> pd.DataFrame:
    path = ROOT / "data" / "live_log" / f"{symbol}_m1_raw.jsonl"
    rows = []
    if not path.is_file(): return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            for q in obj.get("candles", []):
                if len(q) >= 5: rows.append((pd.to_datetime(q[0], unit="ms", utc=True), *map(float, q[1:5])))
        except (ValueError, TypeError, json.JSONDecodeError): continue
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])

def latest_data(symbol: str) -> tuple[pd.DataFrame, dict]:
    base = read_csv(ROOT / "data" / "raw" / "dukascopy" / f"{symbol}_1m.csv")
    live = read_live(symbol)
    merged = pd.concat([base, live], ignore_index=True).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return merged.reset_index(drop=True), {"csv_rows": len(base), "live_rows": len(live), "merged_rows": len(merged), "start": str(merged.timestamp.iloc[0]), "end": str(merged.timestamp.iloc[-1])}

def aligned_latest() -> tuple[pd.DataFrame, dict]:
    frames, report = {}, {}
    for symbol in SYMBOLS:
        raw, report[symbol] = latest_data(symbol); frames[symbol] = build_candles(raw, 5).set_index("timestamp")
    common = frames["xagusd"].index.intersection(frames["xauusd"].index).intersection(frames["eurusd"].index)
    silver = frames["xagusd"].loc[common].rename(columns={c: f"silver_{c}" for c in frames["xagusd"].columns})
    for symbol in ("xauusd", "eurusd"):
        other = frames[symbol].loc[common].rename(columns={c: f"{symbol}_{c}" for c in frames[symbol].columns})
        silver = silver.join(other)
    silver["cross_market_complete"] = True
    report["aligned"] = {"rows": len(silver), "start": str(silver.index.min()), "end": str(silver.index.max())}
    return silver.reset_index(), report

def model() -> LGBMClassifier:
    return LGBMClassifier(objective="multiclass", num_class=3, n_estimators=240, learning_rate=.03, num_leaves=15,
                          min_child_samples=100, colsample_bytree=.8, reg_lambda=2, reg_alpha=.05,
                          verbosity=-1, random_state=42, n_jobs=1)

def regression_model(alpha: float | None = None) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile" if alpha is not None else "huber",
        alpha=alpha,
        n_estimators=180,
        learning_rate=.035,
        num_leaves=15,
        min_child_samples=100,
        reg_lambda=2,
        verbosity=-1,
        random_state=42,
        n_jobs=1,
    )

def score(y: np.ndarray, p: np.ndarray, emission: float | None = None) -> dict:
    pred = p.argmax(1) if emission is None else np.where(p.max(1) >= emission, p.argmax(1), 1)
    return {"rows": int(len(y)), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "log_loss": float(log_loss(y, p, labels=[0,1,2])),
            "coverage": float(np.mean(p.max(1) >= emission)) if emission is not None else 1.0,
            "confusion_matrix": confusion_matrix(y, pred, labels=[0,1,2]).tolist()}

def regression_score(y: np.ndarray, mean_pred: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> dict:
    sign_mask = y != 0
    sign_accuracy = float(np.mean(np.sign(mean_pred[sign_mask]) == np.sign(y[sign_mask]))) if sign_mask.any() else None
    return {
        "rows": int(len(y)),
        "mae": float(mean_absolute_error(y, mean_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, mean_pred))),
        "r2_vs_zero": float(r2_score(y, mean_pred)),
        "sign_accuracy": sign_accuracy,
        "interval_10_90_coverage": float(np.mean((y >= q10) & (y <= q90))),
        "interval_10_90_width": float(np.mean(q90 - q10)),
    }

def fit_regressors(frame: pd.DataFrame) -> dict[str, dict]:
    x = frame[list(SPOT_FEATURE_COLUMNS)]
    y = frame["target_return"]
    fitted = {
        "mean": regression_model().fit(x, y),
        "q10": regression_model(.1).fit(x, y),
        "q50": regression_model(.5).fit(x, y),
        "q90": regression_model(.9).fit(x, y),
    }
    return {
        key: {"model": value, "features": list(SPOT_FEATURE_COLUMNS)}
        for key, value in fitted.items()
    }

def predict_regressors(frame: pd.DataFrame, fitted: dict[str, dict]) -> dict[str, np.ndarray]:
    x = frame[list(SPOT_FEATURE_COLUMNS)]
    return {
        key: np.asarray(fitted[key]["model"].predict(x), dtype=float)
        for key in ("mean", "q10", "q50", "q90")
    }

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "pred_1").mkdir(exist_ok=True); (OUT / "pred_2").mkdir(exist_ok=True)
    aligned, data_report = aligned_latest(); features = generate_spot_features(aligned, ("silver_price", "volatility", "trend_momentum", "cross_market", "time_session"))
    candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0]
    end = len(features); a, b, c = int(end*.60), int(end*.75), int(end*.90)
    selection = []
    for k in candidates:
        labeled, label_report = make_spot_multiclass_labels(features, 3, k, 5)
        labeled = complete_feature_rows(labeled)
        n = len(labeled); ta, tb, tc = int(n*.60), int(n*.75), int(n*.90)
        train, calibration, validation, test = labeled.iloc[:ta-6], labeled.iloc[ta+3:tb-3], labeled.iloc[tb+3:tc-3], labeled.iloc[tc+3:]
        if min(map(len, (train, calibration, validation, test))) < 100 or train.label.nunique() < 3: continue
        clf = model().fit(train[list(SPOT_FEATURE_COLUMNS)], train.label.astype(int)); cal_raw = clf.predict_proba(calibration[list(SPOT_FEATURE_COLUMNS)])
        cal = ChronologicalProbabilityCalibrator().fit(cal_raw, calibration.label.astype(int).to_numpy(), calibration_fraction=.99)
        val_p = cal.transform(clf.predict_proba(validation[list(SPOT_FEATURE_COLUMNS)])); y = validation.label.astype(int).to_numpy()
        best = max(((score(y, val_p, t)["macro_f1"], t, score(y, val_p, t)) for t in [.34,.40,.45,.50,.55,.60,.65,.70,.75,.80]), key=lambda z: z[0])
        selection.append({"k": k, "label_report": label_report, "neutral_confidence_threshold": best[1], "validation": best[2], "rows": {"train":len(train),"calibration":len(calibration),"validation":len(validation),"test":len(test)}})
    chosen = max(selection, key=lambda x: (x["validation"]["macro_f1"], x["validation"]["balanced_accuracy"], x["validation"]["coverage"]))
    k, confidence = chosen["k"], chosen["neutral_confidence_threshold"]
    labeled, label_report = make_spot_multiclass_labels(features, 3, k, 5)
    raw_labeled_rows = len(labeled)
    labeled = complete_feature_rows(labeled)
    label_report["rows_dropped_incomplete_features"] = raw_labeled_rows - len(labeled)
    n=len(labeled); ta,tb,tc=int(n*.60),int(n*.75),int(n*.90)
    train, calibration, validation, test = labeled.iloc[:ta-6], labeled.iloc[ta+3:tb-3], labeled.iloc[tb+3:tc-3], labeled.iloc[tc+3:]
    clf = model().fit(train[list(SPOT_FEATURE_COLUMNS)], train.label.astype(int)); cal_raw=clf.predict_proba(calibration[list(SPOT_FEATURE_COLUMNS)])
    calibrator=ChronologicalProbabilityCalibrator().fit(cal_raw, calibration.label.astype(int).to_numpy(), calibration_fraction=.99)
    final= model().fit(pd.concat([train, calibration, validation])[list(SPOT_FEATURE_COLUMNS)], pd.concat([train, calibration, validation]).label.astype(int))
    test_p=calibrator.transform(final.predict_proba(test[list(SPOT_FEATURE_COLUMNS)])); y=test.label.astype(int).to_numpy()
    regression_frame = features.copy()
    regression_frame["target_return"] = np.log(
        regression_frame.silver_close.shift(-3) / regression_frame.silver_close
    )
    regression_frame = regression_frame[
        regression_frame[list(SPOT_FEATURE_COLUMNS)].notna().all(axis=1)
        & regression_frame.target_return.notna()
    ].reset_index(drop=True)
    rn = len(regression_frame)
    rta, rtb, rtc = int(rn * .60), int(rn * .75), int(rn * .90)
    regression_train = regression_frame.iloc[:rta - 6]
    regression_validation = regression_frame.iloc[rta + 3:rtb - 3]
    regression_test = regression_frame.iloc[rtc + 3:]
    validation_regressors = fit_regressors(regression_train)
    validation_predictions = predict_regressors(regression_validation, validation_regressors)
    regression_validation_score = regression_score(
        regression_validation.target_return.to_numpy(),
        validation_predictions["mean"],
        validation_predictions["q10"],
        validation_predictions["q90"],
    )
    final_regression_train = regression_frame.iloc[:rtc - 3]
    regression = fit_regressors(final_regression_train)
    regression_predictions = predict_regressors(regression_test, regression)
    regression_test_score = regression_score(
        regression_test.target_return.to_numpy(),
        regression_predictions["mean"],
        regression_predictions["q10"],
        regression_predictions["q90"],
    )
    regression_meta = {
        "target": "log(close[t+3]/close[t])",
        "horizon_minutes": 15,
        "horizon_bars": 3,
        "split": {
            "train": len(regression_train),
            "validation": len(regression_validation),
            "test": len(regression_test),
            "purge_bars": 3,
            "embargo_bars": 3,
        },
        "validation": regression_validation_score,
        "test": regression_test_score,
    }
    for value in regression.values():
        value["metadata"] = regression_meta
    meta={"model_version":"xagusd_5m_15m_multiclass_regression_v1","dataset_id":SPOT_CONFIG.dataset_id,"prediction_horizon_minutes":15,"horizon_bars":3,"label_mapping":LABELS,"selected_atr_k":k,"selected_neutral_confidence_threshold":confidence,"features":list(SPOT_FEATURE_COLUMNS),"feature_count":len(SPOT_FEATURE_COLUMNS),"runtime_versions":runtime_versions(),"data":data_report,"label_report":label_report,"split":{"train":len(train),"calibration":len(calibration),"validation":len(validation),"test":len(test),"purge_bars":3,"embargo_bars":3},"validation_selection":selection,"test":score(y,test_p,confidence),"test_calibration_metrics":calibration_metrics(test_p,y),"regression":regression_meta,"test_start":str(test.timestamp.iloc[0]),"test_end":str(test.timestamp.iloc[-1]),"status":"research_artifact_not_live_ready"}
    dump_versioned_artifact({"model":clf,"features":list(SPOT_FEATURE_COLUMNS),"metadata":meta}, OUT/"pred_1"/"classifier.joblib", artifact_kind="spot_15m_multiclass_classifier")
    dump_versioned_artifact({"model":final,"calibrator":calibrator,"regression":regression,"features":list(SPOT_FEATURE_COLUMNS),"metadata":meta}, OUT/"pred_2"/"calibrated_predictor.joblib", artifact_kind="spot_15m_multiclass_calibrated_predictor")
    (OUT/"metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (OUT/"selection.csv").write_text(pd.DataFrame(selection).to_csv(index=False), encoding="utf-8")
    (OUT/"README.md").write_text("# 15-minute multiclass and regression model\n\n`pred_1` is the base Down/Neutral/Up classifier. `pred_2` contains the calibrated classifier, validation-selected neutral threshold, and 15-minute mean plus 10/50/90% quantile return regressors. The final test blocks are untouched by selection.\n", encoding="utf-8")
    print(json.dumps({"output":str(OUT),"latest":data_report,"selected_k":k,"confidence_threshold":confidence,"classification_test":meta["test"],"regression_validation":regression_validation_score,"regression_test":regression_test_score}, indent=2))

if __name__ == "__main__": main()
