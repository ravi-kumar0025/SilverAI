"""Development-only, cached, interval-purged spot target study."""
from __future__ import annotations

import csv, hashlib, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.market_candle_builder import align_market_data
from src.evaluation.selectivity import selective_metrics
from src.features.market_features import SPOT_FEATURE_COLUMNS, generate_spot_features
from src.market_model_config import SPOT_CONFIG
from src.directional_labels import make_spot_labels, make_triple_barrier_labels

OUT = Path("reports/spot_v3_20261115")
THRESHOLDS = (0.52, 0.55, 0.58, 0.60, 0.65)
SEED = 42

def model():
    return LGBMClassifier(objective="binary", n_estimators=220, learning_rate=.03,
        num_leaves=15, min_child_samples=100, colsample_bytree=.7, subsample=.7,
        subsample_freq=1, reg_lambda=2., reg_alpha=.05, verbosity=-1,
        random_state=SEED, n_jobs=1)

def calibrate(raw, y):
    z = np.log(np.clip(raw, 1e-6, 1-1e-6) / np.clip(1-raw, 1e-6, 1-1e-6)).reshape(-1, 1)
    return LogisticRegression(C=1., max_iter=500, random_state=SEED).fit(z, y)

def calibrated(c, raw):
    z = np.log(np.clip(raw, 1e-6, 1-1e-6) / np.clip(1-raw, 1e-6, 1-1e-6)).reshape(-1, 1)
    return c.predict_proba(z)[:, 1]

def metrics(y, p):
    pred = (p >= .5).astype(int)
    return {"rows": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, np.c_[1-p, p], labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)), "up_rate": float(y.mean())}

def folds(n, horizon, count=6):
    block = n // 10
    for i in range(count):
        val_start, val_end = (i + 3) * block, (i + 4) * block
        cal_end = val_start - SPOT_CONFIG.purge_bars
        cal_start = cal_end - block
        train_end = max(1, cal_start - horizon - SPOT_CONFIG.embargo_bars)
        if val_end > val_start and cal_end > cal_start and train_end > 100:
            yield i + 1, slice(0, train_end), slice(cal_start, cal_end), slice(val_start, val_end)

def run_variant(frame, name, horizon, label_col, label_report):
    frame = frame.dropna(subset=list(SPOT_FEATURE_COLUMNS) + [label_col]).reset_index(drop=True)
    rows = []
    for fold, tr, ca, va in folds(len(frame), horizon):
        train, cal, valid = frame.iloc[tr], frame.iloc[ca], frame.iloc[va]
        m = model().fit(train[list(SPOT_FEATURE_COLUMNS)], train[label_col].astype(int))
        cp = m.predict_proba(cal[list(SPOT_FEATURE_COLUMNS)])[:, 1]
        c = calibrate(cp, cal[label_col].astype(int).to_numpy())
        raw = m.predict_proba(valid[list(SPOT_FEATURE_COLUMNS)])[:, 1]
        calp = calibrated(c, raw); y = valid[label_col].astype(int).to_numpy()
        r, q = metrics(y, raw), metrics(y, calp)
        base = max(float(y.mean()), 1 - float(y.mean()))
        out = {"variant": name, "fold": fold, "rows": len(y), "up_rate": float(y.mean()),
               "always_up_accuracy": float((y == 1).mean()), "majority_accuracy": base,
               "raw_accuracy": r["accuracy"], "calibrated_accuracy": q["accuracy"],
               "raw_auc": r["auc"], "calibrated_auc": q["auc"], "raw_log_loss": r["log_loss"],
               "calibrated_log_loss": q["log_loss"], "raw_brier": r["brier"], "calibrated_brier": q["brier"]}
        for t in THRESHOLDS:
            s = selective_metrics(y, calp, t)
            out[f"t{t:.2f}_signals"], out[f"t{t:.2f}_accuracy"] = s["signals"], s["accuracy"]
            out[f"t{t:.2f}_lo"], out[f"t{t:.2f}_hi"] = s["accuracy_ci95_lower"], s["accuracy_ci95_upper"]
        rows.append(out)
    return rows, label_report, len(frame)

def main():
    OUT.mkdir(parents=True, exist_ok=True); cache = OUT / "cache"; cache.mkdir(exist_ok=True)
    all_rows, reports = [], {}
    for minutes in (15, 60):
        cache_file = cache / f"features_{minutes}m.parquet"
        try:
            features = pd.read_parquet(cache_file) if cache_file.exists() else None
        except Exception:
            features = None
        if features is None:
            aligned, _ = align_market_data(minutes)
            features = generate_spot_features(aligned, SPOT_CONFIG.feature_groups)
            try: features.to_parquet(cache_file, index=False)
            except Exception: features.to_pickle(cache_file.with_suffix(".pkl"))
        ts = pd.to_datetime(features.timestamp, utc=True)
        dev = features.loc[ts < pd.Timestamp(SPOT_CONFIG.final_holdout_start, tz="UTC")].reset_index(drop=True)
        for hours, bars in ((1, minutes and int(60/minutes)), (2, int(120/minutes)), (4, int(240/minutes))):
            atr, ar = make_spot_labels(dev, bars, .3, candle_minutes=minutes)
            atr = atr[atr.cross_market_complete].reset_index(drop=True)
            atr["study_label"] = atr.label
            key = f"atr_{hours}h_{minutes}m"; rr, rep, n = run_variant(atr, key, bars, "study_label", ar)
            all_rows += rr; reports[key] = {"label_report": rep, "rows_after_label": n}
            tb, tr = make_triple_barrier_labels(dev, bars, .3, timeout_policy="neutral", candle_minutes=minutes)
            tb = tb[tb.cross_market_complete & tb.triple_barrier_label.ne(0.5)].reset_index(drop=True); tb["study_label"] = tb.triple_barrier_label
            key = f"triple_neutral_{hours}h_{minutes}m"; rr, rep, n = run_variant(tb, key, bars, "study_label", tr)
            all_rows += rr; reports[key] = {"label_report": rep, "rows_after_label": n}
    pd.DataFrame(all_rows).to_csv(OUT / "step0_fold_table.csv", index=False)
    (OUT / "step0_study.json").write_text(json.dumps({"metadata":{"seed":SEED,"config_sha256":hashlib.sha256(Path("config.yaml").read_bytes()).hexdigest()},"reports":reports,"folds":all_rows}, indent=2, default=str), encoding="utf-8")
    ledger = OUT / "ledger.csv"; existing = list(csv.DictReader(ledger.open(encoding="utf-8"))) if ledger.exists() else []
    prior_configs = {row.get("config") for row in existing}
    with ledger.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not existing: w.writerow(["count", "config", "result"])
        for key in reports:
            if key not in prior_configs:
                w.writerow([len(existing)+1, key, "completed_development_only"]); existing.append({"config": key})
    print(f"variants={len(reports)} folds={len(all_rows)} ledger_entries={len(existing)}")

if __name__ == "__main__": main()
