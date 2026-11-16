"""Generate report assets from canonical stored artifacts only.

This script never trains, scores, evaluates, downloads, or reads the discarded
run_remaining_study outputs. It writes only report figures/tables/index files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import joblib
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "spot_v3_20261115"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"
MANIFEST = ROOT / "data" / "dataset_manifest.json"
CONFIG = ROOT / "config.yaml"
DATA = ROOT / "data" / "raw" / "dukascopy"
CONFIG_VALUES = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
DATASET_ID = json.loads(MANIFEST.read_text(encoding="utf-8")).get("dataset_id")
assert DATASET_ID == "4be8732dbab3aaa6", DATASET_ID
HOLDOUT_START = pd.Timestamp(CONFIG_VALUES["final_holdout_start"])
HOLDOUT_END = pd.Timestamp(CONFIG_VALUES["final_holdout_end"])
PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "ink": "#1F2937", "grid": "#D9E2EC"}


def caption(ax, date_range: str) -> None:
    ax.text(.5, -.19, f"dataset_id={DATASET_ID} | range={date_range}", transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#4B5563")


def save(fig, name: str, date_range: str) -> None:
    fig.patch.set_facecolor("white")
    caption(fig.axes[0], date_range)
    fig.savefig(FIGURES / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def load_json(name: str) -> dict:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def raw_frame(symbol: str) -> pd.DataFrame | None:
    path = DATA / f"{symbol}_1m.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    return frame[(frame.timestamp <= HOLDOUT_END)].reset_index(drop=True)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6)); ax.axis("off")
    labels = ["Dukascopy\nOHLC", "Closed-candle\nbuilder", "34-feature\nschema", "Labels", "Classifier +\ncalibrator", "Return + quantile\nregressors", "Threshold\n/ forecast", "Live bridge", "API + UI"]
    positions = [(x, .63) for x in np.linspace(.12, .88, 5)] + [(x, .28) for x in np.linspace(.22, .78, 4)]
    for (x, y), label in zip(positions, labels):
        ax.add_patch(FancyBboxPatch((x-.085, y-.10), .17, .20, boxstyle="round,pad=.02", fc="#F5F9FC", ec=PALETTE["blue"], lw=1.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color=PALETTE["ink"])
    for left, right in zip(positions[:4], positions[1:5]): ax.annotate("", (right[0]-.09, right[1]), (left[0]+.09, left[1]), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"], "lw": .9})
    for left, right in zip(positions[5:8], positions[6:9]): ax.annotate("", (right[0]-.09, right[1]), (left[0]+.09, left[1]), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"], "lw": .9})
    ax.annotate("", (.22, .38), (.80, .53), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"], "lw": .9, "connectionstyle": "arc3,rad=-.15"})
    ax.annotate("", (.78, .53), (.78, .38), arrowprops={"arrowstyle": "->", "color": PALETTE["ink"], "lw": .9})
    ax.set_title("SilverAI spot research pipeline", fontsize=12, weight="bold")
    save(fig, "fig_3_1_architecture.png", "canonical pipeline")


def price_history() -> bool:
    frame = raw_frame("xagusd")
    if frame is None or frame.empty: return False
    daily = frame.set_index("timestamp")["close"].resample("D").last().dropna()
    fig, ax = plt.subplots(figsize=(6.5, 3.7)); ax.plot(daily.index, daily.values, color=PALETTE["blue"], lw=1)
    ax.axvspan(HOLDOUT_START, HOLDOUT_END, color=PALETTE["orange"], alpha=.22, label="used holdout")
    ax.set_title("XAG/USD price history", fontsize=12, weight="bold"); ax.set_xlabel("UTC date"); ax.set_ylabel("Bid close")
    ax.legend(frameon=False); ax.grid(axis="y", color=PALETTE["grid"], lw=.7); save(fig, "fig_3_2_price_history.png", "2023-11-01 to 2026-11-13")
    return True


def coverage_gaps() -> bool:
    frames = {s: raw_frame(s) for s in ("xagusd", "xauusd", "eurusd")}
    if any(v is None or v.empty for v in frames.values()): return False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.1), sharex=True)
    for color, (symbol, frame) in zip((PALETTE["blue"], PALETTE["orange"], PALETTE["green"]), frames.items()):
        daily = frame.set_index("timestamp").resample("D").size(); ax1.plot(daily.index, daily.values, label=symbol.upper(), color=color, lw=1)
        gaps = frame.timestamp.diff().dt.total_seconds().div(60).dropna(); largest = gaps[gaps > 5].sort_values(ascending=False).head(10)
        if len(largest): ax2.scatter(frame.timestamp.iloc[largest.index], largest.values, s=8, label=symbol.upper(), color=color)
    ax1.set_title("Data coverage by day"); ax1.set_ylabel("1m rows/day"); ax1.legend(frameon=False, ncol=3); ax1.grid(axis="y", color=PALETTE["grid"], lw=.7)
    ax2.set_title("Largest detected gaps"); ax2.set_ylabel("Gap duration (minutes)"); ax2.set_xlabel("UTC date"); ax2.legend(frameon=False, ncol=3); ax2.grid(axis="y", color=PALETTE["grid"], lw=.7)
    ax1.axvspan(HOLDOUT_START, HOLDOUT_END, color=PALETTE["orange"], alpha=.12); ax2.axvspan(HOLDOUT_START, HOLDOUT_END, color=PALETTE["orange"], alpha=.12)
    save(fig, "fig_3_3_coverage_gaps.png", "2023-11-01 to 2026-11-13"); return True


def labels_and_drops(metrics: dict) -> None:
    breakdown = metrics["triple_barrier_timeout_breakdown"]["sign"]["report"]
    values = {"both barrier": breakdown["both_barrier_rows"], "missing future": breakdown["missing_future_rows"], "warmup/ATR": breakdown["rows_before"] - breakdown["rows_kept"] - breakdown["both_barrier_rows"] - breakdown["missing_future_rows"]}
    fig, ax = plt.subplots(figsize=(6.5, 3.7)); ax.bar(list(values), list(values.values()), color=[PALETTE["red"], PALETTE["orange"], PALETTE["purple"]]); ax.set_title("Triple-barrier path exclusions"); ax.set_ylabel("Rows"); ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", color=PALETTE["grid"], lw=.7); save(fig, "fig_3_4_label_drops.png", "development artifact")
    fig, ax = plt.subplots(figsize=(6.5, 2.5)); ax.axis("off")
    boxes=[("ATR dead zone", "Up/Down outside k·ATR"), ("Triple barrier", "first upper/lower path hit"), ("Timeout", "configured timeout policy")]
    for i,(a,b) in enumerate(boxes):
        x=.06+i*.32; ax.add_patch(FancyBboxPatch((x,.33),.25,.34,boxstyle="round,pad=.02",fc="#F5F9FC",ec=PALETTE["blue"])); ax.text(x+.125,.53,a,ha="center",weight="bold",fontsize=9); ax.text(x+.125,.43,b,ha="center",fontsize=8)
    ax.set_title("Label construction", fontsize=12, weight="bold"); save(fig, "fig_3_4_label_construction.png", "development artifact")


def class_balance(metrics: dict) -> None:
    rows=[]
    for i, fold in enumerate(metrics["corrected_folds"], 1): rows.append({"fold": i, "up_rate": fold["raw"].get("up_rate"), "rows": fold["validation_rows"]})
    table=pd.DataFrame(rows); table.to_csv(TABLES/"class_balance_per_fold.csv",index=False)
    fig, ax=plt.subplots(figsize=(6.5,3.5)); ax.bar(table.fold.astype(str), table.up_rate, color=PALETTE["blue"]); ax.axhline(.5,color=PALETTE["red"],ls="--",lw=1); ax.set_title("Development class balance by fold"); ax.set_xlabel("Fold"); ax.set_ylabel("Up-rate"); ax.set_ylim(0,1); save(fig,"fig_3_5_class_balance.png","development artifact")


def walk_forward(metrics: dict) -> None:
    fig, ax=plt.subplots(figsize=(6.5,3.8)); y=0
    for i, fold in enumerate(metrics["corrected_folds"], 1):
        x=0
        parts=[("train",fold["train_rows"],PALETTE["blue"]),("purge",3,"#BDBDBD"),("calibration",fold["calibration_rows"],PALETTE["green"]),("embargo",3,"#999999"),("validation",fold["validation_rows"],PALETTE["orange"])]
        for label,n,color in parts:
            ax.barh(y,n,left=x,color=color,height=.62,label=label if i==1 else None); x+=n
        y+=1
    ax.set_yticks(range(len(metrics["corrected_folds"])),[f"Fold {i}" for i in range(1,len(metrics["corrected_folds"])+1)]); ax.set_xlabel("Rows (stored fold metadata)"); ax.set_title("Purged walk-forward layout"); ax.legend(frameon=False,ncol=5,fontsize=8); save(fig,"fig_3_6_walk_forward.png","development artifact; used holdout excluded from selection")


def eda() -> bool:
    x=raw_frame("xagusd"); g=raw_frame("xauusd")
    if x is None or g is None or x.empty or g.empty: return False
    x=x.set_index("timestamp"); hourly=x["close"].pct_change().groupby(x.index.hour).std(); returns=x["close"].pct_change().dropna(); ratio=(x["close"].resample("D").last()/g.set_index("timestamp")["close"].resample("D").last()).dropna()
    fig,axs=plt.subplots(1,3,figsize=(6.5,2.5)); axs[0].hist(returns.clip(returns.quantile(.01),returns.quantile(.99)),bins=50,color=PALETTE["blue"]); axs[0].set_title("Returns"); axs[0].set_xlabel("Close-to-close return")
    axs[1].bar(hourly.index,hourly.values,color=PALETTE["green"]); axs[1].set_title("Volatility by UTC hour"); axs[1].set_xlabel("Hour"); axs[1].set_ylabel("Std return")
    axs[2].plot(ratio.index,ratio.values,color=PALETTE["purple"]); axs[2].set_title("Gold/silver ratio"); axs[2].set_xlabel("UTC date");
    for ax in axs: ax.tick_params(labelsize=8); ax.grid(axis="y",color=PALETTE["grid"],lw=.6)
    save(fig,"fig_3_8_eda.png","2023-11-01 to 2026-11-13"); return True


def results(metrics: dict, step0: dict) -> None:
    rows=[]
    for i,fold in enumerate(metrics["corrected_folds"],1): rows.append({"fold":i,"rows":fold["validation_rows"],"raw_accuracy":fold["raw"]["accuracy"],"calibrated_accuracy":fold["calibrated"]["accuracy"],"raw_auc":fold["raw"]["auc"],"calibrated_auc":fold["calibrated"]["auc"],"raw_log_loss":fold["raw"]["log_loss"],"calibrated_log_loss":fold["calibrated"]["log_loss"],"raw_brier":fold["raw"]["brier"],"calibrated_brier":fold["calibrated"]["brier"]})
    table=pd.DataFrame(rows); table.to_csv(TABLES/"results_per_fold.csv",index=False)
    fig,ax=plt.subplots(figsize=(6.5,3.5)); ax.plot(table.fold,table.raw_accuracy,"o-",label="raw",color=PALETTE["blue"]); ax.plot(table.fold,table.calibrated_accuracy,"o-",label="calibrated",color=PALETTE["orange"]); ax.axhline(.5,color=PALETTE["red"],ls="--",lw=1); ax.set_title("Per-fold accuracy"); ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy"); ax.set_ylim(0,1); ax.legend(frameon=False); save(fig,"fig_4_1_accuracy.png","development artifact")
    if step0.get("folds"):
        df=pd.DataFrame(step0["folds"]); keep=[c for c in ["variant","fold","raw_accuracy","calibrated_accuracy","raw_auc","calibrated_auc"] if c in df]
        df[keep].to_csv(TABLES/"horizon_label_comparison.csv",index=False)
        means=df.groupby("variant",as_index=False)[[c for c in keep if c not in {"variant","fold"}]].mean(numeric_only=True)
        fig,ax=plt.subplots(figsize=(6.5,3.7)); means.plot(x="variant",y="calibrated_accuracy",kind="bar",ax=ax,color=PALETTE["blue"],legend=False); ax.axhline(.5,color=PALETTE["red"],ls="--",lw=1); ax.set_title("Stored horizon/label comparison"); ax.set_xlabel("Variant"); ax.set_ylabel("Mean calibrated accuracy"); ax.tick_params(axis="x",rotation=70); save(fig,"fig_4_3_horizon_label_comparison.png","development artifact")
    threshold=REPORT/"step0_fold_table.csv"
    if threshold.is_file():
        td=pd.read_csv(threshold); cols=[c for c in td.columns if c.startswith("t") and c.endswith("_accuracy")]
        if cols:
            td.to_csv(TABLES/"thresholds_per_fold.csv",index=False); fig,ax=plt.subplots(figsize=(6.5,3.7));
            for c in cols: ax.plot(td.fold,td[c],"o-",label=c.removeprefix("t").removesuffix("_accuracy"))
            ax.set_title("Stored confident-signal accuracy by fold"); ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy"); ax.legend(frameon=False,ncol=3,fontsize=8); save(fig,"fig_4_4_thresholds.png","development artifact")


def regression_results() -> bool:
    path = ROOT / "reports" / "spot_v3_20261116_regression" / "regression_metrics.csv"
    if not path.is_file(): return False
    table = pd.read_csv(path)
    table.to_csv(TABLES / "regression_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    axes[0].bar(table["fold"].astype(str), table["r2_vs_zero"], color=PALETTE["blue"])
    axes[0].axhline(0, color=PALETTE["red"], ls="--", lw=1)
    axes[0].set_title("Regression R² vs zero-return"); axes[0].set_xlabel("Fold"); axes[0].set_ylabel("R²")
    axes[1].plot(table["fold"].astype(str), table["sign_accuracy"], "o-", color=PALETTE["orange"], label="sign accuracy")
    axes[1].plot(table["fold"].astype(str), table["band_coverage"], "o-", color=PALETTE["green"], label="10–90% coverage")
    axes[1].axhline(.5, color=PALETTE["red"], ls="--", lw=1); axes[1].set_ylim(0, 1)
    axes[1].set_title("Regression sign and interval coverage"); axes[1].set_xlabel("Fold"); axes[1].set_ylabel("Rate"); axes[1].legend(frameon=False, fontsize=8)
    for ax in axes: ax.grid(axis="y", color=PALETTE["grid"], lw=.7)
    save(fig, "fig_4_9_regression_results.png", "development artifact; 1-hour return target")
    return True


def session_regime() -> bool:
    """Plot stored hour-of-day AUC; do not infer a regime edge from it."""
    path = REPORT / "remaining_auc_by_hour.csv"
    if not path.is_file():
        return False
    table = pd.read_csv(path)
    required = {"variant", "fold", "hour", "auc"}
    if not required.issubset(table.columns) or table.empty:
        return False
    table.to_csv(TABLES / "session_regime_auc_by_hour.csv", index=False)
    summary = table.groupby(["variant", "hour"], as_index=False)["auc"].mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    for variant, group in summary.groupby("variant"):
        ax.plot(group["hour"], group["auc"], marker="o", ms=3, lw=1, label=str(variant))
    ax.axhline(.5, color=PALETTE["red"], ls="--", lw=1)
    ax.set_title("Stored AUC by UTC hour")
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("ROC AUC")
    ax.set_xticks(sorted(summary["hour"].unique()))
    ax.grid(axis="y", color=PALETTE["grid"], lw=.7)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_4_6_session_regime.png", "development artifact; stored hourly AUC only")
    return True


def stored_classifier_diagnostics(metrics: dict) -> list[str]:
    generated = []
    folds = metrics.get("corrected_folds", [])
    if folds:
        table = pd.DataFrame([{"fold": i, "raw_auc": f["raw"].get("auc"), "calibrated_auc": f["calibrated"].get("auc")} for i, f in enumerate(folds, 1)])
        table.to_csv(TABLES / "auc_per_fold.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.5, 3.5)); ax.plot(table.fold, table.raw_auc, "o-", label="raw", color=PALETTE["blue"]); ax.plot(table.fold, table.calibrated_auc, "o-", label="calibrated", color=PALETTE["orange"]); ax.axhline(.5, color=PALETTE["red"], ls="--", lw=1); ax.set_title("Per-fold AUC"); ax.set_xlabel("Fold"); ax.set_ylabel("ROC AUC"); ax.legend(frameon=False); ax.grid(axis="y", color=PALETTE["grid"], lw=.7); save(fig, "fig_4_2_auc_curves.png", "development artifact"); generated.append("fig_4_2_auc_curves.png")
    costs = REPORT / "costs.csv"
    if costs.is_file():
        table = pd.read_csv(costs); table.to_csv(TABLES / "cost_sensitivity.csv", index=False); fig, ax = plt.subplots(figsize=(6.5, 3.5)); ax.plot(table.cost_multiple, table.net_return, "o-", color=PALETTE["red"]); ax.axhline(0, color=PALETTE["ink"], lw=1); ax.set_title("Stored cost sensitivity"); ax.set_xlabel("Cost multiple"); ax.set_ylabel("Net return (stored units)"); ax.grid(axis="y", color=PALETTE["grid"], lw=.7); save(fig, "fig_4_7_profit_costs.png", "development artifact"); generated.append("fig_4_7_profit_costs.png")
    seed = REPORT / "seed_stability.csv"
    if seed.is_file():
        table = pd.read_csv(seed); table.to_csv(TABLES / "seed_stability.csv", index=False); fig, ax = plt.subplots(figsize=(6.5, 3.5)); ax.plot(table.seed, table.accuracy, "o-", color=PALETTE["blue"]); ax.axhline(.5, color=PALETTE["red"], ls="--", lw=1); ax.set_title("Stored seed stability"); ax.set_xlabel("Seed"); ax.set_ylabel("Accuracy"); ax.grid(axis="y", color=PALETTE["grid"], lw=.7); save(fig, "fig_4_8_seed_stability.png", "development artifact"); generated.append("fig_4_8_seed_stability.png")
    return generated


def main() -> None:
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8")); assert manifest.get("dataset_id")==DATASET_ID, manifest.get("dataset_id"); print(f"dataset_id={DATASET_ID}")
    FIGURES.mkdir(parents=True,exist_ok=True); TABLES.mkdir(parents=True,exist_ok=True)
    metrics=load_json("metrics.json"); step0=load_json("step0_study.json")
    generated=[]; missing=[]
    architecture(); generated.append("fig_3_1_architecture.png")
    for name,func in (("fig_3_2_price_history.png",price_history),("fig_3_3_coverage_gaps.png",coverage_gaps),("fig_3_8_eda.png",eda)):
        if func(): generated.append(name)
        else: missing.append(name)
    labels_and_drops(metrics); generated += ["fig_3_4_label_drops.png","fig_3_4_label_construction.png"]
    class_balance(metrics); generated.append("fig_3_5_class_balance.png")
    walk_forward(metrics); generated.append("fig_3_6_walk_forward.png")
    results(metrics,step0); generated += ["fig_4_1_accuracy.png"]
    if regression_results(): generated.append("fig_4_9_regression_results.png")
    generated += stored_classifier_diagnostics(metrics)
    if (FIGURES/"fig_4_3_horizon_label_comparison.png").is_file(): generated.append("fig_4_3_horizon_label_comparison.png")
    else: missing.append("fig_4_3_horizon_label_comparison.png")
    if (FIGURES/"fig_4_4_thresholds.png").is_file(): generated.append("fig_4_4_thresholds.png")
    else: missing.append("fig_4_4_thresholds.png")
    if not session_regime():
        missing.append("fig_4_6_session_regime.png")
    missing += ["fig_3_7_feature_groups_heatmap.png", "fig_4_5_reliability.png"]
    (TABLES/"experiment_ledger.csv").write_text((REPORT/"ledger.csv").read_text(encoding="utf-8"),encoding="utf-8")
    payload=joblib.load(REPORT/"spot_lightgbm.joblib")["payload"]
    (TABLES/"feature_list.csv").write_text("feature\n"+"\n".join(payload["features"])+"\n",encoding="utf-8")
    index=["# Figures index","",f"dataset_id: {DATASET_ID}","","Generated:"]+[f"- {x}" for x in generated]+["","MISSING:"]+[f"- {x}" for x in missing]
    (REPORT/"figures_index.md").write_text("\n".join(index)+"\n",encoding="utf-8")
    print(f"generated={len(generated)} missing={len(missing)}")

if __name__ == "__main__": main()
