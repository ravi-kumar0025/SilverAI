"""Build report figures/tables only from stored development artifacts."""
from pathlib import Path
import json, shutil
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"reports/spot_v3_20261115"; FIG=OUT/"figures"; TAB=OUT/"tables"
sys.path.insert(0,str(ROOT))
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
def save(fig,name):
    fig.savefig(FIG/name,dpi=200,facecolor="white",bbox_inches="tight"); plt.close(fig)
def main():
    d=json.loads((OUT/"step0_study.json").read_text(encoding="utf-8")); folds=pd.DataFrame(d["folds"])
    extras=json.loads((OUT/"best_variant_extras.json").read_text(encoding="utf-8"))
    (TAB/"primary_finding.csv").write_text("finding,status\npre_registered_promotion_rule,NOT_MET\n",encoding="utf-8")
    folds.to_csv(TAB/"metrics_per_fold.csv",index=False)
    for k in ("thresholds","meta_model","baselines","costs","seed_stability"): pd.DataFrame(extras[k]).to_csv(TAB/(k+".csv"),index=False)
    for src in ("remaining_candidate_folds.csv","remaining_auc_by_hour.csv","live_style_thresholds.csv"):
        if (OUT/src).exists(): shutil.copy2(OUT/src,TAB/src)
    pd.DataFrame({"feature":__import__("src.features.spot_features",fromlist=["SPOT_FEATURE_COLUMNS"]).SPOT_FEATURE_COLUMNS}).to_csv(TAB/"feature_list.csv",index=False)
    shutil.copy2(OUT/"ledger.csv",TAB/"experiment_ledger.csv")
    tb=d["reports"]["triple_neutral_1h_60m"]["label_report"]
    drop={k:tb.get(k,0) for k in ("missing_future_rows","both_barrier_rows","timeout_rows")}
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(drop.keys(),drop.values(),color=["#4472c4","#ed7d31","#70ad47"]); ax.set_title("Triple-barrier dropped/path rows | development artifacts"); ax.set_ylabel("rows"); save(fig,"03_label_path_breakdown.png")
    g=folds.groupby("variant",as_index=False).agg(raw_accuracy=("raw_accuracy","mean"),calibrated_accuracy=("calibrated_accuracy","mean"),raw_auc=("raw_auc","mean")); fig,ax=plt.subplots(figsize=(10,4)); g.plot(x="variant",y=["raw_accuracy","calibrated_accuracy"],kind="bar",ax=ax,color=["#4472c4","#ed7d31"]); ax.axhline(.5,color="black",lw=.8); ax.set_title("Walk-forward mean accuracy | development artifacts"); ax.set_ylabel("accuracy"); ax.tick_params(axis="x",rotation=70); save(fig,"08_horizon_label_comparison.png")
    best=folds[folds.variant==extras["best_variant"]]; fig,ax=plt.subplots(figsize=(7,4)); ax.plot(best.fold,best.raw_accuracy,"o-",label="raw"); ax.plot(best.fold,best.calibrated_accuracy,"o-",label="calibrated"); ax.axhline(.5,color="black",lw=.8); ax.set_title("Best variant per-fold accuracy | development artifacts"); ax.set_xlabel("fold"); ax.set_ylabel("accuracy"); ax.legend(); save(fig,"07_per_fold_accuracy.png")
    old=json.loads((OUT/"metrics.json").read_text(encoding="utf-8")); old_mean=sum(x["calibrated"]["accuracy"] for x in old["corrected_folds"])/len(old["corrected_folds"]); new_mean=float(best.calibrated_accuracy.mean()); fig,ax=plt.subplots(figsize=(5,4)); ax.bar(["3-fold","6-fold"],[old_mean,new_mean],color=["#ed7d31","#4472c4"]); ax.axhline(.5,color="black",lw=.8); ax.set_title("Selection-bias check: pre-registered rule NOT MET"); ax.set_ylabel("calibrated accuracy"); save(fig,"07b_three_vs_six_fold_selection_bias.png")
    th=pd.DataFrame(extras["thresholds"]); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(th.threshold,th.accuracy,"o-"); ax.fill_between(th.threshold,th.wilson_lo,th.wilson_hi,alpha=.2); ax.set_title("Confident-signal accuracy | development artifacts"); ax.set_xlabel("threshold"); ax.set_ylabel("accuracy / Wilson 95% CI"); save(fig,"10_confident_thresholds.png")
    seed=pd.DataFrame(extras["seed_stability"]); fig,ax=plt.subplots(figsize=(6,4)); ax.bar(seed.seed.astype(str),seed.accuracy,color="#4472c4"); ax.set_title("Seed stability | development artifacts"); ax.set_xlabel("seed"); ax.set_ylabel("accuracy"); save(fig,"14_seed_stability.png")
    fig,ax=plt.subplots(figsize=(9,4)); ax.axis("off"); boxes=[("Dukascopy",.05),("candles",.22),("features",.39),("model",.56),("signal log/API",.73),("web UI",.9)];
    for label,x in boxes: ax.text(x,.5,label,ha="center",va="center",bbox=dict(boxstyle="round",fc="#eaf2f8",ec="#4472c4"));
    for (_,x),(_,y) in zip(boxes,boxes[1:]): ax.annotate("",xy=(y-.06,.5),xytext=(x+.06,.5),arrowprops=dict(arrowstyle="->"));
    ax.set_title("Spot pipeline architecture | stored project architecture"); save(fig,"16_architecture.png")
    missing={"01_price_history":"no stored figure artifact","02_coverage":"no stored coverage figure artifact","04_timeline":"no stored fold timeline artifact","05_feature_importance":"no stored importance artifact","06_lead_lag":"no stored lead-lag artifact","09_reliability":"no stored probability-bin artifact","11_session_regime":"no stored subgroup artifact","12_roc_confusion":"no stored curve artifact","13_profit_drawdown":"no stored equity curve artifact","15_signal_timeline":"no stored signal timeline artifact","17_live_validation":"live machine artifacts not available"}
    lines=["# Figures index","","Generated:"]+[f"- {p.name}: generated from stored development artifacts" for p in sorted(FIG.glob("*.png"))]+["","MISSING:"]+[f"- {k}: {v}" for k,v in missing.items()]
    (OUT/"figures_index.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"generated={len(list(FIG.glob('*.png')))} missing={len(missing)} tables={len(list(TAB.glob('*.csv')))}")
if __name__=="__main__": main()
