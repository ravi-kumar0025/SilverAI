"""Best development variant: OOF abstention, baselines, costs and seed stability."""
from pathlib import Path
import csv, json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.directional_labels import make_triple_barrier_labels
from src.features.market_features import SPOT_FEATURE_COLUMNS, generate_spot_features
from src.market_model_config import SPOT_CONFIG
from src.data.market_candle_builder import align_market_data
from tools.run_step0_study import folds, model, calibrate, calibrated, OUT

def fit_rows(frame, seed):
    out=[]
    for fold,tr,ca,va in folds(len(frame), 2):
        train, cal, valid=frame.iloc[tr],frame.iloc[ca],frame.iloc[va]
        m=model(); m.set_params(random_state=seed); m.fit(train[list(SPOT_FEATURE_COLUMNS)],train.y.astype(int))
        c=calibrate(m.predict_proba(cal[list(SPOT_FEATURE_COLUMNS)])[:,1],cal.y.astype(int).to_numpy())
        p=calibrated(c,m.predict_proba(valid[list(SPOT_FEATURE_COLUMNS)])[:,1])
        q=valid.copy(); q["p"]=p; q["fold"]=fold; out.append(q)
    return pd.concat(out,ignore_index=True)

def main():
    cache=OUT/"cache"/"features_60m.pkl"; f=pd.read_pickle(cache)
    f=f[pd.to_datetime(f.timestamp,utc=True)<pd.Timestamp(SPOT_CONFIG.final_holdout_start,tz="UTC")].reset_index(drop=True)
    f,_=make_triple_barrier_labels(f,2,.3,timeout_policy="neutral",candle_minutes=60)
    f=f[f.cross_market_complete & f.triple_barrier_label.ne(0.5)].copy().rename(columns={"triple_barrier_label":"y"}).dropna(subset=list(SPOT_FEATURE_COLUMNS)+["y"])
    pred=fit_rows(f,42); pred["act"]=(pred.p>=.5).astype(int); pred["correct"]=(pred.act==pred.y).astype(int)
    meta=[]; prior=[]
    for fold in sorted(pred.fold.unique()):
        cur=pred[pred.fold==fold].copy()
        X=lambda z: np.c_[z.atr_14_pct.rank(pct=True), z.hour_sin, z.hour_cos, (z.p-.5).abs()]
        if prior:
            train=pd.concat(prior); mm=LogisticRegression(max_iter=500).fit(X(train),train.correct)
            cur["meta_act"]=(mm.predict_proba(X(cur))[:,1]>=.5)&cur.act.astype(bool)
        else: cur["meta_act"]=False
        meta.append({"fold":fold,"signals":int(cur.meta_act.sum()),"coverage":float(cur.meta_act.mean()),"accuracy":float(cur.loc[cur.meta_act,"correct"].mean()) if cur.meta_act.any() else None})
        prior.append(cur)
    thresholds=[]
    for t in (.52,.55,.58,.60,.65,.70,.75):
        s=(pred.p>=t)|(pred.p<=1-t); n=int(s.sum()); c=int(pred.loc[s,"correct"].sum()); p=c/n if n else None
        z=1.959963984540054; lo=hi=None
        if n:
            den=1+z*z/n; cen=(p+z*z/(2*n))/den; rad=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/den; lo=max(0,cen-rad); hi=min(1,cen+rad)
        thresholds.append({"threshold":t,"signals":n,"coverage":float(s.mean()),"accuracy":p,"wilson_lo":lo,"wilson_hi":hi})
    b=[]
    close=f.silver_close.to_numpy(); nextret=np.r_[close[1:]/close[:-1]-1,np.nan]
    for name,signal in [("always_up",np.ones(len(f))), ("last_candle",(f.silver_close>f.silver_open).astype(int).to_numpy()), ("rsi_rule",(f.rsi_14<30).astype(int).where((f.rsi_14<30)|(f.rsi_14>70),np.nan).to_numpy())]:
        mask=np.isfinite(signal); y=f.y.to_numpy(); b.append({"baseline":name,"signals":int(mask.sum()),"coverage":float(mask.mean()),"accuracy":float((signal[mask]==y[mask]).mean())})
    costs=[]
    direction=np.where(pred.p.to_numpy()>=.5,1,-1); ret=nextret[pred.index]; signal=np.ones(len(pred),bool)
    for mult in (1,2,3):
        cost=2*mult*(SPOT_CONFIG.spread_bps+SPOT_CONFIG.slippage_bps)/10000
        costs.append({"cost_multiple":mult,"signals":len(pred),"gross_return":float(np.nansum(direction*ret)),"cost":float(len(pred)*cost),"net_return":float(np.nansum(direction*ret)-len(pred)*cost)})
    seeds=[]
    for seed in (1,42,99):
        q=fit_rows(f,seed); seeds.append({"seed":seed,"accuracy":float((q.p>=.5).astype(int).eq(q.y.astype(int)).mean()),"auc_proxy":float(q.p.corr(q.y))})
    result={"best_variant":"triple_neutral_2h_60m","thresholds":thresholds,"meta_model":meta,"baselines":b,"costs":costs,"seed_stability":seeds}
    (OUT/"best_variant_extras.json").write_text(json.dumps(result,indent=2,default=lambda x: x.item() if hasattr(x,"item") else str(x)),encoding="utf-8")
    for key,rows in [("thresholds",thresholds),("meta_model",meta),("baselines",b),("costs",costs),("seed_stability",seeds)]: pd.DataFrame(rows).to_csv(OUT/f"{key}.csv",index=False)
    ledger=OUT/"ledger.csv"; lines=ledger.read_text(encoding="utf-8").splitlines();
    if not any("best_variant_extras" in x for x in lines): ledger.write_text(ledger.read_text(encoding="utf-8")+f"14,best_variant_extras,completed_development_only\n",encoding="utf-8")
    print("best_variant_extras complete")
if __name__=="__main__": main()
