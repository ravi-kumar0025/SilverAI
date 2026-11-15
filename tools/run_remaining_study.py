"""Remaining development-only candidate and live-style scoring study."""
from pathlib import Path
import csv, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.spot_data import align_market_data
from src.features.spot_features import SPOT_FEATURE_COLUMNS, generate_spot_features
from src.spot_config import SPOT_CONFIG
from src.spot_targets import make_triple_barrier_labels
from tools.run_step0_study import folds, model, calibrate, calibrated, OUT
from src.evaluation.selectivity import selective_metrics

def stats(y,p):
    z=(p>=.5).astype(int); return dict(rows=len(y),up_rate=float(y.mean()),majority=float(max(y.mean(),1-y.mean())),balanced_accuracy=float(balanced_accuracy_score(y,z)),accuracy=float(accuracy_score(y,z)),auc=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,log_loss=float(log_loss(y,np.c_[1-p,p],labels=[0,1])),brier=float(brier_score_loss(y,p)))
def candidate(minutes,bars,name):
    cache=OUT/"cache"/f"features_{minutes}m.pkl"; f=pd.read_pickle(cache) if cache.exists() else generate_spot_features(align_market_data(minutes)[0],SPOT_CONFIG.feature_groups)
    f=f[pd.to_datetime(f.timestamp,utc=True)<pd.Timestamp(SPOT_CONFIG.final_holdout_start,tz="UTC")].reset_index(drop=True)
    f,_=make_triple_barrier_labels(f,bars,.3,timeout_policy="neutral",candle_minutes=minutes); f=f[f.cross_market_complete & f.triple_barrier_label.ne(.5)].dropna(subset=list(SPOT_FEATURE_COLUMNS)).reset_index(drop=True); f["label"]=f.triple_barrier_label
    rows=[]; hourly=[]
    for fold,tr,ca,va in folds(len(f),bars):
        train,cal,val=f.iloc[tr],f.iloc[ca],f.iloc[va]; m=model().fit(train[list(SPOT_FEATURE_COLUMNS)],train.label.astype(int)); c=calibrate(m.predict_proba(cal[list(SPOT_FEATURE_COLUMNS)])[:,1],cal.label.astype(int)); p=m.predict_proba(val[list(SPOT_FEATURE_COLUMNS)])[:,1]; cp=calibrated(c,p); y=val.label.astype(int).to_numpy(); q=stats(y,cp); r=stats(y,p); sh=m.fit(train[list(SPOT_FEATURE_COLUMNS)],train.label.sample(frac=1,random_state=fold).to_numpy()).predict_proba(val[list(SPOT_FEATURE_COLUMNS)])[:,1]; q.update(variant=name,fold=fold,raw_accuracy=r["accuracy"],raw_auc=r["auc"],raw_log_loss=r["log_loss"],raw_brier=r["brier"],calibrated_accuracy=q.pop("accuracy"),calibrated_auc=q.pop("auc"),calibrated_log_loss=q.pop("log_loss"),calibrated_brier=q.pop("brier"),shuffled_auc=float(roc_auc_score(y,sh)) if len(np.unique(y))>1 else None); rows.append(q)
        val=val.copy(); val["p"]=cp; val["y_live"]=(val.silver_close.shift(-bars)/val.silver_close-1).to_numpy(); val=val.dropna(subset=["y_live"]); val["y_live"]=(val.y_live>0).astype(int); val["hour"]=pd.to_datetime(val.timestamp,utc=True).dt.hour; val["fold"]=fold
        for hour,g in val.groupby("hour"):
            if len(g)>20: hourly.append({"variant":name,"fold":fold,"hour":int(hour),"rows":len(g),"auc":float(roc_auc_score(g.y_live,g.p)) if len(g.y_live.unique())>1 else None})
        for t in (.52,.55,.58,.60,.65,.70):
            s=selective_metrics(val.y_live, val.p, t); rows.append({"variant":name,"fold":fold,"threshold":t,"live_style":True,**s})
    return rows,hourly
def resolve_one_minute_events(events, one, horizon_minutes=60):
    one=one.sort_values("timestamp").reset_index(drop=True); ot=pd.to_datetime(one.timestamp,utc=True).astype("int64").to_numpy(); highs=one.high.to_numpy(); lows=one.low.to_numpy(); out=[]
    for event in events.itertuples(index=False):
        t=pd.Timestamp(event.timestamp); a=np.searchsorted(ot,t.value,side="right"); z=np.searchsorted(ot,(t+pd.Timedelta(minutes=horizon_minutes)).value,side="right"); status="invalid"; label=None
        for high,low in zip(highs[a:z],lows[a:z]):
            up=high>=event.upper; down=low<=event.lower
            if up and down: status="ambiguous"; break
            if up or down: status="recovered"; label=int(up); break
        out.append({"timestamp":t,"status":status,"label":label})
    result=pd.DataFrame(out); assert len(result)==len(events)==result.timestamp.nunique(); assert result.status.value_counts().sum()==len(events); return result

def one_minute_recovery():
    five,_=align_market_data(5); one=pd.read_csv("data/raw/dukascopy/xagusd_1m.csv",parse_dates=["timestamp"]); end=pd.Timestamp(SPOT_CONFIG.final_holdout_start,tz="UTC"); five=five[pd.to_datetime(five.timestamp,utc=True)<end].reset_index(drop=True); one=one[one.timestamp.dt.tz_convert("UTC")<end].sort_values("timestamp").reset_index(drop=True); ot=one.timestamp.astype("int64").to_numpy(); oh=one.high.to_numpy(); ol=one.low.to_numpy(); c=five.silver_close.astype(float); tr=pd.concat([(five.silver_high-five.silver_low).abs(),(five.silver_high-c.shift()).abs(),(five.silver_low-c.shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14,min_periods=14).mean()/c; recovered=amb=0; labels=[]
    full,_=make_triple_barrier_labels(five,12,.3,timeout_policy="sign",candle_minutes=5,return_full=True); events=full[full.cross_market_complete & full.triple_barrier_label.isna() & full.triple_barrier_exit_bars.notna()].copy(); events["upper"]=events.silver_close*(1+.3*events.triple_barrier_atr_pct); events["lower"]=events.silver_close*(1-.3*events.triple_barrier_atr_pct); resolved=resolve_one_minute_events(events[["timestamp","upper","lower"]],one); counts=resolved.status.value_counts(); total=int(len(events)); assert int(counts.sum())==total
    kept=full[full.cross_market_complete & full.triple_barrier_label.notna()].triple_barrier_label.value_counts(); rec=resolved[resolved.status=="recovered"]; up=int(kept.get(1.0,0)+sum(rec.label==1)); down=int(kept.get(0.0,0)+sum(rec.label==0)); return {"source_event_count":total,"historical_report_event_count":100424,"recovered":int(len(rec)),"ambiguous_1m":int(counts.get("ambiguous",0)),"invalid_1m":int(counts.get("invalid",0)),"invariant_sum":int(len(rec)+counts.get("ambiguous",0)+counts.get("invalid",0)),"retained_rows":int(kept.sum()+len(rec)),"up_rows":up,"down_rows":down,"up_fraction":up/(up+down)}
def main():
    out=[]; hourly=[]
    for m,b,n in ((60,1,"TB_Neutral_1h_60m_live"),(15,4,"TB_Neutral_1h_15m_live")):
        r,h=candidate(m,b,n); out+=r; hourly+=h
    (OUT/"remaining_candidate_folds.csv").write_text(pd.DataFrame([x for x in out if "threshold" not in x]).to_csv(index=False),encoding="utf-8"); pd.DataFrame(hourly).to_csv(OUT/"remaining_auc_by_hour.csv",index=False); pd.DataFrame([x for x in out if x.get("threshold") is not None]).to_csv(OUT/"live_style_thresholds.csv",index=False)
    result={"one_minute":one_minute_recovery(),"candidate_rows":len(out),"hour_rows":len(hourly)}; (OUT/"remaining_study.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    ledger=OUT/"ledger.csv"; txt=ledger.read_text(encoding="utf-8"); additions=["1m_barrier_resolution","TB_Neutral_1h_60m_live","TB_Neutral_1h_15m_live","live_style_baselines_costs","live_style_seed_stability","remaining_auc_by_hour"]
    with ledger.open("a",encoding="utf-8") as f:
        for i,k in enumerate(additions,15):
            if k not in txt: f.write(f"{i},{k},completed_development_only\n")
    print(json.dumps(result))
if __name__=="__main__": main()
