import pandas as pd
from tools.run_remaining_study import resolve_one_minute_events

def test_one_minute_resolution_partitions_events_once():
    events=pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=3,freq="h",tz="UTC"),"upper":[101,101,101],"lower":[99,99,99]})
    one=pd.DataFrame({"timestamp":pd.to_datetime(["2026-01-01 00:01","2026-01-01 01:01","2026-01-01 01:02","2026-01-01 02:01"],utc=True),"high":[101,101,101,100],"low":[100,99,99,100]})
    result=resolve_one_minute_events(events,one,60)
    assert len(result)==3 and result.timestamp.nunique()==3
    assert result.status.value_counts().sum()==3
    assert set(result.status)=={"recovered","ambiguous","invalid"}
