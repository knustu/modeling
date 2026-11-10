import numpy as np
import pandas as pd


def make_synthetic(seed=2026,days=30):
    rng=np.random.default_rng(seed)
    ids=[f'SYNTH_{i:03d}' for i in range(253)]
    capacity=rng.uniform(60,280,253)
    capacity*=42655.4/capacity.sum()
    sites=pd.DataFrame({'site_id':ids,'capacity_kw':capacity})
    ts=pd.date_range('2025-01-01',periods=days*24,freq='h',tz='Asia/Seoul')
    common=rng.uniform(.45,1.,len(ts))
    rows=[]
    for i,site in enumerate(ids):
        sun=np.maximum(0,np.sin(np.pi*(ts.hour.to_numpy()-6+.15*np.sin(i))/12))
        energy=capacity[i]*sun*common*rng.uniform(.8,1.,len(ts))
        rows.append(pd.DataFrame({'site_id':site,'ts':ts,'energy':energy}))
    return sites,pd.concat(rows,ignore_index=True)
