import numpy as np
import pandas as pd


def fixture(cfg):
    rng=np.random.default_rng(cfg['seed']);n=cfg['expected_sites']
    sites=pd.DataFrame({'site_id':[f'SYNTH_{i:03d}' for i in range(n)],'capacity_kw':42655.4/n})
    compositions={}
    for name,shift in [('synthetic_reference',0),('synthetic_alternative',37)]:
        compositions[name]=pd.DataFrame({'site_id':sites.site_id,'group':np.where(np.roll(np.arange(n)<n//2,shift),'G1','G2')})
    parts={}
    for role,start,days in [('development','2025-01-01',14),('validation','2025-02-01',14),('holdout','2025-03-01',14)]:
        ts=pd.date_range(start,periods=days*24,freq='h',tz='Asia/Seoul')
        sun=np.maximum(0,np.sin(np.pi*(ts.hour.to_numpy()-6)/12));actual=20+sun[:,None]*np.linspace(50,90,n)[None,:]
        noise=rng.normal(0,2,(len(ts),n));common=rng.normal(0,.5,(len(ts),1))
        predictions=np.stack([np.maximum(0,actual+noise+common+1.),np.maximum(0,actual+.8*noise-common+.5)])
        decisions=(ts-pd.Timedelta(hours=1)).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
        # Synthetic prices are confined to synthetic fixtures; not competition SMP.
        smp=np.full(len(ts),100.);weight=.1*smp*np.where(np.isin(ts.hour,[11,12,13,14,15]),3.,1.)
        parts[role]={'role':role,'ts':ts,'site_ids':sites.site_id.tolist(),'actual':actual,'predictions':predictions,
            'model_ids':['synthetic_reference','synthetic_alternative'],'decision_at':decisions,
            'trained_through':pd.Timestamp('2024-12-01',tz='Asia/Seoul'),'prediction_kind':'synthetic_oos',
            'forecast_round':'17:00','lineage':'plant_level','weight':weight,'smp':smp,'weather':sun*800,
            'weather_available_at':decisions-pd.Timedelta(hours=1)}
    return sites,compositions,parts
