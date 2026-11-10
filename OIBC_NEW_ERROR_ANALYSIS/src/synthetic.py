import numpy as np
import pandas as pd


def fixture(cfg,days=28):
    rng=np.random.default_rng(cfg['seed']);n=cfg['expected_sites']
    sites=pd.DataFrame({'site_id':[f'SYNTH_{i:03d}' for i in range(n)],'capacity_kw':42655.4/n})
    grid=pd.date_range('2025-01-01T00:00:00+09:00',periods=days*24,freq='h')
    plant=pd.MultiIndex.from_product([sites.site_id,grid],names=['site_id','ts']).to_frame(index=False)
    hour=plant.ts.dt.hour.to_numpy();sun=np.maximum(0,np.sin(np.pi*(hour-6)/12))
    actual=20+80*sun;noise=rng.normal(0,2,len(plant));common=np.sin(plant.ts.dt.dayofyear.to_numpy()/3)
    frames=[]
    for model,bias,scale in [('synthetic_reference',2.,1.),('synthetic_candidate',.5,.7)]:
        f=plant.copy();f['actual_kwh']=actual
        f['forecast_kwh']=np.maximum(0,actual+bias+scale*noise+common)
        f['model_id']=model;f['forecast_round']='17:00';f['validation_fold']='development'
        f['decision_at']=(f.ts-pd.Timedelta(hours=1)).dt.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
        f['trained_through']=pd.Timestamp('2024-12-01T00:00:00+09:00')
        f['prediction_kind']='synthetic_oos'
        f['horizon_hours']=(f.ts-f.decision_at).dt.total_seconds()/3600
        frames.append(f)
    weather=frames[0][['site_id','ts','forecast_round','decision_at']].copy()
    weather['issued_at']=weather.decision_at-pd.Timedelta(hours=7)
    weather['available_at']=weather.decision_at-pd.Timedelta(hours=6)
    weather['irradiance']=sun*800;weather['cloud']=.5+.3*np.sin(hour)
    weather['temperature']=15+sun*10;weather['precipitation']=np.where(hour%5==0,1.,0.)
    weather['provider_disagreement']=abs(np.cos(hour))*100
    comps={}
    for name,shift in [('synthetic_balanced',0),('synthetic_alternative',37)]:
        comps[name]=pd.DataFrame({'site_id':sites.site_id,'group':np.where(np.roll(np.arange(n)<n//2,shift),'G1','G2')})
    return sites,pd.concat(frames,ignore_index=True),comps,grid,weather
