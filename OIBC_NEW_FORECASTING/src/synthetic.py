"""Synthetic fixtures, never a representation of real competition performance."""
import numpy as np
import pandas as pd


def fixture(cfg,days=14):
    rng=np.random.default_rng(cfg['seed'])
    n=cfg['expected_sites']
    sites=pd.DataFrame({'site_id':[f'SYNTH_{i:03d}' for i in range(n)],'capacity_kw':42655.4/n,
                        'wa_station':[f'A{i%3}' for i in range(n)],'wb_station':[f'B{i%2}' for i in range(n)]})
    grid=pd.date_range('2025-01-01T01:00:00+09:00',periods=days*24,freq='h')
    # Synthetic convention ONLY: next-day delivery, decision at previous day 17:00.
    decisions=(grid-pd.Timedelta(hours=1)).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    targets=pd.MultiIndex.from_product([sites.site_id,grid],names=['site_id','ts']).to_frame(index=False)
    targets['decision_at']=targets.ts.map(dict(zip(grid,decisions)))
    labels=targets[['site_id','ts']].copy()
    sun=np.maximum(0,np.sin(np.pi*(labels.ts.dt.hour.to_numpy()-6)/12))
    cloud_by_hour=rng.uniform(.5,1.,len(grid))
    cloud=labels.ts.map(dict(zip(grid,cloud_by_hour))).to_numpy()
    labels['energy_kwh']=sites.capacity_kw.iloc[0]*sun*cloud*rng.uniform(.85,1.,len(labels))
    labels['label_available_at']=labels.ts+pd.Timedelta(hours=1)
    weather={}
    for provider,prefix,count in [('wa','A',3),('wb','B',2)]:
        rows=[]
        for station in range(count):
            w=pd.DataFrame({'station_id':f'{prefix}{station}','ts':grid,
                'issued_at':decisions-pd.Timedelta(hours=7),'available_at':decisions-pd.Timedelta(hours=6)})
            if provider=='wa': w['radiation']=800*np.maximum(0,np.sin(np.pi*(grid.hour-6)/12))*cloud_by_hour
            else: w['temperature']=273.15+15+8*np.sin(2*np.pi*grid.hour/24)
            rows.append(w)
        weather[provider]=pd.concat(rows,ignore_index=True)
    compositions={}
    for name,offset in [('synthetic_balanced',0),('synthetic_alternative',37)]:
        assignment=np.roll(np.arange(n)<n//2,offset)
        compositions[name]=pd.DataFrame({'site_id':sites.site_id,'group':np.where(assignment,'G1','G2')})
    return sites,targets,labels,weather,compositions
