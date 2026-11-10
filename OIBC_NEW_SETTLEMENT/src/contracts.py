import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


class GateError(ValueError):pass


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode('utf-8')).hexdigest()


def save_json(path,value):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):v=v.item()
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False,default=str),encoding='utf-8')


def time_index(values):
    a=[]
    for v in values:
        if isinstance(v,(float,int,np.number)):raise GateError('Gate 1: numeric epoch needs explicit units.')
        t=pd.Timestamp(v)
        if pd.isna(t) or t.tzinfo is None:raise GateError('Gate 1: timezone-aware timestamps required.')
        t=t.tz_convert('Asia/Seoul')
        if t.minute or t.second or t.microsecond or t.nanosecond:raise GateError('Gate 1: hourly boundaries required.')
        a.append(t)
    return pd.DatetimeIndex(a)


def assignment(x,capacity,cfg):
    c=np.asarray(capacity,float);x=np.asarray(x)
    if x.dtype!=bool or x.shape!=c.shape or len(c)!=cfg['expected_sites']:
        raise GateError('Gate 2: one Boolean assignment per authoritative plant required.')
    if not np.isfinite(c).all() or (c<=0).any():raise GateError('Gate 2: invalid capacities.')
    caps=[float(c[x].sum()),float(c[~x].sum())]
    if min(caps)<cfg['minimum_capacity_kw']:raise GateError('Gate 2: infeasible group capacities.')
    return caps


def composition(frame,sites,cfg):
    if list(frame.columns)!=['site_id','group'] or frame.site_id.duplicated().any():raise GateError('Gate 2: malformed composition.')
    if sites.site_id.duplicated().any() or set(frame.site_id)!=set(sites.site_id):raise GateError('Gate 2: exact plant coverage required.')
    if not frame.group.isin(['G1','G2']).all():raise GateError('Gate 2: invalid group labels.')
    x=frame.set_index('site_id').loc[sites.site_id,'group'].eq('G1').to_numpy()
    assignment(x,sites.capacity_kw.to_numpy(),cfg)
    return x


def partition(data,role,cfg,synthetic=False):
    if data['role']!=role:raise GateError('Gate 5/6: partition role mismatch; holdout is not development data.')
    if not synthetic:raise GateError('Historical optimization disabled pending separate authorization.')
    ts=time_index(data['ts'])
    if ts.has_duplicates or not ts.is_monotonic_increasing:raise GateError('Gate 1: duplicate/unsorted timestamps.')
    if len(ts)>1 and not ((ts[1:]-ts[:-1])==pd.Timedelta(hours=1)).all():raise GateError('Gate 1: incomplete hourly coverage.')
    if ts.max()>=pd.Timestamp('2025-06-01T00:00:00+09:00'):raise GateError('Gate 3: competition actuals prohibited.')
    y=np.asarray(data['actual'],float);p=np.asarray(data['predictions'],float)
    n=cfg['expected_sites']
    if y.shape!=(len(ts),n) or p.ndim!=3 or p.shape[1:]!=y.shape:raise GateError('Gate 1: model/time/plant shape mismatch.')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y<0).any() or (p<0).any():raise GateError('Gate 3: finite nonnegative aligned forecasts/actuals required; no silent missingness.')
    ids=data['site_ids']
    if len(ids)!=n or len(set(ids))!=n:raise GateError('Gate 1: duplicate/missing plant IDs.')
    if len(data['model_ids'])!=len(p) or len(set(data['model_ids']))!=len(p):raise GateError('Gate 3: invalid model identifiers.')
    decisions=time_index(data['decision_at'])
    if len(decisions)!=len(ts) or (decisions>=ts-pd.Timedelta(hours=1)).any():raise GateError('Gate 3: invalid issuance availability.')
    trained=pd.Timestamp(data['trained_through'])
    if trained.tzinfo is None or trained>=decisions.min():raise GateError('Gate 3: training leakage.')
    if data['prediction_kind']!='synthetic_oos':raise GateError('Gate 4: OOS provenance required.')
    if data['forecast_round'] not in ['10:00','17:00']:raise GateError('Gate 3: unsupported round.')
    w=np.asarray(data['weight'],float)
    if w.shape!=(len(ts),) or not np.isfinite(w).all() or (w<0).any() or not (w>0).any():raise GateError('Gate 1: missing/negative/nonpositive optimization weights.')
    return ts


def chronological(development,validation,holdout,cfg):
    arrays=[partition(d,r,cfg,True) for d,r in [(development,'development'),(validation,'validation'),(holdout,'holdout')]]
    if arrays[0].max()>=arrays[1].min() or arrays[1].max()>=arrays[2].min():raise GateError('Gate 5/6: overlapping chronological partitions.')
    if pd.Timestamp(validation['decision_at'][0])<=arrays[0].max():raise GateError('Gate 5: validation decisions precede optimizer data availability.')
    if pd.Timestamp(holdout['decision_at'][0])<=arrays[1].max():raise GateError('Gate 6: holdout decisions precede selection freeze.')
    for d in [validation,holdout]:
        for k in ['site_ids','model_ids','forecast_round','lineage']:
            if d[k]!=development[k]:raise GateError('Gate 3: incompatible models, plant order, horizon policy or round.')


def require_plant_lineage(lineage):
    if lineage!='plant_level':raise GateError('Gate 3: direct group forecasts cannot be reused after reassignment.')


def input_manifest(paths,expected_hashes):
    result={}
    for name,path in paths.items():
        p=Path(path)
        if not p.is_file():raise GateError('Gate 1: missing upstream artifact '+name)
        value=hashlib.sha256(p.read_bytes()).hexdigest()
        if expected_hashes.get(name)!=value:raise GateError('Gate 1: upstream version/hash mismatch '+name)
        result[name]={'path':str(p.resolve()),'sha256':value}
    return result
