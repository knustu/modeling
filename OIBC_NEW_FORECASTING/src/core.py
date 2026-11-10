"""Canonical forecasting contracts, information-time guards, models and metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


class ContractError(ValueError):
    pass


def write_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): x=x.item()
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(clean(obj),indent=2,allow_nan=False),encoding='utf-8')


def required(f, columns):
    missing=set(columns)-set(f.columns)
    if missing: raise ContractError(f'Missing canonical columns: {sorted(missing)}')


def times(series):
    result=[]
    for value in series:
        if isinstance(value, (int,float,np.number)): raise ContractError('Numeric epochs need explicit source conversion.')
        t=pd.Timestamp(value)
        if pd.isna(t) or t.tzinfo is None: raise ContractError('Timezone-aware timestamps required.')
        result.append(t.tz_convert('Asia/Seoul'))
    return pd.DatetimeIndex(result)


def hourly(series):
    t=times(series)
    if any(v.minute or v.second or v.microsecond or v.nanosecond for v in t):
        raise ContractError('Hourly boundary required.')
    return t


def numeric(series, missing=False):
    v=pd.to_numeric(series,errors='coerce').to_numpy(float)
    if (series.notna().to_numpy() & np.isnan(v)).any() or np.isinf(v).any():
        raise ContractError('Invalid numeric value.')
    if not missing and not np.isfinite(v).all(): raise ContractError('Missing numeric value.')
    return v


def metadata(sites, cfg):
    required(sites,['site_id','capacity_kw'])
    s=sites.copy()
    if s.site_id.isna().any(): raise ContractError('Missing site ID.')
    s['site_id']=s.site_id.astype(str)
    if s.site_id.str.strip().eq('').any() or s.site_id.duplicated().any():
        raise ContractError('Empty or duplicate site ID.')
    if len(s)!=cfg['expected_sites']: raise ContractError('Unexpected site count.')
    s['capacity_kw']=numeric(s.capacity_kw)
    if (s.capacity_kw<=0).any(): raise ContractError('Nonpositive capacity.')
    return s.reset_index(drop=True)


def grouping(frame, sites, cfg):
    if list(frame.columns)!=['site_id','group']: raise ContractError('Grouping columns must be site_id,group.')
    f=frame.copy()
    if f.site_id.isna().any(): raise ContractError('Missing grouping ID.')
    f['site_id']=f.site_id.astype(str)
    if f.site_id.duplicated().any() or set(f.site_id)!=set(sites.site_id):
        raise ContractError('Grouping must cover every site exactly once.')
    if not f.group.isin(['G1','G2']).all(): raise ContractError('Invalid group label.')
    cap=f.merge(sites[['site_id','capacity_kw']],on='site_id',validate='one_to_one').groupby('group').capacity_kw.sum()
    if set(cap.index)!={'G1','G2'} or (cap<cfg['minimum_capacity_kw']).any():
        raise ContractError('Group capacity violation.')
    # Preserve authoritative metadata order without changing the input file.
    return f.set_index('site_id').loc[sites.site_id].reset_index()


def load_grouping(path, sites, cfg):
    p=Path(path)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    return grouping(pd.read_csv(p,dtype=str,keep_default_na=False),sites,cfg),digest


def target_rows(frame, sites):
    required(frame,['site_id','ts','decision_at'])
    f=frame.copy()
    f['site_id']=f.site_id.astype(str)
    f['ts']=hourly(f.ts)
    f['decision_at']=times(f.decision_at)
    if f.duplicated(['site_id','ts']).any(): raise ContractError('Duplicate site-target rows.')
    if not set(f.site_id)<=set(sites.site_id): raise ContractError('Unknown target site.')
    if (f.decision_at>=f.ts-pd.Timedelta(hours=1)).any():
        raise ContractError('Decision must precede target interval start.')
    if f.groupby('ts').decision_at.nunique().gt(1).any():
        raise ContractError('Inconsistent decisions for a target hour.')
    return f.reset_index(drop=True)


def label_rows(frame, sites, cfg):
    required(frame,['site_id','ts','energy_kwh','label_available_at'])
    f=frame.copy()
    f['site_id']=f.site_id.astype(str)
    f['ts']=hourly(f.ts)
    f['label_available_at']=times(f.label_available_at)
    f['energy_kwh']=numeric(f.energy_kwh,missing=True)
    if f.duplicated(['site_id','ts']).any(): raise ContractError('Duplicate generation labels.')
    if not set(f.site_id)<=set(sites.site_id): raise ContractError('Unknown label site.')
    if (f.energy_kwh.dropna()<0).any(): raise ContractError('Negative generation label.')
    if (f.label_available_at<f.ts).any(): raise ContractError('Label available before interval end.')
    if (f.ts>=pd.Timestamp(cfg['label_cutoff'])).any():
        raise ContractError('Evaluation-period labels prohibited.')
    return f


def feature_frame(targets, sites, forecasts, cfg):
    t=target_rows(targets,sites)
    joined=t.merge(sites,on='site_id',validate='many_to_one')
    ts=joined.ts
    h=ts.dt.hour.to_numpy()
    day=ts.dt.dayofyear.to_numpy()
    X=pd.DataFrame({'hour_sin':np.sin(2*np.pi*h/24), 'hour_cos':np.cos(2*np.pi*h/24),
                    'year_sin':np.sin(2*np.pi*day/365.2425), 'year_cos':np.cos(2*np.pi*day/365.2425),
                    'capacity_kw':joined.capacity_kw.to_numpy()})
    # No coordinates, generation lags, learned plant statistics, or target labels.
    for provider,spec in cfg['providers'].items():
        station=spec['station_column']
        required(joined,[station])
        weather=forecasts.get(provider)
        selected=None
        if weather is not None and len(weather):
            required(weather,['station_id','ts','issued_at','available_at'])
            w=weather.copy()
            if w.station_id.isna().any(): raise ContractError('Missing weather station ID.')
            w['station_id']=w.station_id.astype(str)
            w['ts']=hourly(w.ts)
            w['issued_at']=times(w.issued_at)
            w['available_at']=times(w.available_at)
            if (w.available_at<w.issued_at).any(): raise ContractError('Forecast availability precedes issuance.')
            if w.duplicated(['station_id','ts','issued_at']).any():
                raise ContractError('Duplicate station-target-issuance forecast.')
            # Select latest legitimately available revision at EACH decision time.
            keys=joined[[station,'ts','decision_at']].copy()
            keys[station]=keys[station].astype(str)
            keys['row_id']=np.arange(len(keys))
            match=keys.merge(w,left_on=[station,'ts'],right_on=['station_id','ts'],how='left')
            match=match[(match.issued_at<=match.decision_at)&(match.available_at<=match.decision_at)]
            selected=match.sort_values(['issued_at','available_at']).drop_duplicates('row_id',keep='last').set_index('row_id')
        for name,rule in spec['fields'].items():
            col=rule['column']
            if not rule.get('unit'): raise ContractError('Canonical weather unit required.')
            factor=float(rule['scale']); offset=float(rule['offset'])
            if not np.isfinite([factor,offset]).all() or factor==0: raise ContractError('Invalid unit conversion.')
            if selected is None:
                value=np.full(len(t),np.nan)
            else:
                required(selected,[col])
                values=numeric(selected[col],missing=True)*factor+offset
                values=pd.Series(values,index=selected.index).reindex(range(len(t)))
                value=values.to_numpy(float)
                if np.isinf(value).any(): raise ContractError('Converted weather overflow.')
            key=provider+'__'+name
            X[key]=value
            X[key+'__missing']=np.isnan(value).astype(float)
    return t,X


class Preprocessor:
    def fit(self,X):
        self.columns=list(X.columns)
        a=X.to_numpy(float)
        self.median=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in a.T])
        a=np.where(np.isnan(a),self.median,a)
        self.mean=a.mean(axis=0); self.std=a.std(axis=0)
        self.std[self.std<1e-12]=1.
        return self
    def transform(self,X):
        if list(X.columns)!=self.columns: raise ContractError('Feature order mismatch.')
        a=X.to_numpy(float)
        if np.isinf(a).any(): raise ContractError('Infinite features.')
        return (np.where(np.isnan(a),self.median,a)-self.mean)/self.std


class Forecaster:
    def __init__(self,kind,cfg): self.kind=kind; self.cfg=cfg
    def fit(self,targets,X,energy,capacity):
        y=np.asarray(energy,float)
        cap=np.asarray(capacity,float)
        if not np.isfinite(y).all() or (y<0).any(): raise ContractError('Invalid fit labels.')
        self.normalized=self.cfg['target_normalization']
        y=y/cap if self.normalized else y
        self.columns=list(X.columns)
        if self.kind=='climatology':
            f=pd.DataFrame({'site':targets.site_id.to_numpy(),'hour':targets.ts.dt.hour.to_numpy(),'y':y})
            self.site_hour=f.groupby(['site','hour']).y.mean().to_dict()
            self.hour=f.groupby('hour').y.mean().to_dict(); self.fallback=float(y.mean())
        else:
            self.pre=Preprocessor().fit(X)
            a=self.pre.transform(X)
            if self.kind=='ridge':
                design=np.column_stack([np.ones(len(a)),a])
                penalty=np.eye(design.shape[1])*self.cfg['ridge_alpha']; penalty[0,0]=0.
                self.coef=np.linalg.solve(design.T@design+penalty,design.T@y)
            elif self.kind=='hist_gradient_boosting':
                try:
                    from sklearn.ensemble import HistGradientBoostingRegressor
                except ImportError as exc:
                    raise ContractError('Stronger candidate requires already-installed scikit-learn; no installation attempted.') from exc
                self.estimator=HistGradientBoostingRegressor(max_iter=self.cfg['candidate_iterations'],
                    max_leaf_nodes=15,l2_regularization=1.,early_stopping=False,random_state=self.cfg['seed'])
                self.estimator.fit(a,y)
            else: raise ContractError('Unknown model.')
        return self
    def predict(self,targets,X,capacity):
        if list(X.columns)!=self.columns: raise ContractError('Feature order mismatch.')
        if self.kind=='climatology':
            pred=np.array([self.site_hour.get((site,h),self.hour.get(h,self.fallback))
                for site,h in zip(targets.site_id,targets.ts.dt.hour)])
        else:
            a=self.pre.transform(X)
            pred=np.column_stack([np.ones(len(a)),a])@self.coef if self.kind=='ridge' else self.estimator.predict(a)
        pred=pred*np.asarray(capacity) if self.normalized else pred
        if not np.isfinite(pred).all(): raise ContractError('Nonfinite predictions.')
        return np.maximum(pred,0.)


def plant_predictions(frame, sites, value='forecast_kwh'):
    required(frame,['site_id','ts',value])
    f=frame.copy(); f['ts']=hourly(f.ts)
    if f.duplicated(['site_id','ts']).any(): raise ContractError('Duplicate plant prediction.')
    if set(f.site_id)!=set(sites.site_id): raise ContractError('Incomplete plant IDs.')
    if not f.groupby('ts').site_id.nunique().eq(len(sites)).all(): raise ContractError('Incomplete plant-hour coverage.')
    v=numeric(f[value])
    if (v<0).any(): raise ContractError('Negative values.')
    f[value]=v
    return f


def aggregate(frame, composition, sites, cfg, value='forecast_kwh'):
    f=plant_predictions(frame,sites,value)
    comp=grouping(composition,sites,cfg)
    joined=f.merge(comp,on='site_id',validate='many_to_one')
    return joined.groupby(['group','ts'],as_index=False,sort=True)[value].sum()


def metrics(actual,predicted,capacity=None):
    a=np.asarray(actual,float); p=np.asarray(predicted,float)
    if len(a)!=len(p) or not len(a) or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ContractError('Metrics require nonempty finite aligned values.')
    e=p-a
    out={'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.mean(e**2)))}
    if capacity is not None:
        c=np.asarray(capacity,float)
        if not np.isfinite(c).all() or (c<=0).any(): raise ContractError('Invalid normalization capacities.')
        out['nmae_capacity']=float(np.mean(np.abs(e)/c))
    return out


def penalty(actual,predicted,smp,peak_hours):
    if peak_hours is None: raise ContractError('Official peak timestamp convention not verified.')
    a=actual.copy(); p=predicted.copy(); market=smp.copy()
    for f in (a,p,market): f['ts']=hourly(f.ts)
    if a.duplicated(['group','ts']).any() or p.duplicated(['group','ts']).any() or market.ts.duplicated().any():
        raise ContractError('Duplicate scoring keys.')
    required(a,['group','ts','energy_kwh']); required(p,['group','ts','forecast_kwh']); required(market,['ts','smp'])
    j=a.merge(p,on=['group','ts'],how='outer',validate='one_to_one',indicator=True)
    if not j['_merge'].eq('both').all(): raise ContractError('Unmatched scoring labels/predictions.')
    j=j.drop(columns='_merge').merge(market,on='ts',how='left',validate='many_to_one')
    values=[numeric(j[k]) for k in ['energy_kwh','forecast_kwh','smp']]
    multiplier=np.where(j.ts.dt.hour.isin(peak_hours),3.,1.)
    return float(.1*np.sum(values[2]*multiplier*np.abs(values[1]-values[0])))


def expanding_folds(targets,boundaries):
    # Explicit origin and exclusive validation end; decisions must also be at/after origin.
    previous_end=None
    for origin,end in boundaries:
        origin=pd.Timestamp(origin); end=pd.Timestamp(end)
        if origin.tzinfo is None or end.tzinfo is None or end<=origin:
            raise ContractError('Invalid fold boundaries.')
        if previous_end is not None and origin<previous_end: raise ContractError('Overlapping validation folds.')
        train=(targets.ts<origin)&(targets.label_available_at<=origin)
        valid=(targets.ts>=origin)&(targets.ts<end)&(targets.decision_at>=origin)
        if not train.any() or not valid.any(): raise ContractError('Empty chronological fold.')
        if targets.loc[train,'ts'].max()>=targets.loc[valid,'ts'].min(): raise ContractError('Temporal overlap.')
        yield origin,train.to_numpy(),valid.to_numpy()
        previous_end=end


def official_grid():
    return pd.date_range('2025-06-01T01:00:00+09:00',periods=8760,freq='h')


def validate_forecast(frame):
    if list(frame.columns)!=['group','ts','forecast_kwh']: raise ContractError('Invalid forecast columns.')
    if len(frame)!=17520 or set(frame.group)!={'G1','G2'}: raise ContractError('Expected 17520 rows for G1/G2.')
    if frame.duplicated(['group','ts']).any(): raise ContractError('Duplicate group-hour.')
    # Require textual ISO offset rather than silently normalizing a wrong export.
    if not frame.ts.map(lambda x:isinstance(x,str) and x.endswith('+09:00') and 'T' in x).all():
        raise ContractError('ISO 8601 +09:00 timestamps required.')
    for g in ('G1','G2'):
        if not hourly(frame.loc[frame.group.eq(g),'ts']).sort_values().equals(official_grid()):
            raise ContractError('Wrong official target grid.')
    if (numeric(frame.forecast_kwh)<0).any(): raise ContractError('Negative forecast.')
    return {'rows':17520,'hours_per_group':8760,'valid_summary_contract':True,'official_document_verified':False}
