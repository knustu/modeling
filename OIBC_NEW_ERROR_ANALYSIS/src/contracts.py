import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['model_id','forecast_round','validation_fold']
PAIR=KEY+['site_id','ts']


class DiagnosticError(ValueError):
    pass


def require(frame,columns):
    if not set(columns)<=set(frame):
        raise DiagnosticError('Missing columns: '+str(sorted(set(columns)-set(frame))))
    if frame[columns].isna().any().any():
        raise DiagnosticError('Missing required identifiers or provenance.')


def timestamps(values,hourly=False):
    result=[]
    for value in values:
        if isinstance(value,(int,float,np.number)): raise DiagnosticError('Epoch unit must be explicitly converted.')
        t=pd.Timestamp(value)
        if pd.isna(t) or t.tzinfo is None: raise DiagnosticError('Timezone-aware timestamps required.')
        t=t.tz_convert('Asia/Seoul')
        if hourly and (t.minute or t.second or t.microsecond or t.nanosecond):
            raise DiagnosticError('Nonhourly timestamp.')
        result.append(t)
    return pd.DatetimeIndex(result)


def numbers(series,allow_missing=False):
    x=pd.to_numeric(series,errors='coerce').to_numpy(float)
    if (series.notna().to_numpy() & np.isnan(x)).any() or np.isinf(x).any():
        raise DiagnosticError('Invalid numeric value.')
    if not allow_missing and not np.isfinite(x).all(): raise DiagnosticError('Missing numeric value.')
    return x


def sites_contract(sites,cfg):
    require(sites,['site_id','capacity_kw'])
    s=sites[['site_id','capacity_kw']].copy();s['site_id']=s.site_id.astype(str)
    if s.site_id.duplicated().any() or s.site_id.str.strip().eq('').any(): raise DiagnosticError('Duplicate/empty plant ID.')
    if len(s)!=cfg['expected_sites']: raise DiagnosticError('Plant count mismatch.')
    s['capacity_kw']=numbers(s.capacity_kw)
    if (s.capacity_kw<=0).any(): raise DiagnosticError('Invalid capacity.')
    return s


def composition_contract(frame,sites,cfg):
    if list(frame.columns)!=['site_id','group']: raise DiagnosticError('Expected site_id,group composition.')
    require(frame,['site_id','group'])
    c=frame.copy();c['site_id']=c.site_id.astype(str)
    if c.site_id.duplicated().any() or set(c.site_id)!=set(sites.site_id): raise DiagnosticError('Misaligned grouping IDs.')
    if not c.group.isin(['G1','G2']).all(): raise DiagnosticError('Unknown group.')
    caps=c.merge(sites,on='site_id',validate='one_to_one').groupby('group').capacity_kw.sum()
    if set(caps.index)!={'G1','G2'} or (caps<cfg['minimum_group_capacity_kw']).any():
        raise DiagnosticError('Group capacity violation.')
    return c.set_index('site_id').loc[sites.site_id].reset_index()


def validate_plant(frame,sites,cfg,expected_grid,synthetic=False):
    require(frame,PAIR+['decision_at','trained_through','prediction_kind','horizon_hours'])
    for name in ['forecast_kwh','actual_kwh']:
        if name not in frame: raise DiagnosticError('Missing '+name)
    if len(frame)>cfg['maximum_input_rows']: raise DiagnosticError('Input row budget exceeded; partition by fold/model.')
    f=frame.copy();f['site_id']=f.site_id.astype(str)
    if f.duplicated(PAIR).any(): raise DiagnosticError('Duplicate analytical key.')
    if set(f.site_id)!=set(sites.site_id): raise DiagnosticError('Missing/unknown plant IDs.')
    f['ts']=timestamps(f.ts,True)
    f['decision_at']=timestamps(f.decision_at);f['trained_through']=timestamps(f.trained_through)
    allowed=['synthetic_oos'] if synthetic else ['oos']
    if not f.prediction_kind.isin(allowed).all(): raise DiagnosticError('Only independently evaluated forecasts accepted.')
    if (f.trained_through>=f.decision_at).any() or (f.decision_at>=f.ts-pd.Timedelta(hours=1)).any():
        raise DiagnosticError('Training/issuance leakage or interval mismatch.')
    horizon=numbers(f.horizon_hours)
    actual_horizon=(f.ts-f.decision_at).dt.total_seconds().to_numpy()/3600
    if not np.allclose(horizon,actual_horizon,rtol=0,atol=1e-9): raise DiagnosticError('Horizon metadata mismatch.')
    if (f.ts>=pd.Timestamp(cfg['label_cutoff_exclusive'])).any(): raise DiagnosticError('Evaluation-period actuals prohibited.')
    for col in ['forecast_kwh','actual_kwh']:
        f[col]=numbers(f[col],True)
        if (f[col].dropna()<0).any(): raise DiagnosticError('Negative energy.')
    grid=timestamps(expected_grid,True)
    if not len(grid) or grid.duplicated().any(): raise DiagnosticError('Invalid expected grid.')
    grid=grid.sort_values()
    if len(grid)>1 and not (np.diff(grid.asi8)==3600*10**9).all():
        raise DiagnosticError('Expected grid must be continuous hourly for this partition.')
    for _,g in f.groupby(KEY,sort=True):
        if len(g)!=len(sites)*len(grid) or not np.array_equal(np.sort(g.ts.unique().asi8),grid.asi8):
            raise DiagnosticError('Missing/extra rows: explicit complete site-hour grid required.')
        if not g.groupby('ts').site_id.nunique().eq(len(sites)).all(): raise DiagnosticError('Incomplete site-hour keys.')
        if g.groupby('ts').decision_at.nunique().gt(1).any(): raise DiagnosticError('Inconsistent decision within round.')
    # All models/rounds must refer to exactly the same actuals, including missingness.
    check=f.groupby(['site_id','ts']).actual_kwh.agg(['min','max','count','size'])
    if ((check['max']-check['min']).abs()>1e-9).any() or ((check['count']!=0)&(check['count']!=check['size'])).any():
        raise DiagnosticError('Actual reference differs across comparisons.')
    if not cfg['include_holdout']:
        f=f.loc[f.validation_fold.ne('holdout')].copy()
    if f.empty: raise DiagnosticError('No development validation rows; holdout remains protected.')
    f['error_kwh']=f.forecast_kwh-f.actual_kwh
    f['capacity_kw']=f.site_id.map(sites.set_index('site_id').capacity_kw)
    return f


def phase2_residual_adapter(frame,model_id,fold,provenance):
    """Pure adapter: caller must supply actual verified provenance; never inferred."""
    require(frame,['site_id','ts'])
    if not {'forecast_kwh','energy_kwh'}<=set(frame): raise DiagnosticError('Plant residual input required, not group-only forecast.')
    needed=['decision_at','trained_through','forecast_round','prediction_kind','horizon_hours']
    if not all(k in provenance for k in needed): raise DiagnosticError('Verified issuance/training manifest required.')
    f=frame[['site_id','ts','forecast_kwh','energy_kwh']].rename(columns={'energy_kwh':'actual_kwh'}).copy()
    f['model_id']=model_id;f['validation_fold']=fold
    for key in needed: f[key]=provenance[key]
    return f


def weather_contract(weather,plant,fields):
    key=['site_id','ts','forecast_round','decision_at']
    require(weather,key+['issued_at','available_at'])
    w=weather.copy();w['ts']=timestamps(w.ts,True)
    for col in ['decision_at','issued_at','available_at']:w[col]=timestamps(w[col])
    if w.duplicated(key).any():raise DiagnosticError('Ambiguous weather join.')
    if (w.issued_at>w.decision_at).any() or (w.available_at>w.decision_at).any() or (w.available_at<w.issued_at).any():
        raise DiagnosticError('Weather unavailable at decision.')
    for col in fields:
        if col not in w:raise DiagnosticError('Configured weather field missing.')
        w[col]=numbers(w[col],True)
    return plant.merge(w[key+fields],on=key,how='left',validate='many_to_one')


def verify_group_forecast(expected,supplied):
    keys=KEY+['grouping_id','group','ts']
    require(supplied,keys)
    if supplied.duplicated(keys).any():raise DiagnosticError('Duplicate group forecast.')
    j=expected.merge(supplied,on=keys,how='outer',suffixes=('_expected','_supplied'),indicator=True,validate='one_to_one')
    if not j['_merge'].eq('both').all():raise DiagnosticError('Group-hour alignment mismatch.')
    for col in ['forecast_kwh','actual_kwh']:
        a=numbers(j[col+'_expected'],True);b=numbers(j[col+'_supplied'],True)
        if not np.allclose(a,b,equal_nan=True,rtol=1e-10,atol=1e-8):raise DiagnosticError('Incorrect plant-to-group aggregation.')


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
