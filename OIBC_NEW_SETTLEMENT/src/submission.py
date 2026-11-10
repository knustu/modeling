from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import GateError,composition,time_index,save_json


def grid():return pd.date_range('2025-06-01T01:00:00+09:00',periods=8760,freq='h')


def validate(comp,forecasts,sites,cfg):
    composition(comp,sites,cfg)
    if list(forecasts.columns)!=['group','ts','forecast_kwh'] or len(forecasts)!=17520:raise GateError('Gate 7: required forecast schema/row count.')
    if set(forecasts.group)!={'G1','G2'} or forecasts.duplicated(['group','ts']).any():raise GateError('Gate 7: duplicate/invalid group hours.')
    if not forecasts.ts.map(lambda v:isinstance(v,str) and 'T' in v and v.endswith('+09:00')).all():raise GateError('Gate 7: explicit KST ISO text required.')
    for name in ['G1','G2']:
        actual=time_index(forecasts.loc[forecasts.group.eq(name),'ts']).sort_values().tz_convert('UTC')
        expected=grid().tz_convert('UTC')
        if len(actual)!=len(expected) or not (actual==expected).all():raise GateError('Gate 7: wrong target horizon.')
    values=pd.to_numeric(forecasts.forecast_kwh,errors='coerce').to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any():raise GateError('Gate 7: finite nonnegative predictions required.')
    return {'valid_custom_contract':True,'official_validator_executed':False,'official_rules_verified':False,
            'rows':17520,'data_mode':'synthetic','submittable':False}


def build_synthetic(comp,sites,cfg,output):
    out=Path(output)
    if out.exists():raise FileExistsError('Never overwrite candidate output.')
    # Deliberately zero-valued FORMAT FIXTURE, not a real forecast or final candidate.
    f=pd.concat([pd.DataFrame({'group':g,'ts':[t.isoformat() for t in grid()],'forecast_kwh':0.}) for g in ['G1','G2']],ignore_index=True)
    report=validate(comp,f,sites,cfg)
    out.mkdir(parents=True,exist_ok=False)
    comp.to_csv(out/'composition.csv',index=False);f.to_csv(out/'forecast.csv',index=False)
    save_json(out/'submission_validation.json',report)
    (out/'NON_SUBMITTABLE_SYNTHETIC.txt').write_text('Synthetic format fixture. Never submit. No real inference performed.',encoding='utf-8')
    return report
