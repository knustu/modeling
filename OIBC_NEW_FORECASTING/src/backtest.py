"""Expanding-origin comparisons under common plant forecasts; no real CLI execution."""
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .core import (ContractError,metadata,label_rows,feature_frame,expanding_folds,
    Forecaster,metrics,aggregate,grouping,write_json,official_grid,validate_forecast)


def execute(sites,targets,labels,weather,compositions,cfg,out,boundaries,holdout,synthetic=True):
    if not synthetic:
        raise ContractError('Real execution locked until official rules and local validation are verified.')
    out=Path(out)
    if out.exists(): raise FileExistsError('Use a new output directory; existing files are never overwritten.')
    sites=metadata(sites,cfg); labels=label_rows(labels,sites,cfg)
    for frame in compositions.values(): grouping(frame,sites,cfg)
    t,X=feature_frame(targets,sites,weather,cfg)
    joined=t.merge(labels,on=['site_id','ts'],how='left',validate='one_to_one')
    if joined.label_available_at.isna().any(): raise ContractError('Missing label availability metadata.')
    capacity=t.site_id.map(sites.set_index('site_id').capacity_kw).to_numpy()
    boundaries=list(boundaries)
    if boundaries and pd.Timestamp(boundaries[-1][1])>pd.Timestamp(holdout[0]):
        raise ContractError('Holdout overlaps development validation.')
    folds=list(expanding_folds(joined,boundaries+[holdout]))
    kinds=['climatology','ridge','hist_gradient_boosting']
    # Verify candidate dependency BEFORE creating outputs. Never install packages.
    try:
        import sklearn
    except ImportError as exc: raise ContractError('Already-installed scikit-learn required for stronger candidate.') from exc
    out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir()
    model_rows=[]; group_rows=[]; diagnostics=[]; traces=[]; persisted={}
    for fold_number,(origin,train,valid) in enumerate(folds):
        split='holdout' if fold_number==len(folds)-1 else f'validation_{fold_number}'
        train=train & joined.energy_kwh.notna().to_numpy()
        if not train.any(): raise ContractError('No observed training labels.')
        # Validation excludes entire incomplete hours consistently across ALL models/groups.
        observed=joined.loc[valid].groupby('ts').energy_kwh.count()
        valid=valid & joined.ts.isin(observed[observed.eq(len(sites))].index).to_numpy()
        if not valid.any(): raise ContractError('No fully observed validation hours.')
        traces.append({'split':split,'origin':origin.isoformat(),'train_rows':int(train.sum()),
                       'validation_rows':int(valid.sum()),'training_max_ts':str(joined.loc[train,'ts'].max()),
                       'training_max_availability':str(joined.loc[train,'label_available_at'].max())})
        for kind in kinds:
            model=Forecaster(kind,cfg)
            start=time.perf_counter()
            model.fit(t.loc[train],X.loc[train],joined.loc[train,'energy_kwh'],capacity[train])
            pred=[]; indices=np.flatnonzero(valid)
            for k in range(0,len(indices),cfg['batch_size']):
                batch=indices[k:k+cfg['batch_size']]
                pred.extend(model.predict(t.iloc[batch],X.iloc[batch],capacity[batch]))
            pred=np.asarray(pred)
            score=metrics(joined.loc[valid,'energy_kwh'],pred,capacity[valid])
            model_rows.append(dict(score,model=kind,split=split,runtime_seconds=time.perf_counter()-start))
            plant=t.loc[valid,['site_id','ts']].copy(); plant['forecast_kwh']=pred
            actual=t.loc[valid,['site_id','ts']].copy(); actual['energy_kwh']=joined.loc[valid,'energy_kwh'].to_numpy()
            saved=plant.copy(); saved['ts']=saved.ts.map(lambda v:v.isoformat())
            saved.to_csv(out/f'plant_forecasts_{split}_{kind}.csv',index=False)
            err=plant.copy(); err['energy_kwh']=actual.energy_kwh.to_numpy()
            err['error_kwh']=err.forecast_kwh-err.energy_kwh
            err['hour']=err.ts.dt.hour; err['month']=err.ts.dt.month
            err['daytime_diagnostic']=err.hour.isin(cfg['daytime_diagnostic_hours'])
            err['absolute_error_kwh']=err.error_kwh.abs()
            err.groupby(['hour','month','daytime_diagnostic']).agg(count=('error_kwh','size'),mae=('absolute_error_kwh','mean'),bias=('error_kwh','mean')).to_csv(out/f'error_statistics_{split}_{kind}.csv')
            err.to_csv(out/f'plant_errors_{split}_{kind}.csv',index=False)
            for name,composition in compositions.items():
                grouped=aggregate(plant,composition,sites,cfg)
                truth=aggregate(actual,composition,sites,cfg,'energy_kwh')
                both=grouped.merge(truth,on=['group','ts'],validate='one_to_one')
                both['model']=kind; both['candidate']=name; both['split']=split
                caps=composition.merge(sites[['site_id','capacity_kw']],on='site_id').groupby('group').capacity_kw.sum()
                for g,f in both.groupby('group'):
                    group_rows.append(dict(metrics(f.energy_kwh,f.forecast_kwh,np.full(len(f),caps[g])),
                        model=kind,candidate=name,split=split,group=g))
                both.to_csv(out/f'group_forecasts_{split}_{kind}_{name}.csv',index=False)
                diagnostics.append(both)
            if split=='holdout':
                # One checkpoint per model, no arbitrary-code loading interface.
                with (out/'models'/f'{kind}.pkl').open('xb') as handle:
                    pickle.dump(model,handle,protocol=pickle.HIGHEST_PROTOCOL)
                persisted[kind]=model
    pd.DataFrame(model_rows).to_csv(out/'model_comparison.csv',index=False)
    pd.DataFrame(group_rows).to_csv(out/'grouping_forecast_comparison.csv',index=False)
    # Full-year FORMAT EXAMPLE from synthetic TRAINING climatology, not invented future labels.
    # This is not an official-period real-data inference run or scored experiment.
    comp=next(iter(compositions.values()))
    model=persisted['climatology']; grid=official_grid(); forecasts=[]
    for g in ('G1','G2'):
        subset=sites.loc[sites.site_id.isin(comp.loc[comp.group.eq(g),'site_id'])]
        profile={}
        for hour in range(24):
            profile[hour]=sum(model.site_hour.get((r.site_id,hour),model.hour.get(hour,model.fallback)) *
                (r.capacity_kw if model.normalized else 1.) for r in subset.itertuples())
        forecasts.append(pd.DataFrame({'group':g,'ts':[v.isoformat() for v in grid],
                                       'forecast_kwh':[max(0.,profile[h]) for h in grid.hour]}))
    full=pd.concat(forecasts,ignore_index=True)
    report=validate_forecast(full); full.to_csv(out/'forecast.csv',index=False)
    write_json(out/'validation_report.json',dict(report,data_mode='synthetic',tests_executed=False,
        browser_test_executed=False,official_score_calculated=False))
    write_json(out/'backtest_report.json',{'data_mode':'synthetic','folds':traces,
        'holdout_policy':'Fixed algorithms and hyperparameters; holdout never selected by this runner.',
        'weather_policy':'Latest issued AND available no later than each decision.',
        'scoring':'No SMP supplied or synthesized; settlement penalty not calculated.'})
    write_json(out/'model_metadata.json',{'data_mode':'synthetic','config':cfg,'features':list(X.columns),
        'models':kinds,'sklearn':sklearn.__version__,'approach':'plant_level',
        'checkpoints':'Locally trusted pickle only; never load untrusted pickles.',
        'preprocessing':{k:{'columns':m.pre.columns,'median':m.pre.median,'mean':m.pre.mean,'std':m.pre.std}
                         for k,m in persisted.items() if hasattr(m,'pre')},
        'candidate_ids':list(compositions),'annual_example_model':'training_climatology',
        'official_format_status':'User-summary contract only; official documents unverified.'})
    from .visualizer import render
    render(out/'forecast_diagnostics.html',pd.concat(diagnostics,ignore_index=True),pd.DataFrame(group_rows))
    (out/'forecasting_report.md').write_text('# Synthetic Phase 2 report\n\n'
        'No real competition data was read. No official score was calculated.\n'
        'Three fixed plant-level models were compared across common chronological folds and all fixture compositions.\n'
        'Missing label hours are excluded jointly, not imputed as actual generation.\n'
        'Annual forecast.csv is a synthetic training-climatology FORMAT EXAMPLE only.\n'
        'Tests and offline browser checks are separate; this pipeline does not certify their execution.\n'
        'Direct group forecasting, real source adapters and official-rule confirmation remain pending.\n',encoding='utf-8')
    return model_rows,group_rows
