import time
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import GateError,chronological,composition,digest,save_json,require_plant_lineage
from .objective import aggregate,penalty,score
from .grouping_optimizer import search,multistart
from .corrections import Correction
from .ensemble import combine,fit_weights
from .robustness import paired_bootstrap
from .submission import build_synthetic


def run(sites,compositions,parts,cfg,output,synthetic=False):
    if not synthetic:raise GateError('Real execution disabled; separate rule/provenance review and authorization required.')
    out=Path(output)
    if out.exists():raise FileExistsError('Use a new experiment directory.')
    dev,val,hold=[parts[k] for k in ['development','validation','holdout']]
    chronological(dev,val,hold,cfg);require_plant_lineage(dev['lineage'])
    # Check information-time weather inputs even though prices remain retrospective weights only.
    for part in [dev,val,hold]:
        if (part['weather_available_at']>part['decision_at']).any():raise GateError('Gate 3: weather availability leakage.')
    capacity=sites.capacity_kw.to_numpy();initials=[composition(c,sites,cfg) for c in compositions.values()]
    if not initials:raise GateError('Missing Phase 1 reference fixture.')
    reference=initials[0];start=time.perf_counter()
    optimized,group_reports=multistart(dev['predictions'][0]-dev['actual'],dev['weight'],capacity,initials,cfg)
    # Compare relocation-only and pairwise-only independently on same frozen development residuals.
    for mode in ['relocation','swap']:
        _,report=search(dev['predictions'][0]-dev['actual'],dev['weight'],capacity,reference,cfg,cfg['seed'],mode=mode)
        group_reports.append(report)
    candidate_weights,ensemble_reports=fit_weights(dev['predictions'],dev['actual'],dev['weight'],optimized,cfg,'development')
    fixed={'reference':[1.,0.],'equal':[.5,.5],'development_ensemble':candidate_weights,'alternative':[0.,1.]}
    candidates=[];correction_reports=[]
    # Correction is group-specific: separately fitted for each grouping/model combination.
    for group_name,x in [('reference',reference),('optimized',optimized)]:
        for model_name,weights in fixed.items():
            p=aggregate(combine(dev['predictions'],weights),x);truth=aggregate(dev['actual'],x)
            for kind in ['none','additive','multiplicative','hour','weather']:
                correction=Correction(kind,cfg).fit(p,truth,dev['weight'],dev['ts'],'development',dev['weather'])
                ident=f'{group_name}__{model_name}__{kind}'
                version=digest({'assignment':x.tolist(),'weights':weights,'correction':correction.config(),'config':cfg})
                candidates.append({'id':ident,'x':x.copy(),'weights':weights,'correction':correction,'version':version,
                                   'complexity':int(group_name!='reference')+int(model_name not in ['reference','alternative'])+int(kind!='none')})
                correction_reports.append({'candidate_id':ident,'version':version,'development_W':float(penalty(correction.apply(p,dev['ts'],dev['weather']),truth,dev['weight']).sum())})
    records=[];losses={};series=[]
    for candidate in candidates:
        x=candidate['x'];p=aggregate(combine(val['predictions'],candidate['weights']),x)
        forecast=candidate['correction'].apply(p,val['ts'],val['weather']);truth=aggregate(val['actual'],x)
        metrics=score(forecast,truth,val['weight'],val['smp']);loss=penalty(forecast,truth,val['weight']).sum(axis=1)
        losses[candidate['id']]=loss
        record=dict(metrics,candidate_id=candidate['id'],version=candidate['version'],complexity=candidate['complexity'],sample_hours=len(loss),role='validation',data_mode='synthetic')
        daily=pd.Series(loss,index=val['ts']).groupby(val['ts'].normalize()).sum();record['worst_day_W']=float(daily.max());records.append(record)
        for t in range(len(loss)):
            for g in range(2):series.append({'candidate_id':candidate['id'],'ts':val['ts'][t].isoformat(),'group':f'G{g+1}',
                'actual_kwh':float(truth[t,g]),'forecast_kwh':float(forecast[t,g]),'penalty':float(val['weight'][t]*abs(forecast[t,g]-truth[t,g]))})
    baseline_id='reference__reference__none';baseline=next(r for r in records if r['candidate_id']==baseline_id)
    bootstrap=[]
    for r in records:
        r['delta_W']=r['W']-baseline['W'];r['relative_W_improvement']=(-r['delta_W']/baseline['W']) if baseline['W']>0 else None
        for days in cfg['block_days']:
            bootstrap.append(dict(candidate_id=r['candidate_id'],**paired_bootstrap(losses[r['candidate_id']],losses[baseline_id],val['ts'],days,cfg['bootstrap_replicates'],cfg['seed'],cfg['minimum_bootstrap_blocks'])))
    # Conservative prototype: improvement margin plus all available block upper CIs below zero;
    # an unavailable block CI disqualifies promotion. No historical performance label is issued.
    eligible=[]
    for r in records:
        intervals=[v for v in bootstrap if v['candidate_id']==r['candidate_id']]
        if r['W']<baseline['W']*(1-cfg['relative_selection_margin']) and r['worst_day_W']<=baseline['worst_day_W'] and all(v['status']=='available' and v['ci_high']<0 for v in intervals):eligible.append(r)
    chosen=min(eligible,key=lambda r:(r['complexity'],r['W'])) if eligible else baseline
    frozen=next(c for c in candidates if c['id']==chosen['candidate_id'])
    freeze={'candidate_id':frozen['id'],'version':frozen['version'],'selection_role':'synthetic_experiment_only',
            'primary':'unverified','conservative_backup':baseline_id,'experimental':min(records,key=lambda r:r['W'])['candidate_id'],
            'holdout_used_for_selection':False,'selection_warning':'Exploratory search, approximate intervals, no multiplicity adjustment; not real-world evidence.'}
    # Holdout scoring is omitted by default. No labels or fabricated numbers in the report.
    holdout_report={'status':'not_released','evaluated':False,'used_for_selection':False}
    if cfg['release_holdout']:
        p=aggregate(combine(hold['predictions'],frozen['weights']),frozen['x'])
        forecast=frozen['correction'].apply(p,hold['ts'],hold['weather'])
        holdout_report=dict(score(forecast,aggregate(hold['actual'],frozen['x']),hold['weight'],hold['smp']),
            status='synthetic_frozen_assessment',evaluated=True,used_for_selection=False,candidate_id=frozen['id'])
    out.mkdir(parents=True,exist_ok=False)
    table=pd.DataFrame(records);table.to_csv(out/'joint_optimization_results.csv',index=False)
    # Exact A-F identities, not an assertion that effects add.
    ablations={'A':baseline_id,'B':'optimized__reference__none','C':'reference__reference__additive','D':'optimized__reference__additive',
               'E':'optimized__development_ensemble__none','F':'optimized__development_ensemble__additive'}
    ab=table.loc[table.candidate_id.isin(ablations.values())].copy();ab['configuration']=ab.candidate_id.map({v:k for k,v in ablations.items()});ab.to_csv(out/'ablation_report.csv',index=False)
    table.loc[table.candidate_id.str.endswith('__none')].to_csv(out/'ensemble_comparison.csv',index=False)
    pd.DataFrame(correction_reports).to_csv(out/'forecast_correction_results.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='moves'} for r in group_reports]).to_csv(out/'grouping_optimization_results.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(out/'bootstrap_comparison.csv',index=False)
    save_json(out/'grouping_search_logs.json',group_reports);save_json(out/'holdout_evaluation.json',holdout_report)
    save_json(out/'candidate_manifest.json',freeze);save_json(out/'selected_model_config.json',{'weights':frozen['weights'],'model_ids':dev['model_ids'],'version':frozen['version'],'status':'unverified'})
    save_json(out/'selected_correction_config.json',frozen['correction'].config())
    comp=pd.DataFrame({'site_id':sites.site_id,'group':np.where(frozen['x'],'G1','G2')});comp.to_csv(out/'selected_grouping.csv',index=False)
    build_synthetic(comp,sites,cfg,out/'synthetic_submission')
    save_json(out/'robustness_report.json',{'block_days':cfg['block_days'],'bootstrap_replicates':cfg['bootstrap_replicates'],
        'holdout_selection':False,'seasonal_coverage':'Synthetic winter fixtures only; no seasonal robustness claim.',
        'remaining':'Nested repeated folds, multiplicity controls, round comparisons and stability diagnostics require expansion.'})
    save_json(out/'integration_validation.json',{'mode':'synthetic','gates_1_to_5':'canonical fixture validation executed',
        'gate_6':'holdout role/chronology checked; no assessment by default','gate_7':'custom synthetic format check only',
        'official_rules_verified':False,'upstream_files_modified':False,'real_artifact_integration':'pending verified manifests'})
    save_json(out/'experiment_manifest.json',{'config':cfg,'config_hash':digest(cfg),'seed':cfg['seed'],'runtime_seconds':time.perf_counter()-start,
        'periods':{k:[str(v['ts'][0]),str(v['ts'][-1])] for k,v in parts.items()},'trained_through':str(dev['trained_through']),
        'forecast_round':dev['forecast_round'],'lineage':dev['lineage'],'tests_executed_by_runner':False,'browser_verified':False})
    pd.DataFrame(records).assign(experiment_id=digest(cfg)[:12],seed=cfg['seed']).to_csv(out/'experiment_history.csv',index=False)
    from .visualization import dashboard
    dashboard(out/'optimization_dashboard.html',records,series,group_reports,freeze,sites,reference,optimized)
    (out/'final_optimization_report.md').write_text('# SYNTHETIC Phase 4 prototype\n\nNo real competition data, official score, winner or submission.\n'
        'A-F isolates grouping, correction and ensemble changes; group-specific corrections are refitted only on development for each grouping.\n'
        'Model predictions are synthetic immutable fixtures; selection uses validation, not holdout.\n'
        'Holdout metrics omitted unless synthetic release flag set before run.\n'
        'Optimizer development improvement is not independent performance evidence.\n'
        'Official rule verification, real source adapters, full robustness and browser verification remain pending.\n',encoding='utf-8')
    (out/'execution_instructions.md').write_text('Synthetic artifacts only. Never submit. Read project README for commands. No real execution entry point is enabled.',encoding='utf-8')
    return freeze
