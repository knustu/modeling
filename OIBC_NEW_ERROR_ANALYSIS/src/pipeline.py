import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import (KEY,DiagnosticError,sites_contract,composition_contract,validate_plant,weather_contract)
from .analysis import (summarize,portfolios,interactions,temporal,weather_summaries,weighted,penalty_tables)
from .statistics import compare_models


def write_json(path,value):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [clean(x) for x in v]
        if isinstance(v,np.generic):v=v.item()
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False,default=str),encoding='utf-8')


def run(sites,plant,compositions,grid,weather,cfg,output,synthetic=True,smp=None):
    if not synthetic:raise DiagnosticError('Historical execution disabled until separately authorized and official rules verified.')
    sites=sites_contract(sites,cfg)
    clean=validate_plant(plant,sites,cfg,grid,synthetic=True)
    comps={name:composition_contract(frame,sites,cfg) for name,frame in compositions.items()}
    if not comps:raise DiagnosticError('No grouping candidate supplied.')
    out=Path(output)
    if out.exists():raise FileExistsError('Never overwrite an existing output directory.')
    groups=[];correlations=[];contributions=[];assigned=[]
    for name,comp in comps.items():
        group,plants=portfolios(clean,comp,name);groups.append(group)
        corr,contrib=interactions(plants,cfg)
        corr['grouping_id']=name;contrib['grouping_id']=name;plants['grouping_id']=name
        correlations.append(corr);contributions.append(contrib);assigned.append(plants)
    group=pd.concat(groups,ignore_index=True)
    corr=pd.concat(correlations,ignore_index=True);contrib=pd.concat(contributions,ignore_index=True)
    ps=summarize(clean,KEY+['site_id'])
    ps['individual_mae_rank']=ps.groupby(KEY).mae.rank(ascending=False,method='min')
    ps['normalized_mae_rank']=ps.groupby(KEY).normalized_mae.rank(ascending=False,method='min')
    if len(contrib):
        contrib['interaction_rank']=contrib.groupby(KEY+['grouping_id','group']).absolute_error_attribution.rank(ascending=False,method='min')
    gs=summarize(group,KEY+['grouping_id','group'])
    temp=temporal(group,cfg)
    joined=weather_contract(weather,clean,list(cfg['weather_bins'])) if weather is not None else None
    ws=weather_summaries(joined,cfg) if joined is not None else pd.DataFrame()
    scored,score_status=weighted(group,smp,cfg)
    comparison=compare_models(group,cfg)
    # Findings are descriptive. No data-derived deployment threshold or correction rule is fitted.
    findings=[]
    for row in gs.to_dict('records'):
        if row['sample_count']==0:continue
        for category,predicate in [('A',row['me']>0),('B',row['me']<0)]:
            if not predicate:continue
            findings.append({'finding_id':f'F{len(findings)+1:04d}','model_id':row['model_id'],
                'grouping_id':row['grouping_id'],'validation_fold':row['validation_fold'],'forecast_round':row['forecast_round'],
                'error_category':category,'affected_group':row['group'],'affected_period':f'{group.ts.min()} / {group.ts.max()}',
                'severity':'descriptive_not_significance','supporting_metrics':{'me':row['me'],'mae':row['mae']},
                'sample_count':row['sample_count'],'statistical_uncertainty':'Mean bias alone is not significance; see paired model intervals, which do not test bias.',
                'recommended_investigation':'Test directional bias consistency on NEW development windows before any correction.',
                'evidence':'Observed signed mean on synthetic fixtures only.','hypothesis':'Possible persistent bias; not established on historical competition data.',
                'leakage_risk':'Do not tune corrections on final holdout.'})
    out.mkdir(parents=True,exist_ok=False)
    tables={'plant_error_summary':ps,'group_error_summary':gs,'hourly_error_summary':temp['hour'],
        'monthly_error_summary':temp['month'],'seasonal_error_summary':temp['season'],'daily_error_summary':temp['daily'],
        'day_of_year_summary':temp['day_of_year'],'daytime_error_summary':temp['daytime'],
        'error_correlation':corr,'plant_interaction_summary':contrib,'cancellation_analysis':group,
        'grouping_error_comparison':gs,'forecasting_model_comparison':gs,'weather_error_summary':ws,
        'error_autocorrelation':temp['acf'],'error_episodes':temp['episodes'],'error_pattern_report':pd.DataFrame(findings)}
    for name,table in tables.items():
        table=table.copy();table['data_mode']='synthetic';table['config_version']=cfg['version']
        table.to_csv(out/(name+'.csv'),index=False)
    if scored is None:
        for name in ['penalty_by_hour','penalty_by_month','penalty_concentration']:
            pd.DataFrame([dict(score_status,data_mode='synthetic')]).to_csv(out/(name+'.csv'),index=False)
    else:
        penalty=penalty_tables(scored)
        for key,name in [('hour','penalty_by_hour'),('month','penalty_by_month'),('concentration','penalty_concentration')]:
            table=penalty[key];table['data_mode']='synthetic';table.to_csv(out/(name+'.csv'),index=False)
    write_json(out/'statistical_comparison.json',comparison)
    write_json(out/'diagnostic_findings.json',findings)
    write_json(out/'phase4_handoff.json',{'status':'diagnostic_hypotheses_only','corrections_implemented':False,
        'grouping_changes':False,'real_data_evidence':False,'priorities':[
        {'priority':1,'hypothesis':'Directional bias may persist within forecast conditions.','required_experiment':'New chronological development folds; block-aware bias uncertainty.','mechanism':'Repeated signed errors may reinforce.','evidence':'Synthetic summaries only.'},
        {'priority':2,'hypothesis':'Covariance/absolute attribution may identify portfolio reinforcement.','required_experiment':'Frozen OOS forecasts and independently validated feasible comparisons in a later phase.','mechanism':'Cancellation depends on signed sums, not plant MAE.','evidence':'Diagnostic identities only; no reassignment performed.'},
        {'priority':3,'hypothesis':'Issuance-time weather disagreement may identify higher-risk periods.','required_experiment':'Compatible units, actual forecast provenance, paired windows and multiplicity controls.','mechanism':'Weather uncertainty association, not causality.','evidence':'No real-data association established.'}],
        'holdout_policy':'No Phase 4 tuning against final holdout; new data required after repeated inspection.'})
    write_json(out/'validation_report.json',{'synthetic':True,'unit_tests_executed_by_runner':False,'browser_test_executed':False,
        'plant_rows':len(clean),'group_rows':len(group),'expected_site_count':len(sites),'score_status':score_status,
        'missing_plant_predictions':int(clean.forecast_kwh.isna().sum()),'missing_actuals':int(clean.actual_kwh.isna().sum())})
    write_json(out/'run_metadata.json',{'config':cfg,'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'pandas':pd.__version__,'period_start':str(clean.ts.min()),'period_end':str(clean.ts.max()),
        'input_mode':'synthetic','original_artifacts_modified':False,'metric_definitions':{
        'signed_error':'forecast minus actual','std':'population ddof=0','normalization':'installed kW, never actual energy',
        'absolute_attribution':'mean(e_i * sign(sum e_i)); sums to group MAE on identical complete hours',
        'variance_contribution':'Cov(e_i,sum e_j), ddof=0; sums to aggregate variance',
        'cancellation':'sum(abs(e_i))-abs(sum(e_i)); fraction missing when denominator is zero'}})
    from .visualizer import render
    render(out/'error_analysis_dashboard.html',group,corr,clean,joined,gs,scored,cfg)
    (out/'error_analysis_report.md').write_text('# Synthetic error analysis\n\n'
        'Diagnostic only. No forecasting, correction, training, grouping change or optimization.\n'
        'Group absolute error is abs(sum plant errors), NOT sum(abs plant errors).\n'
        'Complete group hours only; missingness remains visible and paired comparisons refuse unequal coverage.\n'
        'Covariance identities use the same complete hours; zero-variance correlations are undefined.\n'
        'Block intervals use nonoverlapping complete calendar-day blocks, approximately independent only; sensitivity to block size is reported.\n'
        'Short samples and multiple comparisons weaken conclusions. No historical benefit is established by synthetic fixtures.\n'
        'Weather relationships are associations, not causality. Holdout is excluded by default.\n'
        'SMP metrics are unavailable unless explicitly provided with verified timing rules.\n'
        'Tests/browser checks are separate; this report does not claim they passed.\n'
        'Scalability: bounded per-model/fold pivots; input row and dashboard budgets fail explicitly. This version is not a streaming engine.\n',encoding='utf-8')
    return gs
