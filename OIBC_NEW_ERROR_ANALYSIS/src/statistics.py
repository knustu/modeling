import numpy as np
import pandas as pd
from .contracts import KEY,DiagnosticError


def paired_blocks(a,b,days,replicates,seed,minimum_blocks=4):
    """Nonoverlapping complete calendar-day blocks, paired differences, fixed seed.
    Sample whole blocks, not independent hourly rows. CI concerns historical mean loss only.
    """
    if days<1 or replicates<1:raise DiagnosticError('Positive block size/replicates required.')
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):
        raise DiagnosticError('Paired series must have identical unique keys.')
    if not isinstance(a.index,pd.DatetimeIndex) or a.index.tz is None:raise DiagnosticError('Aware temporal index required.')
    f=pd.DataFrame({'a':a,'b':b}).sort_index()
    if not np.isfinite(f.to_numpy()).all():raise DiagnosticError('Missing comparisons must be reported, not silently dropped.')
    daily=[]
    for date,g in f.groupby(f.index.normalize()):
        if len(g)==24 and set(g.index.hour)==set(range(24)):
            daily.append((date,float((g.a-g.b).mean())))
    blocks=[];run=[];last=None
    for date,value in daily:
        if last is not None and date-last!=pd.Timedelta(days=1):run=[]
        run.append(value);last=date
        if len(run)==days:blocks.append(float(np.mean(run)));run=[]
    result={'block_days':days,'block_count':len(blocks),'observed_difference':float((a-b).mean()),
        'expected_hours':len(f),'effective_hours':len(blocks)*days*24,
        'confidence_level':.95,'assumption':'Approximate independence between calendar blocks; not guaranteed.',
        'selection_warning':'Exploratory intervals, no multiplicity correction; not future guarantees.'}
    if len(blocks)<minimum_blocks:return dict(result,status='insufficient_blocks',ci_low=None,ci_high=None)
    values=np.asarray(blocks);rng=np.random.default_rng(seed)
    boot=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(replicates)])
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(result,status='available',block_sample_difference=float(values.mean()),ci_low=float(lo),ci_high=float(hi))


def compare_models(group,cfg):
    results=[]
    for vals,g in group.groupby(['grouping_id','group','forecast_round','validation_fold']):
        wide=g.pivot(index='ts',columns='model_id',values='error_kwh').sort_index().abs()
        # Do not drop missing pairs; explicitly refuse that comparison.
        models=sorted(wide.columns)
        for i,left in enumerate(models):
            for right in models[i+1:]:
                for days in cfg['block_days']:
                    base=dict(zip(['grouping_id','group','forecast_round','validation_fold'],vals),model_a=left,model_b=right)
                    if wide[[left,right]].isna().any().any():
                        results.append(dict(base,status='incomplete_paired_coverage',expected_hours=len(wide),
                            paired_hours=int(wide[[left,right]].notna().all(axis=1).sum()),block_days=days))
                    else:results.append(dict(base,**paired_blocks(wide[left],wide[right],days,cfg['bootstrap_replicates'],cfg['seed'],cfg['minimum_blocks'])))
    return results
