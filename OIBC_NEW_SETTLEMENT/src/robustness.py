import numpy as np
import pandas as pd
from .contracts import GateError


def paired_bootstrap(candidate,reference,ts,days,reps,seed,minimum=4):
    a=np.asarray(candidate,float);b=np.asarray(reference,float)
    if a.shape!=b.shape or a.shape!=(len(ts),) or not np.isfinite(a).all() or not np.isfinite(b).all():raise GateError('Paired complete hourly losses required.')
    f=pd.DataFrame({'delta':a-b},index=ts);daily=[]
    for date,g in f.groupby(f.index.normalize()):
        if len(g)==24 and set(g.index.hour)==set(range(24)):daily.append((date,float(g.delta.sum())))
    blocks=[];run=[];previous=None
    for date,value in daily:
        if previous is not None and date-previous!=pd.Timedelta(days=1):run=[]
        run.append(value);previous=date
        if len(run)==days:blocks.append(sum(run));run=[]
    out={'block_days':days,'block_count':len(blocks),'observed_delta_W':float((a-b).sum()),'expected_hours':len(a),
         'effective_hours':len(blocks)*days*24,'ci_scope':'total delta over covered blocks only','status':'insufficient_blocks'}
    if len(blocks)<minimum:return dict(out,ci_low=None,ci_high=None)
    rng=np.random.default_rng(seed);v=np.asarray(blocks)
    draws=np.array([v[rng.integers(0,len(v),len(v))].sum() for _ in range(reps)])
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(out,status='available',covered_delta_W=float(v.sum()),ci_low=float(lo),ci_high=float(hi),
                warning='Approximate block independence; exploratory, no multiplicity adjustment.')
