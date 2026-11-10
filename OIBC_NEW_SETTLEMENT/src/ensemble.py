import numpy as np
from .contracts import GateError
from .objective import aggregate,penalty


def combine(predictions,weights):
    p=np.asarray(predictions,float);w=np.asarray(weights,float)
    if p.ndim!=3 or w.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1.,rtol=0,atol=1e-12):raise GateError('Gate 3: compatible model arrays and simplex weights required.')
    return np.tensordot(w,p,axes=(0,0))


def fit_weights(predictions,actual,weight,x,cfg,role):
    if role!='development':raise GateError('Gate 5/6: ensemble fit accepts development only.')
    if len(predictions)!=2:raise GateError('This bounded implementation supports exactly two compatible models.')
    truth=aggregate(actual,x);results=[]
    for alpha in cfg['ensemble_grid']:
        w=[float(alpha),float(1-alpha)];p=aggregate(combine(predictions,w),x)
        W=float(penalty(p,truth,weight).sum())
        # Small prespecified search, no time-conditional weights.
        results.append({'weights':w,'development_W':W})
    winner=min(results,key=lambda r:(r['development_W'],sum(v not in [0.,1.] for v in r['weights'])))
    return winner['weights'],results
