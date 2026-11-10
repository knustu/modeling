import numpy as np
from .contracts import GateError,time_index


def price_weights(ts,smp,peak_hours):
    t=time_index(ts);market=smp.copy()
    if list(market.columns)!=['ts','smp']:raise GateError('Gate 1: SMP schema must be ts,smp.')
    market['ts']=time_index(market.ts)
    if market.ts.duplicated().any():raise GateError('Gate 1: duplicate SMP timestamp.')
    j=market.set_index('ts').reindex(t)
    value=j.smp.to_numpy(float)
    if not np.isfinite(value).all():raise GateError('Gate 1: incomplete/invalid SMP.')
    if peak_hours is None:raise GateError('Official peak window is not verified.')
    if (value<0).any():raise GateError('Negative SMP optimization policy requires separate rule review; convex nonnegative-weight assumption rejected.')
    return .1*value*np.where(t.hour.isin(peak_hours),3.,1.)


def aggregate(values,x):
    a=np.asarray(values,float);x=np.asarray(x,bool)
    if a.ndim!=2 or a.shape[1]!=len(x):raise GateError('Invalid plant aggregation shape.')
    return np.column_stack([a[:,x].sum(axis=1),a[:,~x].sum(axis=1)])


def penalty(pred,actual,weight):
    p=np.asarray(pred,float);a=np.asarray(actual,float);w=np.asarray(weight,float)
    if p.shape!=a.shape or p.ndim!=2 or p.shape[1]!=2 or w.shape!=(len(p),):raise GateError('Scoring shapes mismatch.')
    if not all(np.isfinite(v).all() for v in [p,a,w]) or (w<0).any():raise GateError('Invalid scoring values.')
    return w[:,None]*np.abs(p-a)


def score(pred,actual,weight,smp):
    terms=penalty(pred,actual,weight);err=np.asarray(pred)-actual
    revenue=float(np.sum(np.asarray(smp)[:,None]*actual));W=float(terms.sum())
    return {'W':W,'R':revenue,'S':revenue-W,'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.mean(err**2))),
            'bias':float(err.mean()),'G1_W':float(terms[:,0].sum()),'G2_W':float(terms[:,1].sum())}
