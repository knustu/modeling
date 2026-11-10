import time
import numpy as np
from .contracts import assignment,GateError,require_plant_lineage


def search(errors,weight,capacity,initial,cfg,seed,anneal=False,mode='mixed',lineage='plant_level'):
    require_plant_lineage(lineage)
    e=np.asarray(errors,float);w=np.asarray(weight,float);c=np.asarray(capacity,float)
    if e.ndim!=2 or e.shape[1]!=len(c) or w.shape!=(len(e),) or not np.isfinite(e).all() or not np.isfinite(w).all() or (w<0).any():raise GateError('Invalid development residuals/weights.')
    x=np.asarray(initial).copy();assignment(x,c,cfg);best=x.copy()
    total=e.sum(axis=1);group=e[:,x].sum(axis=1)
    def loss(v):return float(np.dot(w,np.abs(v)+np.abs(total-v)))
    current=loss(group);best_value=current;rng=np.random.default_rng(seed);trace=[];start=time.perf_counter()
    for iteration in range(cfg['iterations']):
        moves=[]
        if mode in ['mixed','relocation']:moves.extend((i,) for i in range(len(c)))
        if mode in ['mixed','swap']:
            left=np.flatnonzero(x);right=np.flatnonzero(~x)
            moves.extend((int(rng.choice(left)),int(rng.choice(right))) for _ in range(cfg['swap_trials']))
        if anneal:rng.shuffle(moves)
        accepted=None;candidate_value=current
        for indices in moves:
            proposal=x.copy();proposal[list(indices)]=~proposal[list(indices)]
            try:caps=assignment(proposal,c,cfg)
            except GateError:continue
            signs=np.where(x[list(indices)],-1.,1.)
            updated=group+(e[:,list(indices)]*signs).sum(axis=1)
            value=loss(updated);delta=value-current
            temperature=cfg['annealing_temperature']*(1-iteration/max(1,cfg['iterations']))
            take=delta < -1e-10 or (anneal and temperature>0 and rng.random()<np.exp(-max(0.,delta)/temperature))
            if take and (anneal or value<candidate_value):
                accepted=(indices,proposal,updated,value,caps);candidate_value=value
                if anneal:break
        if accepted is None:
            if anneal:continue
            break
        indices,x,group,value,caps=accepted
        trace.append({'iteration':iteration,'indices':list(indices),'previous_W':current,'W':value,'delta_W':value-current,
                      'G1_capacity_kw':caps[0],'G2_capacity_kw':caps[1],'algorithm':'annealing' if anneal else mode})
        current=value
        if value<best_value:best_value=value;best=x.copy()
    assignment(best,c,cfg)
    return best,{'best_development_W':best_value,'initial_W':loss(e[:,np.asarray(initial)].sum(axis=1)),
                 'runtime_seconds':time.perf_counter()-start,'moves':trace,'seed':seed,'best_preserved':True}


def multistart(errors,weight,capacity,starts,cfg):
    results=[];winner=None
    for i,initial in enumerate(starts):
        for anneal in [False,True]:
            x,report=search(errors,weight,capacity,initial,cfg,cfg['seed']+i,anneal)
            results.append(report)
            if winner is None or report['best_development_W']<winner[1]['best_development_W']:winner=(x,report)
    return winner[0],results
