import numpy as np
from .contracts import GateError


def weighted_median(values,weights):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:raise GateError('Invalid weighted median inputs.')
    order=np.argsort(values,kind='stable');v=values[order];w=weights[order]
    return float(v[np.searchsorted(np.cumsum(w),w.sum()/2,side='left')])


class Correction:
    def __init__(self,kind,cfg):self.kind=kind;self.cfg=cfg
    def fit(self,pred,actual,weight,ts,role,weather=None):
        if role!='development':raise GateError('Gate 5/6: correction fit accepts development only.')
        p=np.asarray(pred,float);a=np.asarray(actual,float);w=np.asarray(weight,float)
        if p.shape!=a.shape or p.shape!=(len(ts),2) or not np.isfinite(p).all() or not np.isfinite(a).all():raise GateError('Invalid correction matrices.')
        self.period=[str(ts[0]),str(ts[-1])];self.cells={};self.alpha=[1.,1.];self.offset=[0.,0.]
        if self.kind=='none':return self
        if self.kind not in ['additive','multiplicative','hour','weather']:raise GateError('Unknown correction.')
        for g in range(2):
            bound=self.cfg['maximum_additive_kwh']
            self.offset[g]=float(np.clip(weighted_median(a[:,g]-p[:,g],w),-bound,bound))
            if self.kind=='multiplicative':
                self.alpha[g]=min(self.cfg['alpha_grid'],key=lambda alpha:np.dot(w,np.abs(np.maximum(0,alpha*p[:,g])-a[:,g])))
            if self.kind in ['hour','weather']:
                bins=self.bin_ids(ts,weather)
                for cell in sorted(set(bins)):
                    mask=bins==cell
                    if mask.sum()>=self.cfg['minimum_cell_hours'] and w[mask].sum()>0:
                        self.cells[f'{g}:{cell}']=float(np.clip(weighted_median(a[mask,g]-p[mask,g],w[mask]),-bound,bound))
        return self
    def bin_ids(self,ts,weather):
        if self.kind=='hour':return np.asarray(ts.hour,int)
        if weather is None:raise GateError('Issuance-valid weather feature required.')
        weather=np.asarray(weather,float)
        if weather.shape!=(len(ts),) or not np.isfinite(weather).all():raise GateError('Invalid weather feature.')
        return np.digitize(weather,[200.,500.])
    def apply(self,pred,ts,weather=None):
        p=np.asarray(pred,float).copy()
        if self.kind=='none':return np.maximum(0,p)
        for g in range(2):
            if self.kind=='multiplicative':p[:,g]*=self.alpha[g]
            elif self.kind=='additive':p[:,g]+=self.offset[g]
            else:
                bins=self.bin_ids(ts,weather);p[:,g]+=np.array([self.cells.get(f'{g}:{cell}',self.offset[g]) for cell in bins])
        return np.maximum(0,p)
    def config(self):return {'kind':self.kind,'period':self.period,'offset':self.offset,'alpha':self.alpha,'cells':self.cells,'status':'unverified'}
