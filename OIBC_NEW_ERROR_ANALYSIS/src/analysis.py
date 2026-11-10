import numpy as np
import pandas as pd
from .contracts import KEY,DiagnosticError,numbers,timestamps


def summary(error,capacity):
    e=np.asarray(error,float);c=np.broadcast_to(np.asarray(capacity,float),e.shape)
    if not np.isfinite(c).all() or (c<=0).any():raise DiagnosticError('Normalization requires finite positive capacities.')
    valid=np.isfinite(e);v=e[valid];normalized=v/c[valid]
    out={'expected_count':len(e),'sample_count':len(v),'missing_count':int((~valid).sum()),
         'coverage':float(valid.mean()) if len(e) else None}
    if not len(v):return dict(out,me=None,mae=None,rmse=None,median_ae=None,std=None,
        q01=None,q05=None,q50=None,q95=None,q99=None,positive_frequency=None,negative_frequency=None,
        normalized_mae=None,normalized_rmse=None,skewness=None,excess_kurtosis=None)
    centered=v-v.mean();sd=v.std(ddof=0)
    return dict(out,me=float(v.mean()),mae=float(np.abs(v).mean()),rmse=float(np.sqrt(np.mean(v*v))),
        median_ae=float(np.median(np.abs(v))),std=float(sd),
        **{key:float(value) for key,value in zip(['q01','q05','q50','q95','q99'],np.quantile(v,[.01,.05,.5,.95,.99]))},
        positive_frequency=float((v>0).mean()),negative_frequency=float((v<0).mean()),
        normalized_mae=float(np.abs(normalized).mean()),normalized_rmse=float(np.sqrt(np.mean(normalized**2))),
        skewness=float(np.mean(centered**3)/sd**3) if sd>0 else None,
        excess_kurtosis=float(np.mean(centered**4)/sd**4-3) if sd>0 else None)


def summarize(frame,keys):
    rows=[]
    for vals,g in frame.groupby(keys,observed=True,dropna=False,sort=True):
        if not isinstance(vals,tuple):vals=(vals,)
        rows.append(dict(zip(keys,vals),**summary(g.error_kwh,g.capacity_kw)))
    return pd.DataFrame(rows)


def portfolios(plant,composition,grouping_id):
    f=plant.merge(composition,on='site_id',validate='many_to_one')
    rows=[]
    for vals,g in f.groupby(KEY+['group','ts'],sort=True):
        observed=g[['forecast_kwh','actual_kwh']].notna().all(axis=1)
        complete=bool(observed.all())
        forecast=float(g.forecast_kwh.sum()) if complete else np.nan
        actual=float(g.actual_kwh.sum()) if complete else np.nan
        signed=forecast-actual
        individual=float(g.error_kwh.abs().sum()) if complete else np.nan
        cancellation=individual-abs(signed)
        if complete and cancellation < -1e-7:raise DiagnosticError('Triangle inequality violated.')
        row=dict(zip(KEY+['group','ts'],vals),grouping_id=grouping_id,forecast_kwh=forecast,actual_kwh=actual,
            error_kwh=signed,individual_absolute_sum=individual,cancellation_kwh=max(0.,cancellation) if complete else np.nan,
            cancellation_fraction=(max(0.,cancellation)/individual if individual>0 else np.nan),
            capacity_kw=float(g.capacity_kw.sum()),plant_count=len(g),observed_plants=int(observed.sum()))
        rows.append(row)
    return pd.DataFrame(rows),f


def interactions(assigned,cfg):
    correlation=[];contribution=[]
    for vals,g in assigned.groupby(KEY+['group'],sort=True):
        if g.ts.nunique()*g.site_id.nunique()>cfg['maximum_pivot_cells']:
            raise DiagnosticError('Pivot budget exceeded; partition input.')
        wide=g.pivot(index='ts',columns='site_id',values='error_kwh').sort_index()
        # Common complete hours for covariance identities; no pairwise covariance mixed with aggregate variance.
        common=wide.dropna(how='any');count=len(common)
        if count>=cfg['minimum_pair_observations']:
            a=common.to_numpy(float);centered=a-a.mean(axis=0)
            covariance=centered.T@centered/count
            std=np.sqrt(np.diag(covariance));denom=std[:,None]*std[None,:]
            corr=np.divide(covariance,denom,out=np.full_like(covariance,np.nan),where=denom>1e-15)
            total=a.sum(axis=1);aggregate_variance=float(total.var(ddof=0))
            for i,site in enumerate(common.columns):
                contribution.append(dict(zip(KEY+['group'],vals),site_id=site,sample_count=count,
                    expected_count=len(wide),coverage=count/len(wide),signed_mean_contribution=float(a[:,i].mean()),
                    covariance_contribution=float(covariance[i].sum()),aggregate_variance=aggregate_variance,
                    variance_share=float(covariance[i].sum()/aggregate_variance) if aggregate_variance>1e-12 else None,
                    absolute_error_attribution=float(np.mean(a[:,i]*np.sign(total)))))
                for j,other in enumerate(common.columns):
                    correlation.append(dict(zip(KEY+['group'],vals),site_i=site,site_j=other,correlation=corr[i,j],
                        covariance=float(covariance[i,j]),sample_count=count,expected_count=len(wide),coverage=count/len(wide)))
        else:
            for site in wide.columns:
                contribution.append(dict(zip(KEY+['group'],vals),site_id=site,sample_count=count,expected_count=len(wide),
                    coverage=count/len(wide),signed_mean_contribution=None,covariance_contribution=None,
                    aggregate_variance=None,variance_share=None,absolute_error_attribution=None))
    return pd.DataFrame(correlation),pd.DataFrame(contribution)


def temporal(group,cfg):
    f=group.copy();f['hour']=f.ts.dt.hour;f['month']=f.ts.dt.month
    f['day_of_year']=f.ts.dt.dayofyear
    f['season']=f.month.map({12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'})
    f['daytime']=f.hour.isin(cfg['daytime_hours'])
    base=KEY+['grouping_id','group']
    output={name:summarize(f,base+[name]) for name in ['hour','month','season','day_of_year','daytime']}
    acf=[];episodes=[]
    for vals,g in f.groupby(base,sort=True):
        g=g.sort_values('ts');series=g.set_index('ts').error_kwh
        for lag in cfg['acf_lags_hours']:
            shifted=series.copy();shifted.index=shifted.index+pd.Timedelta(hours=lag)
            pair=pd.concat([series.rename('now'),shifted.rename('lagged')],axis=1).dropna()
            rho=pair['now'].corr(pair.lagged) if len(pair)>=3 and pair['now'].std()>0 and pair.lagged.std()>0 else None
            acf.append(dict(zip(base,vals),lag_hours=lag,correlation=rho,sample_count=len(pair),expected_count=len(series)))
        previous=None;sign=0;length=0;start=None
        def emit():
            if length:episodes.append(dict(zip(base,vals),start=start,end=previous,sign=sign,length_hours=length))
        for row in g.itertuples():
            current=0 if pd.isna(row.error_kwh) else int(np.sign(row.error_kwh))
            contiguous=previous is not None and row.ts-previous==pd.Timedelta(hours=1)
            if current and current==sign and contiguous:length+=1
            else:
                emit();sign=current;length=1 if current else 0;start=row.ts
            previous=row.ts
        emit()
    output['acf']=pd.DataFrame(acf);output['episodes']=pd.DataFrame(episodes)
    f['date']=f.ts.dt.date.astype(str);output['daily']=summarize(f,base+['date'])
    return output


def weather_summaries(joined,cfg):
    rows=[]
    for field,edges in cfg['weather_bins'].items():
        if field not in joined:continue
        f=joined.copy();f['condition_bin']=pd.cut(f[field],edges,right=False,include_lowest=True).astype('string').fillna('missing_or_outside_bins')
        s=summarize(f,KEY+['condition_bin']);s['weather_variable']=field
        rows.append(s)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def weighted(group,smp,cfg):
    if smp is None:return None,{'status':'unavailable','reason':'No SMP supplied; no prices fabricated.'}
    if not cfg['official_rules_verified'] or cfg['peak_hours'] is None:
        return None,{'status':'unavailable','reason':'Peak timestamp convention not verified.'}
    require_cols={'ts','smp'}
    if not require_cols<=set(smp):raise DiagnosticError('Invalid SMP columns.')
    market=smp.copy();market['ts']=timestamps(market.ts,True);market['smp']=numbers(market.smp)
    if market.ts.duplicated().any():raise DiagnosticError('Duplicate SMP timestamp.')
    f=group.merge(market,on='ts',how='left',validate='many_to_one')
    if f.smp.isna().any():return None,{'status':'unavailable','reason':'Incomplete SMP coverage.'}
    f['multiplier']=np.where(f.ts.dt.hour.isin(cfg['peak_hours']),3.,1.)
    f['penalty']=.1*f.smp*f.multiplier*f.error_kwh.abs()
    return f,{'status':'available','complete_error_hours':int(f.penalty.notna().sum()),'expected_group_hours':len(f),
              'negative_smp':bool((f.smp<0).any()),'interpretation':'Signed price-weighted contribution if SMP can be negative.'}


def penalty_tables(weighted_frame):
    base=KEY+['grouping_id','group'];f=weighted_frame.copy()
    out={}
    for key,val in [('hour',f.ts.dt.hour),('month',f.ts.dt.month)]:
        f[key]=val
        grouped=f.groupby(base+[key]).penalty
        table=grouped.agg(['count','size']).reset_index().rename(columns={'count':'sample_count','size':'expected_count'})
        table['penalty']=grouped.sum(min_count=1).to_numpy()
        table['coverage']=table.sample_count/table.expected_count
        out[key]=table
    rows=[]
    for vals,g in f.groupby(base):
        v=g.penalty.dropna().sort_values(ascending=False).to_numpy();total=float(v.sum())
        for fraction in [.01,.05,.1,.2]:
            k=max(1,int(np.ceil(len(v)*fraction))) if len(v) else 0
            rows.append(dict(zip(base,vals),top_fraction=fraction,sample_count=len(v),expected_count=len(g),
                total_penalty=total,selected_hours=k,concentration=float(v[:k].sum()/total) if len(v) and total>0 and (v>=0).all() else None))
    out['concentration']=pd.DataFrame(rows)
    return out
