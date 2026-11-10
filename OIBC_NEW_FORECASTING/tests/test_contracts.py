import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.core import (ContractError,metadata,grouping,feature_frame,label_rows,target_rows,
    Forecaster,Preprocessor,aggregate,penalty,expanding_folds,official_grid,validate_forecast)
from src.synthetic import fixture


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads(Path('config/forecasting_config.json').read_text())
        cls.sites,cls.targets,cls.labels,cls.weather,cls.compositions=fixture(cls.cfg,days=8)

    def test_metadata_groupings(self):
        s=metadata(self.sites,self.cfg)
        for c in self.compositions.values(): self.assertEqual(len(grouping(c,s,self.cfg)),253)
        bad=self.compositions['synthetic_balanced'].copy();bad.loc[0,'site_id']=bad.loc[1,'site_id']
        with self.assertRaises(ContractError): grouping(bad,s,self.cfg)

    def test_join_and_provider_units(self):
        t,X=feature_frame(self.targets,self.sites,self.weather,self.cfg)
        self.assertEqual(len(X),len(t));self.assertTrue(np.isfinite(X).all().all())
        self.assertTrue(X['wb__temperature'].between(6.9,23.1).all())
        cfg=copy.deepcopy(self.cfg);cfg['providers']['wa']['fields']['radiation']['scale']=.001
        _,Y=feature_frame(self.targets,self.sites,self.weather,cfg)
        np.testing.assert_allclose(X['wa__radiation']*.001,Y['wa__radiation'])

    def test_issuance_availability(self):
        _,X=feature_frame(self.targets,self.sites,self.weather,self.cfg)
        w={k:v.copy() for k,v in self.weather.items()}
        future=w['wa'].copy();future['issued_at']+=pd.Timedelta(hours=8);future['available_at']+=pd.Timedelta(hours=8);future['radiation']=1e10
        w['wa']=pd.concat([w['wa'],future],ignore_index=True)
        _,Y=feature_frame(self.targets,self.sites,w,self.cfg)
        pd.testing.assert_frame_equal(X,Y)
        w['wa']=future
        _,Z=feature_frame(self.targets,self.sites,w,self.cfg)
        self.assertTrue(Z['wa__radiation'].isna().all())

    def test_missing_weather(self):
        _,X=feature_frame(self.targets,self.sites,{},self.cfg)
        self.assertTrue(X['wa__radiation'].isna().all())
        a=Preprocessor().fit(X).transform(X)
        self.assertTrue(np.isfinite(a).all())

    def test_duplicates(self):
        bad=pd.concat([self.targets,self.targets.iloc[:1]])
        with self.assertRaises(ContractError):target_rows(bad,self.sites)
        w={k:v.copy() for k,v in self.weather.items()};w['wa']=pd.concat([w['wa'],w['wa'].iloc[:1]])
        with self.assertRaises(ContractError):feature_frame(self.targets,self.sites,w,self.cfg)
        labels=pd.concat([self.labels,self.labels.iloc[:1]])
        with self.assertRaises(ContractError):label_rows(labels,self.sites,self.cfg)

    def test_cutoff_and_availability(self):
        labels=self.labels.copy();labels.loc[0,'ts']=pd.Timestamp('2025-06-01T01:00:00+09:00')
        labels.loc[0,'label_available_at']=pd.Timestamp('2025-06-01T02:00:00+09:00')
        with self.assertRaises(ContractError):label_rows(labels,self.sites,self.cfg)
        labels=self.labels.copy();labels.loc[0,'label_available_at']=labels.loc[0,'ts']-pd.Timedelta(hours=1)
        with self.assertRaises(ContractError):label_rows(labels,self.sites,self.cfg)

    def test_time_and_decision_guards(self):
        bad=self.targets.copy();bad['ts']=bad.ts.dt.tz_localize(None)
        with self.assertRaises(ContractError):target_rows(bad,self.sites)
        bad=self.targets.copy();bad['decision_at']=bad.ts
        with self.assertRaises(ContractError):target_rows(bad,self.sites)

    def test_fold_isolation(self):
        f=self.targets.merge(self.labels,on=['site_id','ts'])
        f.loc[f.ts<pd.Timestamp('2025-01-02T00:00:00+09:00'),'label_available_at']=pd.Timestamp('2025-01-07T00:00:00+09:00')
        origin,train,valid=next(expanding_folds(f,[('2025-01-04T00:00:00+09:00','2025-01-07T00:00:00+09:00')]))
        self.assertFalse((train&valid).any())
        self.assertTrue((f.loc[train,'label_available_at']<=origin).all())
        self.assertTrue((f.loc[valid,'decision_at']>=origin).all())

    def test_preprocessing_training_only(self):
        X=pd.DataFrame({'a':[1.,3.,np.nan]});pre=Preprocessor().fit(X)
        median=pre.median.copy();mean=pre.mean.copy()
        pre.transform(pd.DataFrame({'a':[1e12,np.nan]}))
        np.testing.assert_array_equal(median,pre.median);np.testing.assert_array_equal(mean,pre.mean)
        self.assertEqual(pre.median[0],2.)

    def test_models_reproducible_shapes_and_normalization(self):
        t,X=feature_frame(self.targets,self.sites,self.weather,self.cfg)
        cap=np.full(len(t),self.sites.capacity_kw.iloc[0]);y=self.labels.energy_kwh.to_numpy()
        train=np.arange(len(t))%3!=0; valid=~train
        for normalized in [True,False]:
            cfg=dict(self.cfg,target_normalization=normalized)
            for kind in ['climatology','ridge','hist_gradient_boosting']:
                a=Forecaster(kind,cfg).fit(t.loc[train],X.loc[train],y[train],cap[train])
                b=Forecaster(kind,cfg).fit(t.loc[train],X.loc[train],y[train],cap[train])
                p=a.predict(t.loc[valid],X.loc[valid],cap[valid]);q=b.predict(t.loc[valid],X.loc[valid],cap[valid])
                self.assertEqual(p.shape,(valid.sum(),));self.assertTrue(np.isfinite(p).all());self.assertTrue((p>=0).all())
                np.testing.assert_allclose(p,q,rtol=1e-12,atol=1e-12)
        cfg=dict(self.cfg,target_normalization=True)
        m=Forecaster('climatology',cfg).fit(t,X,y,cap)
        a=m.predict(t,X,cap);b=m.predict(t,X,2*cap)
        np.testing.assert_allclose(b,2*a)

    def test_aggregation_switching(self):
        p=self.targets[['site_id','ts']].copy()
        p['forecast_kwh']=p.site_id.str[-3:].astype(int).astype(float)
        totals=[]
        for comp in self.compositions.values():
            g=aggregate(p,comp,self.sites,self.cfg)
            self.assertEqual(len(g),2*p.ts.nunique())
            self.assertAlmostEqual(g.forecast_kwh.sum(),p.forecast_kwh.sum())
            totals.append(g.loc[g.group.eq('G1'),'forecast_kwh'].iloc[0])
        self.assertNotEqual(totals[0],totals[1])
        with self.assertRaises(ContractError):aggregate(p.iloc[1:],next(iter(self.compositions.values())),self.sites,self.cfg)

    def test_penalty_independent_example(self):
        ts=[pd.Timestamp('2025-01-01T11:00:00+09:00'),pd.Timestamp('2025-01-01T16:00:00+09:00')]
        a=pd.DataFrame({'group':['G1','G1'],'ts':ts,'energy_kwh':[10.,20.]})
        p=pd.DataFrame({'group':['G1','G1'],'ts':ts,'forecast_kwh':[12.,17.]})
        market=pd.DataFrame({'ts':ts,'smp':[100.,50.]})
        self.assertAlmostEqual(penalty(a,p,market,[11,12,13,14,15]),75.)
        with self.assertRaises(ContractError):penalty(a,p,market,None)
        with self.assertRaises(ContractError):penalty(a,p.iloc[:1],market,[11])

    def test_official_format(self):
        grid=official_grid();self.assertEqual(grid[-1],pd.Timestamp('2026-06-01T00:00:00+09:00'))
        f=pd.concat([pd.DataFrame({'group':g,'ts':[v.isoformat() for v in grid],'forecast_kwh':0.}) for g in ['G1','G2']],ignore_index=True)
        self.assertTrue(validate_forecast(f)['valid_summary_contract'])
        for value in [-1.,float('nan'),float('inf')]:
            bad=f.copy();bad.loc[0,'forecast_kwh']=value
            with self.assertRaises(ContractError):validate_forecast(bad)
        with self.assertRaises(ContractError):validate_forecast(f.iloc[1:])

    def test_offline_source(self):
        from src.visualizer import render
        f=pd.DataFrame({'model':['ridge'],'candidate':['synthetic'],'split':['holdout'],'group':['G1'],
                        'ts':[pd.Timestamp('2025-01-01T12:00:00+09:00')],'energy_kwh':[1.],'forecast_kwh':[2.]})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'view.html';render(path,f,pd.DataFrame())
            text=path.read_text()
            for forbidden in ['<script src=','fetch(','XMLHttpRequest','https://']:
                self.assertNotIn(forbidden,text)
            self.assertIn('image/png',text)


if __name__=='__main__':unittest.main()
