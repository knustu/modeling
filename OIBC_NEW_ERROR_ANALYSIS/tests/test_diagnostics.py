import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.contracts import (DiagnosticError,validate_plant,composition_contract,weather_contract,
                           verify_group_forecast,phase2_residual_adapter)
from src.analysis import summary,portfolios,interactions,weighted,temporal,weather_summaries
from src.statistics import paired_blocks
from src.synthetic import fixture


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads(Path('config/analysis_config.json').read_text())
        cls.cfg.update(expected_sites=4,minimum_group_capacity_kw=20000.,minimum_pair_observations=3)
        cls.sites,cls.raw,cls.comps,cls.grid,cls.weather=fixture(cls.cfg,days=8)
        cls.clean=validate_plant(cls.raw,cls.sites,cls.cfg,cls.grid,True)

    def test_perfect(self):
        s=summary([0,0,0],100);self.assertEqual(s['mae'],0);self.assertEqual(s['rmse'],0)
    def test_positive_bias(self):
        s=summary([3,3,3],100);self.assertEqual(s['me'],3);self.assertEqual(s['positive_frequency'],1)
    def test_negative_bias(self):
        s=summary([-3,-3],100);self.assertEqual(s['negative_frequency'],1);self.assertEqual(s['me'],-3)
    def test_normalization(self):
        self.assertAlmostEqual(summary([10,-10],100)['normalized_mae'],.1)
        with self.assertRaises(DiagnosticError):summary([1],0)
    def test_zero_variance(self):
        self.assertIsNone(summary([2,2],100)['skewness'])
        f=self.clean.copy();f['error_kwh']=1.;f['forecast_kwh']=f.actual_kwh+1
        _,a=portfolios(f,next(iter(self.comps.values())),'c')
        corr,c=interactions(a,self.cfg);self.assertTrue(corr.correlation.isna().all())
    def test_identical_sequences(self):
        x=np.arange(100.);self.assertAlmostEqual(np.corrcoef(x,x)[0,1],1)
    def test_opposing_cancellation(self):
        f=self.clean.copy();m=f.site_id.map(dict(zip(self.sites.site_id,[1.,-1.,1.,-1.])))
        f['error_kwh']=m;f['forecast_kwh']=f.actual_kwh+m
        comp=pd.DataFrame({'site_id':self.sites.site_id,'group':['G1','G1','G2','G2']})
        g,_=portfolios(f,comp,'opposing')
        self.assertTrue(g.error_kwh.eq(0).all());self.assertTrue(g.cancellation_kwh.eq(2).all())
        self.assertTrue((g.error_kwh.abs()<=g.individual_absolute_sum+1e-9).all())
    def test_independent_errors(self):
        r=np.random.default_rng(5);a=r.normal(size=10000);b=r.normal(size=10000)
        self.assertLess(abs(np.corrcoef(a,b)[0,1]),.05)
    def test_correlated_errors(self):
        r=np.random.default_rng(6);a=r.normal(size=10000);b=a+r.normal(scale=.1,size=10000)
        self.assertGreater(np.corrcoef(a,b)[0,1],.98)
    def test_heteroscedastic(self):
        r=np.random.default_rng(7);a=r.normal(size=10000);b=r.normal(scale=4,size=10000)
        self.assertGreater(summary(b,100)['std'],3*summary(a,100)['std'])
    def test_extreme(self):
        a=np.zeros(1000);a[-1]=1000
        self.assertGreater(summary(a,100)['excess_kurtosis'],100)
    def test_temporal_bias(self):
        g,_=portfolios(self.clean,next(iter(self.comps.values())),'c')
        g['error_kwh']=np.where(g.ts.dt.hour==12,20.,0.)
        out=temporal(g,self.cfg)
        self.assertTrue(out['hour'].loc[out['hour'].hour.eq(12),'mae'].eq(20).all())
    def test_seasonal_bias(self):
        a=summary(np.full(100,5.),100);b=summary(np.full(100,-5.),100)
        self.assertGreater(a['me'],b['me'])
    def test_weather_association(self):
        f=self.clean.copy();f['irradiance']=np.tile([0.,600.],len(f)//2)
        f['error_kwh']=np.where(f.irradiance>500,10.,1.)
        s=weather_summaries(f,self.cfg)
        self.assertGreater(s.mae.max(),s.mae.min())
    def test_missing_prediction(self):
        f=self.raw.copy();f.loc[0,'forecast_kwh']=np.nan
        v=validate_plant(f,self.sites,self.cfg,self.grid,True)
        g,_=portfolios(v,next(iter(self.comps.values())),'c')
        self.assertGreater(g.error_kwh.isna().sum(),0)
        self.assertGreater(summary(v.error_kwh,v.capacity_kw)['missing_count'],0)
    def test_missing_row(self):
        with self.assertRaises(DiagnosticError):validate_plant(self.raw.iloc[1:],self.sites,self.cfg,self.grid,True)
    def test_duplicates(self):
        with self.assertRaises(DiagnosticError):validate_plant(pd.concat([self.raw,self.raw.iloc[:1]]),self.sites,self.cfg,self.grid,True)
    def test_misaligned_composition(self):
        c=next(iter(self.comps.values())).copy();c.loc[0,'site_id']='UNKNOWN'
        with self.assertRaises(DiagnosticError):composition_contract(c,self.sites,self.cfg)
    def test_aggregation_verification(self):
        g,_=portfolios(self.clean,next(iter(self.comps.values())),'c');bad=g.copy();bad.loc[0,'forecast_kwh']+=1
        with self.assertRaises(DiagnosticError):verify_group_forecast(g,bad)
    def test_covariance_identity(self):
        g,a=portfolios(self.clean,next(iter(self.comps.values())),'c');cor,c=interactions(a,self.cfg)
        for _,v in c.groupby(['model_id','forecast_round','validation_fold','group']):
            self.assertAlmostEqual(v.covariance_contribution.sum(),v.aggregate_variance.iloc[0],places=8)
        for vals,v in c.groupby(['model_id','forecast_round','validation_fold','group']):
            part=g.loc[(g.model_id==vals[0])&(g.group==vals[-1])]
            self.assertAlmostEqual(v.absolute_error_attribution.sum(),part.error_kwh.abs().mean(),places=7)
    def test_penalty_reference(self):
        cfg=dict(self.cfg,official_rules_verified=True,peak_hours=[11,12,13,14,15])
        g=pd.DataFrame({'ts':pd.to_datetime(['2025-01-01T11:00:00+09:00','2025-01-01T16:00:00+09:00']),'error_kwh':[2.,-3.]})
        market=pd.DataFrame({'ts':g.ts,'smp':[100.,50.]})
        f,status=weighted(g,market,cfg);self.assertEqual(f.penalty.sum(),75.)
    def test_missing_smp(self):
        f,status=weighted(pd.DataFrame(),None,self.cfg);self.assertIsNone(f);self.assertEqual(status['status'],'unavailable')
    def test_timezone(self):
        f=self.raw.copy();f['ts']=f.ts.dt.tz_localize(None)
        with self.assertRaises(DiagnosticError):validate_plant(f,self.sites,self.cfg,self.grid,True)
    def test_issuance_leakage(self):
        w=self.weather.copy();w['available_at']=w.decision_at+pd.Timedelta(hours=1)
        with self.assertRaises(DiagnosticError):weather_contract(w,self.clean,['irradiance'])
    def test_training_leakage(self):
        f=self.raw.copy();f['trained_through']=f.decision_at
        with self.assertRaises(DiagnosticError):validate_plant(f,self.sites,self.cfg,self.grid,True)
    def test_in_sample_rejected(self):
        f=self.raw.copy();f['prediction_kind']='fitted'
        with self.assertRaises(DiagnosticError):validate_plant(f,self.sites,self.cfg,self.grid,True)
    def test_bootstrap_reproducibility(self):
        grid=pd.date_range('2025-01-01',periods=24*28,freq='h',tz='Asia/Seoul')
        a=pd.Series(2.,index=grid);b=pd.Series(1.,index=grid)
        x=paired_blocks(a,b,1,50,12);y=paired_blocks(a,b,1,50,12)
        self.assertEqual(x,y);self.assertEqual(x['ci_low'],1.);self.assertEqual(x['block_count'],28)
    def test_unpaired_rejected(self):
        grid=pd.date_range('2025-01-01',periods=48,freq='h',tz='Asia/Seoul');a=pd.Series(1.,index=grid)
        with self.assertRaises(DiagnosticError):paired_blocks(a,a.iloc[1:],1,10,1)
    def test_provenance_required(self):
        with self.assertRaises(DiagnosticError):phase2_residual_adapter(pd.DataFrame({'site_id':['a'],'ts':['x'],'forecast_kwh':[1],'energy_kwh':[1]}),'m','development',{})
    def test_offline_dashboard(self):
        from src.visualizer import render
        g,a=portfolios(self.clean,next(iter(self.comps.values())),'c');corr,c=interactions(a,self.cfg)
        from src.analysis import summarize
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'view.html';render(path,g,corr,self.clean,None,summarize(g,['model_id','forecast_round','validation_fold','grouping_id','group']),None,self.cfg)
            html=path.read_text()
            for bad in ['fetch(','XMLHttpRequest','<script src=','https://']:self.assertNotIn(bad,html)
            self.assertIn('image/png',html)


if __name__=='__main__':unittest.main()
