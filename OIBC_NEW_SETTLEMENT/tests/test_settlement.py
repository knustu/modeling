import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.contracts import GateError,assignment,composition,chronological,partition,digest,input_manifest,require_plant_lineage
from src.objective import aggregate,penalty,score,price_weights
from src.grouping_optimizer import search
from src.corrections import Correction,weighted_median
from src.ensemble import combine,fit_weights
from src.robustness import paired_bootstrap
from src.submission import build_synthetic,validate
from src.synthetic import fixture


class SettlementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads(Path('config/optimization_config.json').read_text(encoding='utf-8'))
        cls.cfg.update(expected_sites=4,iterations=4,swap_trials=10,bootstrap_replicates=30)
        cls.sites,cls.comps,cls.parts=fixture(cls.cfg)
        cls.x=np.array([True,True,False,False]);cls.c=cls.sites.capacity_kw.to_numpy()
    def test_formula(self):
        ts=pd.DatetimeIndex(['2025-01-01T11:00:00+09:00','2025-01-01T16:00:00+09:00'])
        w=price_weights(ts,pd.DataFrame({'ts':ts,'smp':[100.,50.]}),[11,12,13,14,15])
        a=np.array([[10.,0.],[20.,0.]]);p=np.array([[12.,0.],[17.,0.]])
        s=score(p,a,w,[100.,50.]);self.assertEqual(s['W'],75.);self.assertEqual(s['R'],2000.);self.assertEqual(s['S'],1925.)
    def test_cancellation(self):
        e=np.array([[10.,-10.,3.,-3.]])
        np.testing.assert_array_equal(aggregate(e,self.x),[[0.,0.]])
    def test_revenue_invariance(self):
        a=self.parts['development']['actual'];b=np.array([True,False,True,False])
        np.testing.assert_allclose(aggregate(a,self.x).sum(axis=1),aggregate(a,b).sum(axis=1))
    def test_feasibility(self):
        self.assertGreaterEqual(min(assignment(self.x,self.c,self.cfg)),20000)
        with self.assertRaises(GateError):assignment(np.ones(4,dtype=bool),self.c,self.cfg)
    def test_invalid_capacity(self):
        with self.assertRaises(GateError):assignment(self.x,np.array([1.,1.,1.,1.]),self.cfg)
    def test_composition(self):
        for c in self.comps.values():self.assertEqual(len(composition(c,self.sites,self.cfg)),4)
        c=next(iter(self.comps.values())).copy();c.loc[0,'site_id']='bad'
        with self.assertRaises(GateError):composition(c,self.sites,self.cfg)
    def test_optimizer_reproducible_best(self):
        d=self.parts['development'];args=(d['predictions'][0]-d['actual'],d['weight'],self.c,self.x,self.cfg,42)
        a,r=search(*args,anneal=True);b,s=search(*args,anneal=True)
        np.testing.assert_array_equal(a,b);self.assertEqual(r['best_development_W'],s['best_development_W'])
        self.assertLessEqual(r['best_development_W'],r['initial_W']);assignment(a,self.c,self.cfg)
    def test_swap_replay(self):
        d=self.parts['development'];x,r=search(d['predictions'][0]-d['actual'],d['weight'],self.c,self.x,self.cfg,1,mode='swap')
        replay=self.x.copy()
        for move in r['moves']:
            self.assertEqual(len(move['indices']),2);replay[move['indices']]=~replay[move['indices']];assignment(replay,self.c,self.cfg)
        assignment(x,self.c,self.cfg)
    def test_relocation_infeasible_rejected(self):
        for i in range(4):
            x=self.x.copy();x[i]=~x[i]
            with self.assertRaises(GateError):assignment(x,self.c,self.cfg)
    def test_additive(self):
        ts=self.parts['development']['ts'];a=np.full((len(ts),2),10.);p=a+3
        c=Correction('additive',self.cfg).fit(p,a,np.ones(len(ts)),ts,'development')
        np.testing.assert_allclose(c.apply(p,ts),a)
    def test_nonnegative(self):
        ts=self.parts['development']['ts'];p=np.full((len(ts),2),10.);a=np.zeros_like(p)
        c=Correction('additive',self.cfg).fit(p,a,np.ones(len(ts)),ts,'development')
        self.assertTrue((c.apply(np.zeros_like(p),ts)>=0).all())
    def test_conditional(self):
        ts=self.parts['development']['ts'];cfg=dict(self.cfg,minimum_cell_hours=2)
        a=np.full((len(ts),2),20.);p=a+np.asarray(ts.hour)[:,None]
        c=Correction('hour',cfg).fit(p,a,np.ones(len(ts)),ts,'development')
        np.testing.assert_allclose(c.apply(p,ts),a)
    def test_multiplicative(self):
        ts=self.parts['development']['ts'];a=np.full((len(ts),2),11.);p=np.full_like(a,10.)
        c=Correction('multiplicative',self.cfg).fit(p,a,np.ones(len(ts)),ts,'development')
        np.testing.assert_allclose(c.apply(p,ts),a)
    def test_correction_holdout(self):
        d=self.parts['holdout'];p=aggregate(d['actual'],self.x)
        with self.assertRaises(GateError):Correction('additive',self.cfg).fit(p,p,d['weight'],d['ts'],'holdout')
    def test_simplex(self):
        p=self.parts['development']['predictions'];np.testing.assert_allclose(combine(p,[.5,.5]),p.mean(axis=0))
        for w in [[-1.,2.],[.5,.6],[1.]]:
            with self.assertRaises(GateError):combine(p,w)
    def test_ensemble_holdout(self):
        d=self.parts['holdout']
        with self.assertRaises(GateError):fit_weights(d['predictions'],d['actual'],d['weight'],self.x,self.cfg,'holdout')
    def test_direct_reuse(self):
        with self.assertRaises(GateError):require_plant_lineage('direct_group')
    def test_chronology(self):chronological(self.parts['development'],self.parts['validation'],self.parts['holdout'],self.cfg)
    def test_role_leakage(self):
        with self.assertRaises(GateError):partition(self.parts['holdout'],'development',self.cfg,True)
    def test_training_leakage(self):
        d=copy.deepcopy(self.parts['development']);d['trained_through']=d['decision_at'][0]
        with self.assertRaises(GateError):partition(d,'development',self.cfg,True)
    def test_missing_prediction(self):
        d=copy.deepcopy(self.parts['development']);d['predictions'][0,0,0]=np.nan
        with self.assertRaises(GateError):partition(d,'development',self.cfg,True)
    def test_duplicate_timestamp(self):
        d=copy.deepcopy(self.parts['development']);ts=list(d['ts']);ts[1]=ts[0];d['ts']=ts
        with self.assertRaises(GateError):partition(d,'development',self.cfg,True)
    def test_prices_missing(self):
        ts=self.parts['development']['ts']
        with self.assertRaises(GateError):price_weights(ts,pd.DataFrame({'ts':ts[:1],'smp':[100.]}),[11])
    def test_price_unknown_peak(self):
        ts=self.parts['development']['ts']
        with self.assertRaises(GateError):price_weights(ts,pd.DataFrame({'ts':ts,'smp':100.}),None)
    def test_mae_not_weighted_guarantee(self):
        a=np.zeros((2,2));reference=np.array([[1.,0.],[10.,0.]]);candidate=np.array([[2.,0.],[0.,0.]])
        self.assertLess(np.abs(candidate).mean(),np.abs(reference).mean())
        self.assertGreater(penalty(candidate,a,[100.,1.]).sum(),penalty(reference,a,[100.,1.]).sum())
    def test_paired_bootstrap(self):
        ts=self.parts['development']['ts'];a=np.ones(len(ts));b=np.ones(len(ts))*2
        r=paired_bootstrap(a,b,ts,1,30,7);s=paired_bootstrap(a,b,ts,1,30,7)
        self.assertEqual(r,s);self.assertEqual(r['observed_delta_W'],-len(ts));self.assertLess(r['ci_high'],0)
    def test_versions(self):
        self.assertEqual(digest({'a':1}),digest({'a':1}));self.assertNotEqual(digest({'a':1}),digest({'a':2}))
    def test_missing_upstream(self):
        with self.assertRaises(GateError):input_manifest({'p':'nonexistent_phase4_fixture_file'}, {})
    def test_submission_and_safe_writing(self):
        from src.submission import grid
        with tempfile.TemporaryDirectory() as tmp:
            comp=next(iter(self.comps.values()));out=Path(tmp)/'new'
            build_synthetic(comp,self.sites,self.cfg,out)
            f=pd.read_csv(out/'forecast.csv',dtype={'ts':str})
            self.assertEqual(len(f),17520);self.assertFalse(validate(comp,f,self.sites,self.cfg)['submittable'])
            self.assertEqual(grid()[-1],pd.Timestamp('2026-06-01T00:00:00+09:00'))
            with self.assertRaises(FileExistsError):build_synthetic(comp,self.sites,self.cfg,out)
            for change in ['missing','duplicate','negative','timezone']:
                bad=f.copy()
                if change=='missing':bad=bad.iloc[1:]
                if change=='duplicate':bad.loc[1,'ts']=bad.loc[0,'ts']
                if change=='negative':bad.loc[0,'forecast_kwh']=-1.
                if change=='timezone':bad['ts']=[pd.Timestamp(v).tz_convert('UTC').isoformat() for v in bad.ts]
                with self.assertRaises(GateError):validate(comp,bad,self.sites,self.cfg)
    def test_weighted_median(self):self.assertEqual(weighted_median([-2.,5.],[10.,1.]),-2.)


if __name__=='__main__':unittest.main()
