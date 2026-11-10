import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.grouping import (InputError,features,feasible,history_frame,initialize,
                          optimize,read_composition,run,sites_frame,validate_assignment)
from tests.synthetic_data import make_synthetic


class GroupingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads(Path('config/grouping_config.json').read_text(encoding='utf-8'))
        cls.cfg.update(timestamp_convention='interval_start',beam_width=200,
                       max_iterations=2,swap_trials=30,random_starts=1)
        cls.sites,cls.raw=make_synthetic(days=8)

    def prepared(self,raw=None):
        s=sites_frame(self.sites,self.cfg)
        p,_=history_frame(self.raw if raw is None else raw,s,self.cfg)
        z,_=features(p,s,self.cfg)
        return s,p,z

    def test_assignments(self):
        c=self.sites.capacity_kw.to_numpy()
        x=initialize(c,self.cfg)
        check=validate_assignment(x,self.sites,self.cfg)
        self.assertEqual(check['site_count'],253)
        self.assertEqual(check['g1_count']+check['g2_count'],253)
        self.assertTrue(check['capacity_preserved'])
        self.assertGreaterEqual(check['g1_capacity_kw'],20000)
        self.assertGreaterEqual(check['g2_capacity_kw'],20000)

    def test_duplicate_ids(self):
        s=self.sites.copy()
        s.loc[1,'site_id']=s.loc[0,'site_id']
        with self.assertRaises(InputError):sites_frame(s,self.cfg)

    def test_invalid_capacity(self):
        for value in [None,0,-1,float('inf'),'invalid']:
            s=self.sites.copy()
            s['capacity_kw']=s.capacity_kw.astype(object)
            s.loc[0,'capacity_kw']=value
            with self.assertRaises(InputError):sites_frame(s,self.cfg)

    def test_infeasible(self):
        s=self.sites.copy()
        s['capacity_kw']=1.
        with self.assertRaisesRegex(InputError,'불가능'):sites_frame(s,self.cfg)

    def test_reproducible(self):
        s,p,z=self.prepared()
        c=s.capacity_kw.to_numpy()
        a=initialize(c,self.cfg,17)
        b=initialize(c,self.cfg,17)
        np.testing.assert_array_equal(a,b)
        x,l,_=optimize(a,c,z,self.cfg,19)
        y,k,_=optimize(b,c,z,self.cfg,19)
        np.testing.assert_array_equal(x,y)
        self.assertEqual(l,k)

    def test_accepted_moves(self):
        s,p,z=self.prepared()
        c=s.capacity_kw.to_numpy()
        initial=initialize(c,self.cfg)
        final,log,_=optimize(initial,c,z,self.cfg,42)
        replay=initial.copy()
        for entry in log[1:]:
            idx=entry['affected_indices']
            replay[idx]=~replay[idx]
            self.assertTrue(feasible(replay,c,20000))
            self.assertGreater(entry['improvement'],0)
        np.testing.assert_array_equal(replay,final)

    def test_missing_and_zero(self):
        raw=self.raw.drop(self.raw.index[::23]).copy()
        raw.loc[raw.site_id.eq('SYNTH_000'),'energy']=0.
        s,p,z=self.prepared(raw)
        self.assertTrue(np.isfinite(z).all())
        self.assertGreater(int(p.isna().sum().sum()),0)

    def test_entire_history_missing_for_one_site(self):
        raw=self.raw.loc[~self.raw.site_id.eq('SYNTH_000')]
        s,p,z=self.prepared(raw)
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue(p['SYNTH_000'].isna().all())

    def test_cutoff(self):
        s,p,z=self.prepared()
        future=self.raw.iloc[:253].copy()
        future['ts']=pd.Timestamp('2025-06-01',tz='Asia/Seoul')
        future['energy']=1e12
        _,q,w=self.prepared(pd.concat([self.raw,future],ignore_index=True))
        pd.testing.assert_frame_equal(p,q)
        np.testing.assert_array_equal(z,w)

    def test_interval_end_boundary(self):
        cfg=dict(self.cfg,timestamp_convention='interval_end')
        raw=self.raw.iloc[:1].copy()
        raw['ts']=pd.Timestamp('2025-06-01 00:00',tz='Asia/Seoul')
        p,_=history_frame(raw,self.sites,cfg)
        self.assertEqual(p.index[0],pd.Timestamp('2025-05-31 23:00',tz='Asia/Seoul'))

    def test_duplicates_rejected(self):
        raw=pd.concat([self.raw,self.raw.iloc[:1]],ignore_index=True)
        with self.assertRaises(InputError):self.prepared(raw)

    def test_invalid_energy(self):
        raw=self.raw.copy()
        raw['energy']=raw.energy.astype(object)
        raw.loc[0,'energy']='invalid'
        with self.assertRaises(InputError):self.prepared(raw)

    def test_outputs_and_offline_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'new_run'
            candidates,rows=run(self.sites,self.raw,self.cfg,output,synthetic=True)
            self.assertEqual(set(candidates),{'balanced','profile','optimized'})
            for name in candidates:
                read_composition(output/f'composition_{name}.csv',self.sites,self.cfg)
            for name in ['grouping_comparison.csv','grouping_report.md','grouping_visualization.html','validation_report.json']:
                self.assertTrue((output/name).is_file())
            html=(output/'grouping_visualization.html').read_text(encoding='utf-8')
            for forbidden in ['<script src=','fetch(','XMLHttpRequest','https://']:
                self.assertNotIn(forbidden,html)
            self.assertIn('산점도 SVG 이미지 저장',html)
            self.assertIn('SYNTH_000',html)
            with self.assertRaises(FileExistsError):
                run(self.sites,self.raw,self.cfg,output,synthetic=True)

    def test_bad_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.csv'
            pd.DataFrame({'site_id':self.sites.site_id,'group':'G3'}).to_csv(path,index=False)
            with self.assertRaises(InputError):read_composition(path,self.sites,self.cfg)


if __name__=='__main__':unittest.main()
