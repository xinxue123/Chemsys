"""Risk-oriented checks for continuous-domain transport boundaries."""
import unittest
from copy import deepcopy
import numpy as np
from chem.benchmarks import continuous_benchmark_data as data
from chem.benchmarks import continuous_benchmark_engine as e
from chem.benchmarks import continuous_benchmark_llm as llm

class ContinuousBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.pool=data.read(data.ROOT/'SnAr/pool.json')
        self.state={'history':[data.reveal('SnAr',k) for k in self.pool['initial_ids']],
                    'knowledge':{'confidence':.7,'clauses':[]},'decisions':[],'validations':[],'region_trust':{}}
    def test_raw_units_and_typed_evidence(self):
        row=llm.observations(self.pool,self.state)[0];axis='tau'
        raw={'actions':[{'id':'H1','operation':'add','bounds':[.2,.8],'reason':'test preference',
            'support':[{'index':1,'axis':axis,'value':row['point'][axis],'objective':row['objective']}]}],'summary':'uncertain preference'}
        self.assertEqual(llm.validate_region(raw,self.pool,self.state)['knowledge']['clauses'][0]['axis'],axis)
        bad=deepcopy(raw);bad['actions'][0]['support'][0]['objective']+=1
        with self.assertRaises(ValueError):llm.validate_region(bad,self.pool,self.state)
        bad=deepcopy(raw);bad['actions'][0]['support'][0]['index']=5
        with self.assertRaises(ValueError):llm.validate_region(bad,self.pool,self.state)
    def test_geometry_rename_cannot_restore_blocked_trust(self):
        spec={'allowed':{'tau':[.2,.8]},'confidence':.7};d={'region':spec,'fallback':True,'amplitude':0.}
        gate=e.trust.permission(self.state,d,1);key=gate['geometry_key']
        d.update(fallback=False,amplitude=1.)
        self.assertTrue(e.trust.permission(self.state,d,2)['trust_blocked'])
        d['region']['confidence']=.99
        self.assertEqual(e.trust.permission(self.state,d,2)['geometry_key'],key)
        self.assertEqual(e.trust.permission(self.state,d,2)['amplitude_after_trust'],0.)
    def test_proposal_projection_excludes_queried_points(self):
        first=self.pool['initial_ids'][0];point=self.pool['points'][self.pool['ids'].index(first)]
        projected=llm.project(self.pool,self.state,1,point)
        self.assertNotIn(projected['candidate_id'],self.pool['initial_ids'])
        self.assertGreater(projected['normalized_distance'],0.)
    def test_HPLC_unique_inputs_and_first_source_measurement(self):
        p=data.read(data.ROOT/'HPLC/pool.json');raw=np.loadtxt(data.ROOT/'sources/hplc.csv',delimiter=',')
        self.assertEqual(len(p['ids']),1007)
        self.assertEqual(len({tuple(r[a] for a in p['axes']) for r in p['points']}),len(p['ids']))
        with np.load(data.ROOT/'HPLC/oracle.npz') as f:
            for j,k in enumerate(p['source_row_indices_0based']):
                self.assertEqual(float(f['raw_values'][j]),float(raw[k,6]))
                self.assertEqual(k,int(np.flatnonzero(np.all(raw[:,:6]==raw[k,:6],axis=1))[0]))

if __name__=='__main__':unittest.main()
