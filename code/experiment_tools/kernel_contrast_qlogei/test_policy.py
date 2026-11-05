"""Scientific invariants: budget, nominal encoding, feedback and GP custody."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from . import policy, models


def example():
    features = ((0.,'A'),(.5,'A'),(0.,'B'),(.5,'B'))
    pool = SimpleNamespace(ids=('anchor','base','contrast','joint'),
        axes=('temperature','catalyst'),kinds=('continuous','categorical'),
        features=features,ix={k:i for i,k in enumerate(('anchor','base','contrast','joint'))})
    context = dict(available=np.array([False,True,True,True]),bo_index=1,
        base_logei=np.array([-1.,0.,-.1,-.2]),tie_priority=np.array([0,1,3,2]))
    fits = [dict(id='first',source='native',MLL=-1.,rho=[.8,.9],
        mu=np.array([0.,1.,1.,1.]),sd=np.ones(4),observation_sd=np.ones(4)),
        dict(id='second',source='alternative',MLL=-1.001,rho=[.8,.1],
        mu=np.array([0.,1.,4.,9.]),sd=np.ones(4),observation_sd=np.ones(4))]
    return pool,[dict(candidate_id='anchor',y=0.)],context,fits


class DecisionInvariants(unittest.TestCase):
    def test_input_contrast_and_budget(self):
        p,h,c,m = example()
        d,_ = policy.select(p,h,c,m,1)
        self.assertEqual(d['selected_id'],'contrast')
        self.assertEqual(d['anchor_id'],'anchor')
        self.assertEqual(d['isolated_axis'],'catalyst')
        # The joint change has larger disagreement but is not an isolated contrast.
        c['base_logei'][2] = -9.
        self.assertEqual(policy.select(p,h,c,m,1)[0]['selected_id'],'base')

    def test_nonplausible_fit_cannot_intervene(self):
        p,h,c,m = example(); m[1]['MLL'] = -2.
        self.assertEqual(policy.select(p,h,c,m,1)[0]['selected_id'],'base')

    def test_terminal_uses_exact_BO_tie(self):
        p,h,c,m = example(); c['base_logei'][2] = 0.
        d,_ = policy.select(p,h,c,m,16)
        self.assertEqual(d['selected_id'],'base')
        self.assertEqual(d['native_logei_gap'],0.)

    def test_response_blind_anchors_and_nominal_bijection(self):
        p,h,c,m = example(); first = policy.select(p,h,c,m,1)[0]
        h[0]['y'] = 1e9
        p.features = tuple((x,{'A':'zinc','B':'iron'}[cat]) for x,cat in p.features)
        p.ids = tuple('renamed_'+k for k in p.ids); p.ix = {k:i for i,k in enumerate(p.ids)}
        h[0]['candidate_id'] = p.ids[0]
        second = policy.select(p,h,c,m,1)[0]
        self.assertEqual(first['selected_id'],'contrast')
        self.assertEqual(second['selected_id'],'renamed_contrast')
        self.assertEqual(first['standardized_disagreement'],second['standardized_disagreement'])

    def test_permuting_candidate_storage_preserves_decision(self):
        p,h,c,m = example(); order = np.array([3,2,0,1])
        p.ids = tuple(p.ids[j] for j in order); p.features = tuple(p.features[j] for j in order)
        p.ix = {k:i for i,k in enumerate(p.ids)}
        for key in ('available','base_logei','tie_priority'):
            c[key] = c[key][order]
        c['bo_index'] = p.ix['base']
        for model in m:
            for key in ('mu','sd','observation_sd'):
                model[key] = model[key][order]
        self.assertEqual(policy.select(p,h,c,m,1)[0]['selected_id'],'contrast')

    def test_precommitted_feedback_distinguishes_explanations(self):
        p,h,c,m = example(); d,_ = policy.select(p,h,c,m,1)
        locked = deepcopy(d)
        f = policy.feedback(d,dict(candidate_id='contrast',y=4.),h)
        self.assertEqual(f['pair_comparison']['preferred'],'second')
        self.assertAlmostEqual(f['pair_comparison']['log_predictive_ratio'],-4.5)
        self.assertEqual(d,locked)
        self.assertEqual(f['observed_contrast_y'],4.)

    def test_same_data_alternative_does_not_change_native_GP(self):
        from chem.benchmarks import knowledge_gp as native
        root = Path(__file__).resolve().parents[3]
        # Two different native model families; no candidate outcome is queried.
        archived = root/'experiments/runs/knowledge_loop_20261101/v9_kf_rank/runs'
        for task in ('Reizman_Suzuki_Case_2','SnAr'):
            with self.subTest(task=task),TemporaryDirectory(dir=root/'.tmp') as tmp:
                import json
                pool = native.load_task(root,task,seed=0)
                history = json.loads((archived/(task+'_bo')/'state.json').read_text(encoding='utf-8'))['history'][:4]
                gp = native.GPAdapter(pool,Path(tmp)/'native')
                context = gp.context(history,1)
                before = models.state_hash(context['stats']['model'])
                own = models.parameters(context['stats']['model'])
                fits = models.alternatives(gp,history,context,[dict(id='parent',role='native',parameters=own)],Path(tmp)/'alt')
                self.assertEqual(before,models.state_hash(context['stats']['model']))
                self.assertEqual(len(fits),2)
                self.assertTrue(all(np.isfinite(f['MLL']) for f in fits))
                gp.release()


if __name__ == '__main__':
    unittest.main()
