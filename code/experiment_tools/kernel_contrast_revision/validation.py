"""Numerical and information-boundary checks before new experimental choices."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import unittest
import numpy as np
import torch
from . import geometry,models,policy


def fixture():
    ids = ('anchor','base','test','other')
    pool = SimpleNamespace(ids=ids,ix={k:i for i,k in enumerate(ids)},
        axes=('temperature','catalyst'),kinds=('continuous','categorical'),
        features=((0.,'A'),(.5,'A'),(0.,'B'),(.01,'B')))
    context = dict(available=np.array([False,True,True,True]),bo_index=1,
        base_logei=np.array([-1.,0.,-.1,-.2]),tie_priority=np.array([0,1,2,3]),epsilon=1.)
    fits = [dict(id='native',MLL=-1.,rho=[.8,.2],focus_axes=[],
        mu=np.array([0.,0.,0.,0.]),sd=np.ones(4),observation_sd=np.ones(4)),
        dict(id='axis_1_transfer',MLL=-1.,rho=[.8,.8],focus_axes=[1],
        mu=np.array([0.,0.,2.,2.]),sd=np.ones(4),observation_sd=np.ones(4))]
    return pool,[dict(candidate_id='anchor',y=0.)],context,fits


class RevisionChecks(unittest.TestCase):
    def test_irregular_pool_has_penalized_matches(self):
        p,h,c,f = fixture()
        p.axes = tuple('abcdef');p.kinds = ('continuous',)*6
        p.features = ((0.,)*6,(.1,.2,.3,.4,.5,.6),(.2,.1,.4,.3,.6,.5),(.9,)*6)
        m = geometry.Matches(p,h)
        values = m.for_axes([0])
        self.assertTrue((values['quality'][1:]>0).all())
        self.assertLess(values['quality'][2],1.)
        self.assertEqual(m.describe([0],2)['kind'],'joint_condition_test')

    def test_two_expert_score_and_budget(self):
        p,h,c,f = fixture()
        d,a = policy.select(p,h,c,f,'hypotheses',None,1)
        self.assertEqual(d['selected_id'],'test')
        self.assertAlmostEqual(a['score'][2],2/np.sqrt(2))
        c['base_logei'][2:] = -10
        self.assertEqual(policy.select(p,h,c,f,'hypotheses',None,1)[0]['selected_id'],'base')

    def test_terminal_follows_native_tie_priority(self):
        p,h,c,f = fixture();c['base_logei'][2]=0.;c['epsilon']=0.
        self.assertEqual(policy.select(p,h,c,f,'hypotheses',None,16)[0]['selected_id'],'base')

    def test_nominal_renaming_and_anchor_response_do_not_change_decision(self):
        p,h,c,f = fixture();d,_ = policy.select(p,h,c,f,'hypotheses',None,1)
        p.features = tuple((x,{'A':'zinc','B':'iron'}[cat]) for x,cat in p.features)
        h[0]['y'] = 1e20
        p.ids = tuple('x_'+k for k in p.ids);p.ix={k:i for i,k in enumerate(p.ids)};h[0]['candidate_id']=p.ids[0]
        new,_ = policy.select(p,h,c,f,'hypotheses',None,1)
        self.assertEqual(new['selected_id'],'x_'+d['selected_id'])
        self.assertAlmostEqual(new['score'],d['score'])

    def test_candidate_storage_permutation(self):
        p,h,c,f = fixture(); order=np.array([3,2,0,1])
        p.ids=tuple(p.ids[i] for i in order);p.features=tuple(p.features[i] for i in order);p.ix={k:i for i,k in enumerate(p.ids)}
        for key in ('available','base_logei','tie_priority'):c[key]=c[key][order]
        c['bo_index']=p.ix['base']
        for model in f:
            for key in ('mu','sd','observation_sd'):model[key]=model[key][order]
        self.assertEqual(policy.select(p,h,c,f,'hypotheses',None,1)[0]['selected_id'],'test')

    def test_feedback_changes_next_choice_and_fixed_weights_do_not(self):
        p,h,c,f = fixture(); prior=policy.initial_weights(f)
        d,_=policy.select(p,h,c,f,'feedback',None,1,initial_stored=prior)
        locked=deepcopy(d);obs=dict(candidate_id='test',y=-3.)
        fb=policy.feedback(d,obs);self.assertEqual(d,locked)
        self.assertLess(np.exp(fb['log_weights_after']['axis_1_transfer']),.001)
        h.append(obs);c['available'][2]=False
        new,_=policy.select(p,h,c,f,'feedback',fb['log_weights_after'],2,initial_stored=prior)
        self.assertEqual(new['selected_id'],'base')
        self.assertEqual(new['fixed_initial_selected_id'],'other')
        self.assertTrue(new['feedback_weights_change_selection'])
        fixed,_=policy.select(p,h,c,f,'soft_fixed',prior,2,initial_stored=prior)
        self.assertEqual(fixed['selected_id'],'other')
        self.assertEqual(policy.feedback(fixed,dict(candidate_id='other',y=-3.))['log_weights_after'],fixed['log_weights_before'])

    def test_current_training_MLL_not_recounted_after_initialization(self):
        p,h,c,f=fixture();prior=policy.initial_weights(f);before=policy.weights_for(f,'feedback',prior)[0]
        f[1]['MLL']=-1e6
        after=policy.weights_for(f,'feedback',prior)[0]
        np.testing.assert_array_equal(before,after)
        self.assertEqual(policy.weights_for(f,'hypotheses',None)[0][1],0.)

    def test_all_native_families_component_targets_and_custody(self):
        from . import study
        from kernel_contrast_qlogei import models as original
        study.initialize();previous=study.old.check(True)
        diagnostics=[]
        for task in study.native.TASKS:
            pool=study.native.load_task(study.ROOT,task,seed=0)
            initial=study.read(study.ROOT/previous['controls'][task]['path'])['history'][:4]
            gp=study.native.GPAdapter(pool,study.BASE/'preflight_cache'/study.SHORT[task])
            context=gp.context(initial,1);native_hash=models.state_hash(context['stats']['model'])
            fits=models.library(gp,initial,context,study.BASE/'libraries'/study.SHORT[task])
            self.assertEqual(len(fits),1+2*len(pool.axes))
            self.assertEqual(models.state_hash(context['stats']['model']),native_hash)
            x,y,_,_,_=original.data(gp,initial)
            probes=np.array([[.5 if k=='continuous' else 0. for k in pool.kinds]]*2)
            target_errors=[]
            for item in fits[1:]:
                raw=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
                m=original.factory(pool,x,y);m.load_state_dict(raw['state']);m.eval()
                axis=item['focus_axes'][0];probe=probes.copy();probe[1,axis]+=.2 if pool.kinds[axis]=='continuous' else 1.
                for changed in item['changed_components']:
                    component=m.covar_module.get_submodule(changed['module'])
                    with torch.no_grad():k=component(torch.tensor(probe,dtype=torch.double)).to_dense().numpy()
                    rho=k[0,1]/np.sqrt(k[0,0]*k[1,1]);err=abs(rho-item['component_target'])
                    self.assertLess(err,1e-10);target_errors.append(err)
            d,a=policy.select(pool,initial,context,fits,'feedback',None,1)
            cover=np.max([a['matches'].for_axes([i])['quality'] for i in range(len(pool.axes))],axis=0)
            self.assertTrue((cover[context['available']]>0).all())
            diversity=max(float(np.max(np.abs(np.array(m['rho'])-fits[0]['rho']))) for m in fits[1:])
            self.assertGreater(diversity,.05)
            diagnostics.append(dict(task=task,experts=len(fits),max_component_target_error=max(target_errors),
                max_correlation_diversity=diversity,budget_candidates=int(a['budget'].sum()),
                matching_coverage=int((cover[context['available']]>0).sum()),
                maximum_disagreement_in_budget=d['maximum_budget_score'],native_state_unchanged=True))
            gp.release()
        study.write(study.BASE/'preflight_models.json',diagnostics,replace=True)


def preflight():
    from . import study
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(RevisionChecks)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    value=dict(status='PASS_REVISION_PREFLIGHT' if result.wasSuccessful() else 'FAIL_REVISION_PREFLIGHT',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        new_experimental_queries=0,new_LLM_calls=0,test_output=stream.getvalue(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    study.write(study.BASE/'preflight.json',value,replace=True)
    if not result.wasSuccessful():
        print(stream.getvalue());raise RuntimeError('Revision preflight failed')
    return {k:v for k,v in value.items() if k!='test_output'}


if __name__=='__main__':
    print(json.dumps(preflight()))
