"""Closed-loop commitments, trust, and bounded decisions on synthetic inputs."""
from copy import deepcopy
from types import SimpleNamespace
import json
import unittest

import numpy as np

from chem.benchmarks import knowledge_numeric_engine as e
from chem.benchmarks import knowledge_numeric_evidence as credit
from chem.benchmarks import knowledge_numeric_policy as policy
from chem.benchmarks.knowledge_llm import digest


def setup():
    ids=[f'{s}_{l}' for s in 'ABCD' for l in ('plus','ref')]
    rows=[[l,s] for s in 'ABCD' for l in ('plus','ref')]
    pool=SimpleNamespace(task='synthetic',scale=100.,unit='percent',axes=['ligand','solvent'],
        kinds=['categorical','categorical'],ids=ids,ix={k:i for i,k in enumerate(ids)},
        features=np.asarray(rows,object),points=[dict(ligand=l,solvent=s) for l,s in rows])
    history=[dict(candidate_id=ids[j],y=.3,raw_value=30.) for j in (1,3,5,7)]
    state=dict(history=history,hypotheses=[],calls=[],decisions=[],verification_used=0,status='READY')
    a=dict(op='add',id=None,plus_id='A_plus',minus_id='A_ref',
        focus={'ligand':dict(direction='prefer_plus',allowed=['plus','ref'])},
        background={'solvent':dict(allowed=list('ABCD'))},
        numeric_prior=dict(delta_mean_raw=30.,delta_sd_raw=10.),reason='mechanistic_prior: synthetic')
    proposal=dict(actions=[a],summary='synthetic prospective declaration')
    gp=SimpleNamespace(numeric_calibrate=True,birth_store=FakeBirth())
    return pool,state,gp,proposal


class FakeBirth:
    def birth(self,h,history,context):
        return dict(snapshot_sha256='frozen-birth',prior_mean=.3,prior_variance=.1,noise_variance=.01)

    def predict(self,h,history,indices,calibrate=True):
        js=np.asarray(indices);n=len(js)
        def gaussian(mu):
            return dict(mean=np.asarray(mu),latent_variance=np.full(n,.04),observation_variance=np.full(n,.05))
        null=gaussian(np.full(n,.3))
        full=gaussian(np.where(np.isin(js,[0,4]),.65,.3))
        cut=gaussian(np.where(np.isin(js,[0,4]),.5,.3))
        actual=full if calibrate else cut
        return dict(**actual,full=full,null=null,cut_feedback=cut,
            variance_only=dict(actual,mean=null['mean']),beta_mean=.3,beta_variance=.1,
            residual_feature=np.where(js==2,0.,1.),block_receipt=dict(synthetic=True))


def context(pool,state,bo=2):
    available=np.array([key not in {r['candidate_id'] for r in state['history']} for key in pool.ids])
    base=np.full(len(pool.ids),-2.);base[bo]=0.;base[0]=-.03;base[4]=-.04
    stats=dict(mu_y=np.full(8,.3),sd_y=np.full(8,.2),sd_predictive_y=np.full(8,np.sqrt(.05)),
        fit_parameters=dict(synthetic=True),fit_key='fit')
    return dict(stats=stats,base_logei=base,available=available,eligible=available,
        bo_index=bo,epsilon=.1,tie_priority=np.arange(8),acquisition_meta=dict(synthetic=True))


def first():
    pool,state,gp,proposal=setup();ctx=context(pool,state)
    commit=e.plan(pool,gp,state,1,ctx,proposal,[pool.ids[j] for j in np.flatnonzero(ctx['eligible'])])
    return pool,state,gp,commit


def observation(key,y):return dict(candidate_id=key,y=y,raw_value=y*100)


class NumericEngineTests(unittest.TestCase):
    def test_verified_prediction_creates_credit_then_changes_next_choice(self):
        pool,state,gp,commit=first()
        self.assertEqual(commit['mode'],'optional_verification');self.assertEqual(commit['selected_id'],'A_plus')
        self.assertEqual(state['hypotheses'],[])
        json.dumps(commit,allow_nan=False) # includes explicit no-op lambda serialization
        e.feedback(state,commit,observation('A_plus',.65))
        h=state['hypotheses'][0]
        self.assertGreater(credit.weight(h),0.);self.assertEqual(len(h['ledger']['events']),1)
        next_commit=e.plan(pool,gp,state,2,context(pool,state))
        self.assertEqual(next_commit['mode'],'knowledge_guided');self.assertEqual(next_commit['selected_id'],'C_plus')
        self.assertGreater(len(next_commit['selected_mixture']),0)
        self.assertLessEqual(sum(x['effective_mixture_weight'] for x in next_commit['deployed_versions']),.5)
        self.assertEqual(len(state['history']),5)

    def test_wrong_forecast_cannot_receive_credit(self):
        pool,state,gp,commit=first();e.feedback(state,commit,observation('A_plus',.2))
        self.assertEqual(credit.weight(state['hypotheses'][0]),0.)
        self.assertEqual(e.plan(pool,gp,state,2,context(pool,state))['mode'],'base_bo')

    def test_duplicates_prefix_and_forecast_tampering_are_atomic(self):
        pool,state,gp,commit=first();before=deepcopy(state)
        for mode in ('point','prefix','forecast','semantic'):
            bad=deepcopy(commit)
            if mode=='point':bad['selected_id']='C_plus'
            elif mode=='prefix':bad['history_sha256']='changed'
            elif mode=='semantic':bad['hypotheses_at_commit'][0]['numeric_prior']['delta_mean_raw']=99.
            else:
                pred=bad['shadows'][0]['prediction'];pred['current_gp_sha256']='changed'
                pred['prediction_sha256']=digest({k:v for k,v in pred.items() if k!='prediction_sha256'})
            with self.assertRaises(ValueError):e.feedback(state,bad,observation('A_plus',.65))
            self.assertEqual(state,before)
        e.feedback(state,commit,observation('A_plus',.65));after=deepcopy(state)
        with self.assertRaises(ValueError):e.feedback(state,commit,observation('A_plus',.65))
        self.assertEqual(state,after)

    def test_cut_arm_uses_cut_distribution_and_verification_never_deploys(self):
        pool,state,gp,commit=first();e.feedback(state,commit,observation('A_plus',.65))
        ctx=context(pool,state)
        verify=e.plan(pool,gp,state,2,ctx,mode='verification')
        self.assertEqual(verify['mode'],'base_bo');self.assertEqual(verify['deployed_versions'],[])
        with self.assertRaises(ValueError):e.plan(pool,gp,state,2,ctx,mode='no_calibration')
        gp.numeric_calibrate=False
        cut=e.plan(pool,gp,state,2,ctx,mode='no_calibration')
        self.assertEqual(cut['shadows'][0]['prediction']['models']['raw_expert'],
                         cut['shadows'][0]['prediction']['models']['cut_feedback'])

    def test_information_of_bo_outside_scope_is_computed(self):
        pool,state,gp,proposal=setup()
        proposal['actions'][0]['background']['solvent']['allowed']=['A','C']
        ctx=context(pool,state)
        commit=e.plan(pool,gp,state,1,ctx,proposal,['A_plus','C_plus'])
        self.assertEqual(commit['mode'],'optional_verification')
        record=commit['numeric_candidates'][0]
        self.assertIn(2,record['query_indices']);self.assertNotIn(2,record['scope_indices'])
        class EquallyInformative(FakeBirth):
            def predict(self,*args,**kwargs):
                pred=super().predict(*args,**kwargs);pred['residual_feature'][:]=1.;return pred
        gp.birth_store=EquallyInformative()
        denied=e.plan(pool,gp,state,1,ctx,proposal,['A_plus','C_plus'])
        self.assertEqual(denied['mode'],'base_bo')

    def test_noop_and_uncovered_weighted_winner_fall_back_to_bo(self):
        pool,state,gp,commit=first();e.feedback(state,commit,observation('A_plus',.65))
        # Negative expert only lowers BO. An unchanged third point cannot gain
        # a knowledge-guided label without a non-noop frozen forecast.
        class Negative(FakeBirth):
            def predict(self,*args,**kwargs):
                pred=super().predict(*args,**kwargs)
                js=np.asarray(args[2]);mu=np.where(js==2,-1.,.3)
                for key in ('full','cut_feedback'):
                    pred[key]['mean']=mu.copy()
                pred['mean']=mu;return pred
        gp.birth_store=Negative();ctx=context(pool,state);ctx['base_logei'][4]=-.01
        denied=e.plan(pool,gp,state,2,ctx)
        self.assertEqual(denied['selected_id'],'B_plus');self.assertEqual(denied['mode'],'base_bo')
        self.assertEqual(denied['fallback_reason'],'uncovered_or_no_effect_weighted_winner')

    def test_negative_inheritance_deduplicates_physical_observation(self):
        _,state,_,commit=first();e.feedback(state,commit,observation('A_plus',.65))
        h=state['hypotheses'][0];event=h['ledger']['events'][0]
        inherited=dict(event,applicable=True,**{name:-.2 for name in credit.GAINS})
        h['inherited_negative_evidence']=[inherited,dict(inherited,event_id='other-rule')]
        got=credit.evidence_summary(h)
        self.assertEqual(got['observations'],1)
        self.assertEqual(got['effective_scores'],{name:-.2 for name in credit.GAINS})
        self.assertEqual(got['trusted_weight'],0.)

    def test_no_shadow_bo_feedback_still_requires_finite_observation(self):
        pool,state,gp,_=setup();commit=e.plan(pool,gp,state,1,context(pool,state),mode='bo')
        self.assertEqual(commit['shadows'],[]);before=deepcopy(state)
        for field in ('y','raw_value'):
            for value in (float('nan'),float('inf'),True,'0.4'):
                bad=observation(commit['selected_id'],.4);bad[field]=value
                with self.assertRaises(ValueError):e.feedback(state,commit,bad)
                self.assertEqual(state,before)


if __name__=='__main__':unittest.main()
