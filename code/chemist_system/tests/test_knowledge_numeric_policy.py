"""Physical features/versioning/inheritance; no GP, model call or oracle."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from chem.benchmarks import knowledge_numeric_policy as p


def dataset(categorical=True):
    if categorical:
        axes=['ligand','solvent'];kinds=['categorical','categorical']
        rows=[['L+','S1'],['L-','S1'],['L+','S2'],['L-','S2'],
              ['L+','S3'],['L-','S3'],['third','S2']]
    else:
        axes=['temperature','time','catalyst'];kinds=['continuous','continuous','categorical']
        rows=[[.8,.6,'Pd'],[.4,.4,'Pd'],[1.,.8,'Pd'],[0.,.2,'Pd'],
              [.6,.5,'Pd'],[.8,.6,'other'],[.4,.8,'Pd']]
    ids=['a_plus','a_ref','b_plus','b_ref','c_plus','c_ref','extra']
    return SimpleNamespace(axes=axes,kinds=kinds,ids=ids,ix={k:i for i,k in enumerate(ids)},
        features=np.asarray(rows,dtype=object),points=[dict(zip(axes,r)) for r in rows],
        task='synthetic',scale=100.,unit='yield percent')


def history(*pairs):return [dict(candidate_id=k,y=y,raw_value=100*y) for k,y in pairs]


def action(categorical=True,**overrides):
    focus={'ligand':{'direction':'prefer_plus','allowed':['L+','L-','third']}} if categorical else {
        'temperature':{'direction':'increase','bounds':[0.,1.]},
        'time':{'direction':'increase','bounds':[0.,1.]}}
    a=dict(op='add',id=None,plus_id='a_plus',minus_id='a_ref',focus=focus,
        background={'solvent':{'allowed':['S1','S2','S3']}} if categorical else {},
        numeric_prior=dict(delta_mean_raw=15.,delta_sd_raw=8.),reason='mechanistic_prior: uncertain local contrast')
    a.update(overrides);return a


def proposal(*actions):return dict(actions=list(actions),summary='Declare the numerical local hypothesis prospectively.')


def create(categorical=True):
    pool=dataset(categorical)
    past=history(('a_ref',.5),('b_ref',.4),('c_ref',.6)) if categorical else history(('a_ref',.5))
    menu=[key for key in pool.ids if key not in {x['candidate_id'] for x in past}]
    hs,events=p.review_and_apply(pool,[],proposal(action(categorical)),menu,1,past)
    assert events[0]['accepted'],events
    return pool,past,menu,hs


class NumericPolicyTests(unittest.TestCase):
    def test_fixed_continuous_two_focus_feature_and_scope(self):
        pool,past,menu,hs=create(False);h=hs[0]
        np.testing.assert_allclose(h['feature_vector'],[1.,0.,1.,-1.,.5,0.,.5],atol=2e-16)
        self.assertEqual(h['numeric_prior']['delta_mean_y'],.15)
        self.assertEqual(h['numeric_prior']['delta_sd_y'],.08)
        self.assertIsNone(h['birth']);self.assertEqual(h['birth_prefix'],past)
        self.assertEqual(h['ledger']['trusted_weight'],0.)

    def test_categories_are_indicators_not_ordinal_distances(self):
        pool,past,menu,hs=create();original=hs[0]['feature_vector']
        self.assertEqual(original,[1.,0.,1.,0.,1.,0.,0.])
        renamed=deepcopy(pool)
        aliases={'L+':'ligand_900','L-':'ligand_1','third':'ligand_50'}
        renamed.features[:,0]=[aliases[x] for x in renamed.features[:,0]]
        a=action();a['focus']['ligand']['allowed']=list(aliases.values())
        changed,ev=p.review_and_apply(renamed,[],proposal(a),menu,1,past)
        self.assertTrue(ev[0]['accepted']);self.assertEqual(changed[0]['feature_vector'],original)

    def test_numeric_abstention_does_not_create_or_retire_an_expert(self):
        pool,past,menu,hs=create()
        blank,ev=p.review_and_apply(pool,[],proposal(action(numeric_prior=None)),menu,1,past)
        self.assertEqual(blank,[]);self.assertEqual(ev[0]['outcome'],'numeric_abstention')
        out,ev=p.review_and_apply(pool,hs,proposal(action(op='revise',id='H001',numeric_prior=None)),menu,2,past)
        self.assertEqual(out,hs);self.assertFalse(ev[0]['created_expert'])

    def test_prior_validation_rejects_nonfinite_boolean_and_nonpositive_values(self):
        pool=dataset();past=history(('a_ref',.5))
        for value in (True,False,0.,-1.,float('nan'),float('inf'),'8'):
            for key in ('delta_mean_raw','delta_sd_raw'):
                a=action();a['numeric_prior'][key]=value
                out,ev=p.review_and_apply(pool,[],proposal(a),['a_plus'],1,past)
                self.assertFalse(ev[0]['accepted']);self.assertEqual(out,[])

    def test_geometry_still_checks_original_example_before_background_transfer(self):
        pool=dataset();past=history(('b_ref',.5))
        out,ev=p.review_and_apply(pool,[],proposal(action(minus_id='b_ref')),['a_plus'],1,past)
        self.assertFalse(ev[0]['accepted']);self.assertEqual(out,[])
        pool,past,menu,hs=create()
        out,ev=p.review_and_apply(pool,hs,proposal(action(plus_id='a_ref')),menu,2,past)
        self.assertFalse(ev[0]['accepted']);self.assertEqual(out,hs)

    def test_version_identity_includes_numeric_anchors_and_feature(self):
        pool,past,menu,hs=create()
        a=action(op='revise',id='H001',numeric_prior=dict(delta_mean_raw=16.,delta_sd_raw=8.))
        out,ev=p.review_and_apply(pool,hs,proposal(a),menu,2,past)
        self.assertTrue(ev[0]['created_expert']);self.assertEqual(out[-1]['version'],2)
        self.assertNotEqual(out[-1]['semantic_key'],hs[0]['semantic_key'])
        self.assertEqual(out[0]['status'],'retracted');self.assertEqual(out[-1]['ledger']['trusted_weight'],0.)
        for mutate in ('numeric_prior','feature_vector','birth_prefix'):
            tampered=deepcopy(hs)
            if mutate=='numeric_prior':tampered[0]['numeric_prior']['delta_mean_raw']=99.
            elif mutate=='feature_vector':tampered[0]['feature_vector'][0]=.9
            else:tampered[0]['birth_prefix'][0]['y']=.9
            with self.assertRaises(ValueError):p.review_and_apply(pool,tampered,proposal(),menu,2,past)

    def test_cross_background_negative_evidence_survives_once_without_positive_credit(self):
        pool,past,menu,hs=create();h=hs[0]
        h['ledger']['events']=[dict(event_id='event-one',point_id='b_plus',reference_id='a_ref',query_round=1,
            gp_gain=-.2,null_gain=.3,mean_gain=-.4,null_mean_gain=-.1,comparison_success=True)]
        h['ledger']['seen']=['b_plus'];h['ledger']['trusted_weight']=.4
        past+=history(('b_plus',.3));menu.remove('b_plus')
        a=action(op='revise',id='H001',plus_id='c_plus',minus_id='c_ref')
        out,ev=p.review_and_apply(pool,hs,proposal(a),menu,2,past)
        self.assertTrue(ev[0]['created_expert'])
        inherited=out[-1]['inherited_negative_evidence']
        self.assertEqual(len(inherited),1);self.assertTrue(inherited[0]['applicable'])
        self.assertEqual([inherited[0][k] for k in p.GAIN_FIELDS],[-.2,0.,-.4,-.1])
        self.assertEqual(out[-1]['ledger']['events'],[]);self.assertEqual(out[-1]['ledger']['trusted_weight'],0.)
        a['numeric_prior']['delta_mean_raw']=16.
        again,ev=p.review_and_apply(pool,out,proposal(a),menu,3,past)
        self.assertTrue(ev[0]['created_expert'])
        self.assertEqual(len(again[-1]['inherited_negative_evidence']),1)
        self.assertEqual(p.evidence_summary(again[-1])['effective_scores']['gp_gain'],-.2)

    def test_direction_family_and_new_scope_control_negative_applicability(self):
        pool,past,menu,hs=create()
        hs[0]['ledger']['events']=[dict(event_id='bad',point_id='b_plus',reference_id='a_ref',query_round=1,
                                      gp_gain=-.2,null_gain=-.2,mean_gain=-.2)]
        past+=history(('b_plus',.3));menu.remove('b_plus')
        a=action(op='revise',id='H001',plus_id='c_plus',minus_id='c_ref',background={'solvent':{'allowed':['S3']}})
        out,ev=p.review_and_apply(pool,hs,proposal(a),menu,2,past)
        self.assertTrue(ev[0]['accepted']);self.assertFalse(out[-1]['inherited_negative_evidence'][0]['applicable'])
        self.assertEqual(out[-1]['inherited_negative_evidence'][0]['null_mean_gain'],0.)

    def test_bounded_actions_live_rules_and_duplicate_semantics(self):
        pool,past,menu,hs=create()
        same,ev=p.review_and_apply(pool,hs,proposal(action()),menu,2,past)
        self.assertEqual(same,hs);self.assertEqual(ev[0]['outcome'],'duplicate_semantics_preserved')
        actions=[action(numeric_prior=dict(delta_mean_raw=x,delta_sd_raw=8.)) for x in (16.,17.,18.)]
        four,ev=p.review_and_apply(pool,hs,proposal(*actions),menu,2,past)
        self.assertEqual(sum(h['status']!='retracted' for h in four),4)
        out,ev=p.review_and_apply(pool,four,proposal(action(numeric_prior=dict(delta_mean_raw=19.,delta_sd_raw=8.))),menu,3,past)
        self.assertFalse(ev[0]['accepted']);self.assertEqual(out,four)
        with self.assertRaises(ValueError):p.review_and_apply(pool,[],proposal(*([action()]*4)),menu,1,past)


if __name__=='__main__':unittest.main()
