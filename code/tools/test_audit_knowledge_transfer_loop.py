"""Synthetic independent V5 audit checks, no production imports or targets."""
from copy import deepcopy
from math import log
import unittest
import numpy as np

import audit_knowledge_transfer_loop as a


def example():
    features = [(.3,'A',.5),(.7,'A',.5),(.4,'B',.8),(.8,'B',.9),(.8,'B',.1)]
    ids = ['known-A','origin-plus','known-B','new-B','incomparable-B']
    pool = dict(ids=ids,ix={x:i for i,x in enumerate(ids)},features=features,
                axes=['temperature','solvent','time'],kinds=['continuous','categorical','continuous'])
    labels = sorted(['A','B'],key=a.digest)
    h = dict(id='H001',version=1,status='active',created_round=1,plus_id='origin-plus',minus_id='known-A',
        plus_features=list(features[1]),minus_features=list(features[0]),
        focus={'temperature':{'direction':'increase','bounds':[.1,.9]}},
        background={'solvent':{'allowed':labels},'time':{'bounds':[0.,1.]}},categorical_preference={},
        scope={'background_radius':.25,'axes':{'temperature':{'kind':'continuous','bounds':[.1,.9]},
            'solvent':{'kind':'categorical','allowed':labels},'time':{'kind':'continuous','bounds':[0.,1.]}}},
        inherited_negative_evidence=[],ledger=dict(prediction_score=0.,score=0.,successes=0,failures=0,
            trusted_weight=0.,weight=0.,events=[],seen=[]))
    h['semantic_key']=a.digest(a.contract(h))
    return pool,h


class IndependentTransferAuditTests(unittest.TestCase):
    def test_declared_background_allows_transfer_but_not_incomparable_reference(self):
        pool,h=example()
        a.rule_geometry_audit(pool,h)
        prefix=[dict(candidate_id='known-A',y=.9),dict(candidate_id='known-B',y=.5)]
        ps=a.probes(pool,h,prefix,np.array([False,False,False,True,True]))
        self.assertEqual(set(ps),{3})
        self.assertEqual(ps[3]['reference_id'],'known-B')
        self.assertEqual(a.outside_origin_background(pool,h,'new-B'),['solvent','time'])
        self.assertEqual(a.outside_origin_background(pool,h,'origin-plus'),[])

    def test_undeclared_expansion_and_focus_override_rejected(self):
        pool,h=example()
        altered=deepcopy(h);altered['background']={}
        with self.assertRaises(AssertionError):a.rule_geometry_audit(pool,altered)
        altered=deepcopy(h);altered['background']['temperature']={'bounds':[0.,1.]}
        with self.assertRaises(AssertionError):a.rule_geometry_audit(pool,altered)

    def test_declaration_bound_to_original_action(self):
        _,h=example()
        action=dict(plus_id=h['plus_id'],minus_id=h['minus_id'],background=deepcopy(h['background']))
        c=dict(query=1,hypotheses_at_commit=[h],proposal={'actions':[action]},
               reviews=[dict(action_index=0,accepted=True,outcome='add',id=h['id'],version=1)])
        a.declarations_audit(c)
        c['proposal']['actions'][0]['background']={}
        with self.assertRaises(AssertionError):a.declarations_audit(c)

    def test_tiny_increment_cannot_release_old_credit(self):
        limit,ceiling,after=a.recovery_limit(.1,.45,1e-9,True)
        self.assertAlmostEqual(limit,5e-10,places=20)
        self.assertEqual(after,ceiling)
        self.assertLess(after,.100000001)
        self.assertLess(after,.45)
        for gain,success in [(1.,False),(-.2,True),(0.,True)]:
            limit,_,after=a.recovery_limit(.1,.45,gain,success)
            self.assertEqual(limit,0.)
            self.assertEqual(after,.1)
        self.assertEqual(a.recovery_limit(.3,.1,.8,True)[2],.1)

    def test_transfer_roles_distinguish_forecast_weighted_and_optional(self):
        pool,h=example()
        sh=dict(id=h['id'],version=1,prediction={'reference_id':'known-B'})
        c=dict(hypotheses_at_commit=[h],selected_id='new-B',shadows=[sh],
               selected_mixture=[],verification={'id':h['id'],'version':1},mode='optional_verification')
        got=a.transfer_role(pool,c)
        self.assertEqual((got['transfer_forecasts'],got['weighted_transfer_forecasts'],got['optional_target_transfers']),(1,0,1))
        c.update(mode='knowledge_guided',verification=None,selected_mixture=[dict(id=h['id'],version=1,weight=.2)])
        got=a.transfer_role(pool,c)
        self.assertEqual((got['weighted_transfer_forecasts'],got['guided_point_has_transfer_expert']),(1,1))
        c.update(selected_id='origin-plus',mode='base_bo')
        self.assertEqual(a.transfer_role(pool,c)['transfer_forecasts'],0)

    def test_verification_control_retains_credit_but_cannot_use_it(self):
        ids=['ref-low','ref-high','new-a','new-b']
        pool=dict(ids=ids,ix={x:i for i,x in enumerate(ids)},axes=['temperature'],kinds=['continuous'],
            features=[(.1,),(.2,),(.5,),(.7,)])
        events=[dict(query_round=i,point_id='past-'+str(i),reference_id='ref-low',delta_log_score=1.,fact_success=True) for i in (1,2)]
        h=dict(id='H001',version=1,status='active',focus={'temperature':{'direction':'increase','bounds':[0.,1.]}},
            background={},categorical_preference={},scope={'background_radius':.25,'axes':{'temperature':{'kind':'continuous','bounds':[0.,1.]}}},
            inherited_negative_evidence=[],ledger=dict(events=events,prediction_score=2.,score=2.,successes=2,failures=0,trusted_weight=.5))
        h['semantic_key']=a.digest(a.contract(h))
        prefix=[dict(candidate_id='ref-low',y=.4),dict(candidate_id='ref-high',y=.6)]
        arrays=dict(eligible=np.array([False,False,True,True]),base_logei=np.array([-2.,-2.,0.,-.05]),
            tie_priority=np.arange(4),mu_y=np.array([.4,.6,.8,.3]),sd_y=np.sqrt([.01,.01,.001,.04]),
            sd_predictive_y=np.sqrt([.011,.011,.002,.05]))
        rows,calc=a.calculations(pool,[h],prefix,arrays)
        self.assertEqual(rows[0]['weight'],.5)
        offset=calc[('H001',1)]['offsets']
        combined=arrays['base_logei']+np.logaddexp(log(.5),log(.5)+offset)
        selected=a.select(combined,arrays['eligible'],arrays['tie_priority'])
        self.assertEqual(selected,3)
        params=dict(last_verification_round=10,maximum_optional_verifications=1,verification_maximum_gap=log(1.1),minimum_incremental_information=.01)
        c=dict(query=3,hypotheses_at_commit=[h],coverage=rows,epsilon=1.,
            selected_id='new-b',mode='knowledge_guided',verification=None,combined_logei=float(combined[3]),
            deployed_versions=[dict(id='H001',version=1,weight=.5,effective_mixture_weight=.5)])
        a.selection_audit(pool,prefix,c,arrays,'full',0,params)
        c.update(selected_id='new-a',mode='base_bo',combined_logei=0.,deployed_versions=[])
        a.selection_audit(pool,prefix,c,arrays,'verification',0,params)
        c['deployed_versions']=[dict(id='H001',version=1,weight=.5,effective_mixture_weight=.5)]
        with self.assertRaises(AssertionError):a.selection_audit(pool,prefix,c,arrays,'verification',0,params)


if __name__=='__main__':unittest.main()
