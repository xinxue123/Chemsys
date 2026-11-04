"""Independent V7 support/stage audit tamper tests; synthetic and no APIs."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import audit_knowledge_actionable_loop as audit


def fixture():
    ids = ['plus','outside','z_ref','a_ref','mid','low','lower','last','invalid_high','same_high']
    features = [[.8,.5,'L','B','S'],[.4,.8,'L','B','S'],
        [.2,.5,'L','B','S'],[.6,.5,'L','B','S'],[.7,.4,'L','B','S'],
        [.8,.5,'Q','B','S'],[.8,.5,'L','Q','S'],[.8,.5,'L','B','Q'],
        [.2,.1,'Q','C','T'],[.8,.5,'L','B','S']]
    pool = dict(ids=ids,ix={key:i for i,key in enumerate(ids)},
        axes=['temperature','time','ligand','base','solvent'],
        kinds=['continuous','continuous','categorical','categorical','categorical'],features=features)
    prefix = [dict(candidate_id=k,y=y) for k,y in zip(ids[2:],[5.,5.,4.,3.,2.,1.,100.,99.])]
    arrays = dict(base_logei=np.array([0.,-.2]+[-10.]*8),eligible=np.array([True,True]+[False]*8),
                  tie_priority=np.arange(10))
    menu = ['outside','plus']
    return pool,prefix,menu,arrays


def support(pool,prefix,menu,arrays,t=2,quota=0):
    return audit.expected_proposal_support(pool,prefix,menu,arrays,t,quota,1.)


def stage_rows(wins=5,weighted=1,verification=False):
    rows=[]
    for j in range(8):
        task=f'T{j}'
        for mode in ('bo','full'):
            rows.append(dict(task=task,mode=mode,status='COMPLETE',
                SearchAUC=1.+int(mode=='full' and j<wins),
                selection_modes={'knowledge_guided':weighted if mode=='full' and j==0 else 0}))
        if verification:
            rows.append(dict(task=task,mode='verification',status='COMPLETE',SearchAUC=1.,selection_modes={}))
    return rows,[f'T{j}' for j in range(8)]


class ActionableAuditTests(unittest.TestCase):
    def test_compact_focus_and_top_four_use_all_legal_own_references(self):
        p,h,m,a=fixture();s=support(p,h,m,a)
        self.assertEqual([x['point_id'] for x in s['candidate_support']],m)
        row=s['candidate_support'][1]
        self.assertEqual(row['compatible_reference_count'],6)
        self.assertTrue(row['references_truncated'])
        self.assertEqual([x['reference_id'] for x in row['references']],['z_ref','a_ref','mid','low'])
        self.assertEqual(row['references'][0]['required_focus_axes'],['temperature'])
        self.assertEqual(row['references'][1]['required_focus_axes'],[])
        self.assertEqual(row['references'][1]['additional_changed_axes'],['temperature'])
        self.assertEqual(row['references'][2]['additional_changed_axes'],['temperature','time'])
        self.assertEqual(row['references'][3]['required_focus_axes'],['ligand'])
        audit.support_payload_audit(s,p,h,m,a,2,0,1.)

    def test_missing_reference_false_count_or_truncation_is_detected(self):
        p,h,m,a=fixture();s=support(p,h,m,a)
        for fault in ('missing','count','truncation','wrong_ref'):
            bad=deepcopy(s);row=bad['candidate_support'][1]
            if fault=='missing':row['references'].pop()
            elif fault=='count':row['compatible_reference_count']=4
            elif fault=='truncation':row['references_truncated']=False
            else:row['references'][0]['reference_id']='invalid_high'
            with self.subTest(fault=fault),self.assertRaises(AssertionError):
                audit.support_payload_audit(bad,p,h,m,a,2,0,1.)

    def test_focus_codec_and_extra_quality_fields_are_rejected(self):
        p,h,m,a=fixture();s=support(p,h,m,a)
        for fault in ('remove_required','invent_axis','ordinal','gp_mean'):
            bad=deepcopy(s);ref=bad['candidate_support'][1]['references'][0]
            if fault=='remove_required':ref['required_focus_axes']=[]
            elif fault=='invent_axis':ref['additional_changed_axes'].append('unknown')
            elif fault=='ordinal':ref['direction']='higher_category_code'
            else:bad['candidate_support'][1]['GP_mean']=.9
            with self.subTest(fault=fault),self.assertRaises(AssertionError):
                audit.support_payload_audit(bad,p,h,m,a,2,0,1.)

    def test_known_value_sort_physical_tie_and_menu_order_are_checked(self):
        p,h,m,a=fixture();s=support(p,h,m,a)
        bad=deepcopy(s);bad['candidate_support'][1]['references'][:2]=list(reversed(bad['candidate_support'][1]['references'][:2]))
        with self.assertRaises(AssertionError):audit.support_payload_audit(bad,p,h,m,a,2,0,1.)
        changed=deepcopy(h);changed[5]['y']=20.
        with self.assertRaises(AssertionError):audit.support_payload_audit(s,p,changed,m,a,2,0,1.)
        with self.assertRaises(AssertionError):audit.support_payload_audit(s,p,h,list(reversed(m)),a,2,0,1.)

    def test_BO_flag_true_and_noneligible_or_over_gap_false(self):
        p,h,m,a=fixture();s=support(p,h,m,a)
        self.assertFalse(s['candidate_support'][0]['passes_narrow_cost_gate'])
        self.assertTrue(s['candidate_support'][1]['passes_narrow_cost_gate'])
        for index in (0,1):
            bad=deepcopy(s);bad['candidate_support'][index]['passes_narrow_cost_gate']=not bad['candidate_support'][index]['passes_narrow_cost_gate']
            with self.assertRaises(AssertionError):audit.support_payload_audit(bad,p,h,m,a,2,0,1.)
        a['base_logei'][1]=-.01;a['eligible'][1]=False
        self.assertFalse(support(p,h,m,a)['candidate_support'][0]['passes_narrow_cost_gate'])

    def test_window_is_prefix_quota_not_final_quota_and_does_not_hide_cost(self):
        p,h,m,a=fixture();early=support(p,h,m,a,t=2,quota=0)
        self.assertTrue(early['optional_window_open'])
        with self.assertRaises(AssertionError):audit.support_payload_audit(early,p,h,m,a,2,1,1.)
        for t,quota in ((11,0),(2,1)):
            closed=support(p,h,m,a,t=t,quota=quota)
            self.assertFalse(closed['optional_window_open'])
            self.assertEqual(closed['candidate_support'],early['candidate_support'])

    def test_radius_boundary_and_real_labels(self):
        p=dict(ids=['x','r'],ix={'x':0,'r':1},axes=['temperature','label'],
               kinds=['continuous','categorical'],features=[[.5,'real'],[.25,'real']])
        h=[dict(candidate_id='r',y=.4)];a=dict(base_logei=np.array([0.,-1.]),eligible=np.array([True,False]),tie_priority=np.arange(2))
        row=support(p,h,['x'],a)['candidate_support'][0]['references'][0]
        self.assertEqual(row['required_focus_axes'],[])
        p['features'][1][0]=.25-2e-12
        self.assertEqual(support(p,h,['x'],a)['candidate_support'][0]['references'][0]['required_focus_axes'],['temperature'])
        p['features'][1][1]=2
        with self.assertRaises(AssertionError):support(p,h,['x'],a)

    def test_failed_stage_one_is_predeclared_stop_not_missing_stage_two(self):
        rows,tasks=stage_rows(wins=4)
        result=audit.stage_scope(rows,tasks)
        self.assertEqual(result['stage_status'],'PREDECLARED_STOP_AFTER_STAGE1')
        self.assertEqual(result['stage2_missing_tasks'],[])
        self.assertFalse(result['stage2_trigger'])
        rows.append(dict(task='T0',mode='verification',status='RUNNING'))
        self.assertTrue(audit.stage_scope(rows,tasks)['stage2_policy_violation'])

    def test_stage_two_requires_five_wins_and_actual_weighted_choice(self):
        rows,tasks=stage_rows(wins=5,weighted=0)
        self.assertFalse(audit.stage_scope(rows,tasks)['stage2_trigger'])
        rows,tasks=stage_rows(wins=5,weighted=1)
        self.assertEqual(audit.stage_scope(rows,tasks)['stage_status'],'STAGE2_REQUIRED_NOT_FULLY_AUDITED')
        rows,tasks=stage_rows(wins=5,weighted=1,verification=True)
        self.assertEqual(audit.stage_scope(rows,tasks)['stage_status'],'STAGE2_COMPLETE')

    def test_subset_audit_cannot_decide_whole_stage(self):
        rows,tasks=stage_rows()
        self.assertIsNone(audit.stage_scope(rows,tasks,filtered=True)['stage2_trigger'])
        self.assertEqual(audit.stage_scope(rows[:2],tasks)['stage_status'],'STAGE1_NOT_FULLY_AUDITED')


if __name__=='__main__':unittest.main()
