"""Input-only comparability hints; no GP, oracle, model or filesystem calls."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem.benchmarks import knowledge_actionability as hints
from chem.benchmarks import knowledge_transfer_policy as policy


def pool(axes,kinds,features,ids=None):
    ids=list(ids or [f'p{k}' for k in range(len(features))])
    return SimpleNamespace(axes=tuple(axes),kinds=tuple(kinds),features=tuple(map(tuple,features)),
        ids=ids,ix={key:j for j,key in enumerate(ids)})


def past(*pairs):
    return [dict(candidate_id=key,y=value,raw_value=value) for key,value in pairs]


def category_pool():
    return pool(['a','b','c','d'],['categorical']*4,
        [('A','A','A','A'),('B','B','A','A'),('B','B','B','A'),('B','A','A','A'),
         ('A','B','A','A'),('A','A','B','A'),('A','A','A','B'),('A','A','A','A')],
        ['plus','low','illegal_high','z_ref','a_ref','mid','lower','same_high'])


class ActionabilityTests(unittest.TestCase):
    def test_compressed_hint_represents_all_and_only_legal_combinations(self):
        p=pool(['temperature','time','ligand','base'],['continuous','continuous','categorical','categorical'],
            [(.8,.5,'L+','B+'),(.2,.4,'L-','B+'),(.7,.4,'L+','B+'),(.2,0.,'L-','B-')])
        for ref in p.ids[1:]:
            expected=[]
            for size in (1,2):
                for names in combinations(p.axes,size):
                    focus={}
                    for axis in names:
                        j=p.axes.index(axis);x,r=p.features[0][j],p.features[p.ix[ref]][j]
                        focus[axis]=({'direction':'increase' if x>r else 'decrease','bounds':[0.,1.]}
                            if p.kinds[j]=='continuous' else {'direction':'prefer_plus','allowed':list(dict.fromkeys([x,r]))})
                    action=dict(op='add',id=None,plus_id='p0',minus_id=ref,focus=focus,background={},reason='mechanistic_prior: synthetic comparison')
                    try:
                        policy._build_contract(p,action,{'p0'},{ref:.5})
                    except (ValueError,TypeError):
                        continue
                    expected.append(list(names))
            actual=hints.legal_focus_combinations(p,'p0',ref)
            self.assertEqual(actual,expected)
            row=hints.candidate_support(p,past((ref,.5)),['p0'])[0]
            if not actual:
                self.assertEqual(row['references'],[])
                continue
            hint=row['references'][0];required=set(hint['required_focus_axes'])
            changed=required|set(hint['additional_changed_axes'])
            decoded=[list(c) for size in (1,2) for c in combinations(p.axes,size)
                if required<=set(c)<=changed]
            self.assertEqual(decoded,actual)

    def test_all_small_continuous_changes_allow_all_one_and_two_axis_sets(self):
        p=pool(['x','y','z'],['continuous']*3,[(.2,.2,.2),(.1,.1,.1)])
        self.assertEqual(hints.legal_focus_combinations(p,'p0','p1'),
            [['x'],['y'],['z'],['x','y'],['x','z'],['y','z']])
        row=hints.candidate_support(p,past(('p1',.5)),['p0'])[0]['references'][0]
        self.assertEqual(row['required_focus_axes'],[])
        self.assertEqual(row['additional_changed_axes'],['x','y','z'])

    def test_background_radius_uses_existing_tolerance_and_focus_requires_change(self):
        p=pool(['x','y'],['continuous']*2,[(.5,.5),(.25,.5),(.25-5e-13,.5),(.25-2e-12,.5),(.5,.5)])
        support=hints.candidate_support(p,past(('p1',1.),('p2',.9),('p3',.8),('p4',.7)),['p0'])[0]
        refs={r['reference_id']:r for r in support['references']}
        self.assertEqual(refs['p1']['required_focus_axes'],[])
        self.assertEqual(refs['p2']['required_focus_axes'],[])
        self.assertEqual(refs['p3']['required_focus_axes'],['x'])
        self.assertNotIn('p4',refs)
        self.assertEqual(hints.legal_focus_combinations(p,'p0','p1'),[['x']])
        self.assertEqual(hints.legal_focus_combinations(p,'p0','p4'),[])

    def test_categories_are_nominal_and_three_required_axes_are_not_repaired(self):
        p=category_pool()
        self.assertEqual(hints.legal_focus_combinations(p,'plus','low'),[['a','b']])
        self.assertEqual(hints.legal_focus_combinations(p,'plus','illegal_high'),[])
        row=hints.candidate_support(p,past(('low',.5)),['plus'])[0]['references'][0]
        self.assertEqual(row['required_focus_axes'],['a','b'])
        self.assertEqual(row['additional_changed_axes'],[])
        self.assertNotIn('direction',row)
        self.assertNotIn('preferred_label',row)

    def test_highest_four_legal_references_not_four_highest_raw_observations(self):
        p=category_pool();history=past(('low',2.),('illegal_high',100.),('z_ref',5.),('a_ref',5.),('mid',4.),('lower',3.),('same_high',99.))
        result=hints.candidate_support(p,history,['plus'])[0]
        self.assertEqual([r['reference_id'] for r in result['references']],['a_ref','z_ref','mid','lower'])
        self.assertEqual(result['compatible_reference_count'],5)
        self.assertTrue(result['references_truncated'])
        all_refs=hints.candidate_support(p,history,['plus'],maximum_references=8)[0]
        self.assertEqual(len(all_refs['references']),5)
        self.assertFalse(all_refs['references_truncated'])

    def test_ID_renaming_and_history_order_cannot_change_physical_selection(self):
        p=category_pool();history=past(('low',2.),('illegal_high',100.),('z_ref',5.),('a_ref',5.),('mid',4.),('lower',3.),('same_high',99.))
        before=hints.candidate_support(p,history,['plus'])
        mapping={key:f'new_{len(p.ids)-j:03d}' for j,key in enumerate(p.ids)}
        renamed=pool(p.axes,p.kinds,p.features,[mapping[k] for k in p.ids])
        after=hints.candidate_support(renamed,[dict(r,candidate_id=mapping[r['candidate_id']]) for r in reversed(history)],[mapping['plus']])
        after[0]['point_id']='plus';reverse={v:k for k,v in mapping.items()}
        for r in after[0]['references']:r['reference_id']=reverse[r['reference_id']]
        self.assertEqual(after,before)

    def test_menu_order_empty_support_and_input_state_are_preserved(self):
        p=category_pool();history=past(('plus',.5));menu=['illegal_high','low']
        snapshot=deepcopy((p.__dict__,history,menu))
        result=hints.candidate_support(p,history,menu)
        self.assertEqual([r['point_id'] for r in result],menu)
        self.assertEqual(result[0]['compatible_reference_count'],0)
        result[1]['references'][0]['required_focus_axes'].append('mutated_output')
        self.assertEqual((p.__dict__,history,menu),snapshot)

    def test_helper_has_no_GP_or_oracle_or_file_access(self):
        class InputOnly:
            def __init__(self,p):self.__dict__.update(p.__dict__)
            @property
            def oracle(self):raise AssertionError('oracle accessed')
            @property
            def gp(self):raise AssertionError('GP accessed')
            @property
            def targets(self):raise AssertionError('unqueried targets accessed')
        p=InputOnly(category_pool())
        with patch('builtins.open',side_effect=AssertionError('file opened')),patch.object(Path,'open',side_effect=AssertionError('file opened')):
            result=hints.candidate_support(p,past(('low',.5)),['plus'])
        self.assertEqual(result[0]['compatible_reference_count'],1)
        with self.assertRaises(TypeError):
            hints.candidate_support(p,[],['plus'],context={'mu':1.})

    def test_rejects_queried_duplicate_unknown_or_nonfinite_inputs(self):
        p=category_pool()
        for history,menu in [(past(('plus',.5)),['plus']),([],['plus','plus']),([],['unknown']),
            (past(('low',.5),('low',.4)),['plus']),(past(('low',float('nan'))),['plus'])]:
            with self.subTest(history=history,menu=menu),self.assertRaises(ValueError):
                hints.candidate_support(p,history,menu)
        for count in (0,-1,True,1.5):
            with self.assertRaises(ValueError):hints.candidate_support(p,[],['plus'],maximum_references=count)
        bad=pool(['category'],['categorical'],[(1,),(2,)])
        with self.assertRaises(ValueError):hints.candidate_support(bad,past(('p1',.5)),['p0'])


if __name__=='__main__':unittest.main()
