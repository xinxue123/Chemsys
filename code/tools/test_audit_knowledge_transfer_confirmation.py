"""Seed2 audit boundary tests; all real registry reads are input/source-only."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

import audit_knowledge_transfer_confirmation as a


class ConfirmationAuditTests(unittest.TestCase):
    def protocol(self):
        return dict(variant='v5_seed2_confirm',mechanism='transfer_rule_v5',seed=2,trial_seed=2,tie_seed=2,
                    numerical_seed=0,menu_seed=0,initial=4,queries=16,supported_modes=['bo','full'],
                    supported_conditions=['agent_initial_terra'])

    def test_trial_tie2_never_changes_gp_or_menu_seed(self):
        self.assertEqual(a.trial_seeds(self.protocol()),(2,2))
        for key,value in [('seed',0),('trial_seed',1),('tie_seed',0),('numerical_seed',2),('menu_seed',2),
                          ('numerical_seed',False),('supported_modes',['bo','full','verification'])]:
            p=self.protocol();p[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):a.trial_seeds(p)

    def test_registration_hash_and_no_target_claims_enforced(self):
        value=a.read(a.ROOT/a.RULE_PATH)
        a.registered_record(value,a.RULE_SHA)
        altered=deepcopy(value);altered['trial_seed']=0
        with self.assertRaises(AssertionError):a.registered_record(altered,a.RULE_SHA)
        altered=deepcopy(value);altered['initial_y_read']=True
        altered['registration_sha256']=a.digest({k:v for k,v in altered.items() if k!='registration_sha256'})
        with self.assertRaises(AssertionError):a.registered_record(altered,altered['registration_sha256'])

    def fake_draw(self):
        features=[(.8,'Z'),(.1,'B'),(.5,'A'),(.2,'Z'),(.9,'A'),(.3,'B'),(.7,'Z'),(.4,'A')]
        ids=['z7','a3','z1','a9','p4','r6','b1','x5']
        pool=dict(ids=ids,ix={x:i for i,x in enumerate(ids)},features=features,axes=['x','category'],
                  kinds=['continuous','categorical'],points=[dict(x=x,category=c) for x,c in features])
        order=sorted(range(8),key=lambda j:features[j])
        draw=np.random.default_rng(np.random.SeedSequence([2,0])).choice(8,4,replace=False)
        js=[order[int(k)] for k in draw];chosen=[list(features[j]) for j in js]
        row=dict(drawn_sorted_positions=list(map(int,draw)),initial_ids=[ids[j] for j in js],
            initial_features=chosen,initial_points=[pool['points'][j] for j in js],initial_features_sha256=a.digest(chosen),
            sorted_pool_features_sha256=a.digest([list(features[j]) for j in order]),candidate_count=8,
            axes=pool['axes'],kinds=pool['kinds'],initial_y_read=False)
        return pool,row

    def test_draw_uses_physical_features_and_preserves_rng_order(self):
        pool,row=self.fake_draw();a.validate_input_draw(pool,row)
        renamed=deepcopy(pool);renamed['ids']=list(reversed(pool['ids']));renamed['ix']={x:i for i,x in enumerate(renamed['ids'])}
        other=deepcopy(row);other['initial_ids']=[renamed['ids'][pool['ix'][key]] for key in row['initial_ids']]
        a.validate_input_draw(renamed,other)
        other['initial_ids']=list(reversed(other['initial_ids']))
        with self.assertRaises(AssertionError):a.validate_input_draw(renamed,other)

    def test_duplicate_physical_inputs_cannot_fall_back_to_candidate_id(self):
        pool,row=self.fake_draw();pool['features'][0]=pool['features'][1]
        with self.assertRaises(AssertionError):a.validate_input_draw(pool,row)

    def test_gp_numerical_path_is_not_hidden_by_registration_adapter(self):
        before=(a.ROOT/'runtime/knowledge_loop/snapshots/v5'/a.GP_PATH).read_text(encoding='utf8')
        after=(a.ROOT/a.GP_PATH).read_text(encoding='utf8')
        self.assertEqual(a.gp_numerical_projection(before),a.gp_numerical_projection(after))
        needle='int(pool.task[-1]), seed=0, sources=()'
        self.assertIn(needle,after)
        wrong=after.replace(needle,'int(pool.task[-1]), seed=2, sources=()')
        self.assertNotEqual(a.gp_numerical_projection(before),a.gp_numerical_projection(wrong))

    def test_real_registry_and_protocol_without_oracle_cache_or_credentials(self):
        original=Path.open
        opened=[]
        def guarded(path,*args,**kwargs):
            name=path.name.lower()
            if path.suffix.lower() in ('.npz','.pt') or name in ('config.txt','openai-key.txt') or 'oracle' in name:
                raise AssertionError('forbidden registry-only read: '+str(path))
            opened.append(str(path))
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            p=a.read(a.ROOT/'experiments/runs/knowledge_loop_20261101/v5_seed2_confirm/protocol.json')
            receipt,pools,manifests=a.registration_audit(protocol=p)
        self.assertEqual(receipt['result'],'PASS_INPUT_ONLY_REGISTRATION')
        self.assertEqual((len(pools),len(manifests),receipt['initial_points']),(8,8,32))
        self.assertEqual(receipt['protected_core_files'],9)
        self.assertEqual(receipt['GP_numerical_AST'],'UNCHANGED')
        self.assertTrue(any(x.endswith('pool.json') for x in opened))
        self.assertFalse(any(x.endswith('state.json') for x in opened))


if __name__=='__main__':unittest.main()
