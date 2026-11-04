"""Reject re-sealed wrong BO choices and falsified feedback without oracle calls."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest
import audit_v12_initialization_seeds as audit


class BOAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=audit.ROOT;cls.task='Reizman_Suzuki_Case_2'
        cls.pool=audit.study.NATIVE_LOAD_TASK(cls.root,cls.task,seed=0)
        cls.protocol=audit.study.v9.check(cls.root)
        cls.run_dir=audit.study.v9.directory(cls.root)/'runs'/f'{cls.task}_bo'

    def test_native_archived_BO_is_accepted(self):
        self.assertEqual(audit.audit_bo(self.run_dir,self.pool,self.protocol)['rounds'],16)

    def test_consistently_resealed_wrong_choice_is_rejected(self):
        native=audit.study.old.read;sf=self.run_dir/'state.json';cf=self.run_dir/'rounds/01_commit.json'
        state=native(sf);c=native(cf);key=c['ordered_ids'][1]
        c['selected_id']=key;c['BO_reference_id']=key;c['commit_sha256']=audit.digest({k:v for k,v in c.items() if k!='commit_sha256'})
        state['history'][4]['candidate_id']=key;state['decisions'][0].update(selected_id=key,BO_reference_id=key,commit_sha256=c['commit_sha256'])
        def read(path):return deepcopy(state if Path(path)==sf else c) if Path(path) in (sf,cf) else native(path)
        with patch.object(audit.study.old,'read',side_effect=read):
            with self.assertRaisesRegex(AssertionError,'independent acquisition argmax'):audit.audit_bo(self.run_dir,self.pool,self.protocol)

    def test_falsified_observed_feedback_is_rejected(self):
        native=audit.study.old.read;sf=self.run_dir/'state.json';state=native(sf);state['decisions'][0]['raw_value']+=1
        with patch.object(audit.study.old,'read',side_effect=lambda path:deepcopy(state) if Path(path)==sf else native(path)):
            with self.assertRaisesRegex(AssertionError,'entire feedback record'):audit.audit_bo(self.run_dir,self.pool,self.protocol)


if __name__=='__main__':unittest.main()
