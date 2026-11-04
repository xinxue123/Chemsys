"""Regression guards for incomplete corpora and underqualified precision."""
from copy import deepcopy
from pathlib import Path
import unittest

import audit_knowledge_conditional_ei_corpus as audit


class CorpusAuditGuards(unittest.TestCase):
    def setUp(self):
        self.p={'tasks':list(audit.TASKS),'inputs':[{'task':t} for t in audit.TASKS]}
        self.p['protocol_sha256']=audit.old.digest(self.p)
        self.paths=[Path(t+'_full_agent_initial_terra')/'state.json' for t in audit.TASKS]

    def test_complete_manifest(self):
        audit.complete_run_manifest(self.p,self.paths)

    def test_missing_or_duplicated_run_cannot_pass(self):
        for paths in (self.paths[:-1],self.paths[:-1]+self.paths[:1],self.paths+self.paths[:1]):
            with self.assertRaises(AssertionError):audit.complete_run_manifest(self.p,paths)

    def test_protocol_tamper_or_missing_task_cannot_pass(self):
        changed=deepcopy(self.p);changed['tasks'].pop()
        with self.assertRaises(AssertionError):audit.complete_run_manifest(changed,self.paths)
        changed.pop('protocol_sha256');changed['protocol_sha256']=audit.old.digest(changed)
        with self.assertRaises(AssertionError):audit.complete_run_manifest(changed,self.paths)

    def test_precision_requires_both_error_estimates(self):
        audit.require_independent_precision(7e-11,4e-12,7.1e-11)
        for args in ((1e-9,1e-12,1e-11),(1e-11,2e-11,1e-11),
                     (1e-11,1e-12,2e-10),(1e-11,1e-12,1e-12)):
            with self.assertRaises(ArithmeticError):audit.require_independent_precision(*args)

    def test_nonfinite_or_negative_estimates_are_failures(self):
        for args in ((float('nan'),0.,0.),(0.,float('inf'),0.),(0.,-1.,0.)):
            with self.assertRaises(ArithmeticError):audit.require_independent_precision(*args)


if __name__=='__main__':unittest.main()
