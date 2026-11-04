from copy import deepcopy
import unittest
from audit_v12_llm_repeats import validate_response_ids,validate_shared_prefixes


class AuditTests(unittest.TestCase):
    def test_identical_output_with_distinct_response_ids_is_allowed(self):
        rows=[dict(response_id='new1',raw_sha256='same'),dict(response_id='new2',raw_sha256='same')]
        self.assertEqual(validate_response_ids(rows,{'old1'}),2)

    def test_cross_replica_and_historical_response_reuse_are_rejected(self):
        with self.assertRaises(AssertionError):validate_response_ids([dict(response_id='x'),dict(response_id='x')],set())
        with self.assertRaises(AssertionError):validate_response_ids([dict(response_id='x')],{'x'})

    def test_same_history_different_GP_is_not_attributed_to_LLM(self):
        row=dict(task='case',prefix_sha256='history',GP_sha256='GP1',ordered_ids=['a','b'],display_ids=['b','a'])
        self.assertEqual(validate_shared_prefixes([row,deepcopy(row)]),1)
        changed=dict(row,GP_sha256='GP2')
        with self.assertRaises(AssertionError):validate_shared_prefixes([row,changed])


if __name__=='__main__':unittest.main()
