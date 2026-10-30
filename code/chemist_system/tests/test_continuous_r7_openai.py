"""Offline checks for the isolated GPT-6.1 Terra/medium R7 transport."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx

from chem.benchmarks import continuous_r7_openai_llm as llm
from chem.benchmarks import continuous_r7_openai_study as study
from chem.benchmarks import continuous_r7_study as frozen


class ProviderContractTests(unittest.TestCase):
    def test_provider_and_reasoning_are_frozen(self):
        self.assertEqual(llm.MODEL, 'gpt-6.1-terra')
        self.assertEqual(llm.REASONING_EFFORT, 'medium')
        self.assertEqual(llm.ENDPOINT, 'https://api.openai.com/v1')

    def test_key_is_external_and_not_in_repr(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'temporary-key.txt'
            secret = 'sk-unit-test-only-not-a-real-key'
            path.write_text(secret + '\n', encoding='utf-8')
            config = llm.load_config(path)
            self.assertEqual(config.api_key, secret)
            self.assertNotIn(secret, repr(config))
            self.assertNotIn(secret, json.dumps(config.public()))

    def test_transport_saves_only_visible_output(self):
        secret = 'sk-unit-test-only-not-a-real-key'
        config = llm.APIConfig(secret)
        captured = []

        def handler(request):
            captured.append(json.loads(request.content))
            self.assertEqual(request.headers['Authorization'], 'Bearer ' + secret)
            return httpx.Response(200, json={
                'id': 'resp_unit_test', 'model': llm.MODEL,
                'status': 'completed',
                'output': [
                    {'type': 'reasoning', 'summary': [{'text': 'private should not survive'}]},
                    {'type': 'message', 'content': [
                        {'type': 'output_text', 'text': '{"decision":"abstain"}'},
                    ]},
                ],
                'usage': {'input_tokens': 4, 'output_tokens': 5},
            })

        client = llm.APIClient(config)
        client.http.close()
        client.http = httpx.Client(transport=httpx.MockTransport(handler),
                                   base_url=llm.ENDPOINT,
                                   headers={'Authorization': 'Bearer ' + secret})
        try:
            result = client.complete([{'role': 'user', 'content': 'test'}])
        finally:
            client.close()
        self.assertEqual(result['content'], '{"decision":"abstain"}')
        self.assertNotIn('private should not survive', json.dumps(result))
        self.assertNotIn(secret, json.dumps(result))
        self.assertFalse(result['retained_raw_reasoning'])
        self.assertEqual(captured[0]['reasoning']['effort'], 'medium')
        self.assertEqual(captured[0]['model'], llm.MODEL)
        self.assertIs(captured[0]['store'], False)

    def test_mismatched_model_is_rejected(self):
        client = llm.APIClient(llm.APIConfig('sk-unit-test-only'))
        client.http.close()
        client.http = httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(
                200, json={'model': 'not-terra', 'output': [], 'status': 'completed'})),
            base_url=llm.ENDPOINT)
        try:
            with self.assertRaisesRegex(ValueError, 'model differs'):
                client.complete([{'role': 'user', 'content': 'test'}])
        finally:
            client.close()

    def test_isolated_runtime_restores_existing_study(self):
        old_root, old_out, old_runtime = frozen.ROOT, frozen.OUT, frozen._runtime
        with study._isolated_runtime():
            self.assertEqual(frozen.ROOT, study.ROOT)
            self.assertEqual(frozen.OUT, study.OUT)
        self.assertIs(frozen._runtime, old_runtime)
        self.assertEqual(frozen.ROOT, old_root)
        self.assertEqual(frozen.OUT, old_out)


if __name__ == '__main__':
    unittest.main()
