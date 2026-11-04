"""Offline-only transport-contract tests; no account credentials or network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from chem.benchmarks import knowledge_openai_client as api

SECRET = 'sk-unit-test-not-an-account-key'


def completed():
    return {'id': 'resp_test', 'model': api.MODEL, 'status': 'completed',
            'output': [{'type': 'reasoning', 'summary': [{'text': 'PRIVATE_SENTINEL'}]},
                       {'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': '{"ok":true}'}]}],
            'usage': {'input_tokens': 5, 'output_tokens': 8, 'total_tokens': 13,
                      'output_tokens_details': {'reasoning_tokens': 3, 'text': 'PRIVATE_SENTINEL'},
                      'unapproved': 'PRIVATE_SENTINEL'}}


class KnowledgeOpenAIClientTests(unittest.TestCase):
    def client(self, handler):
        client = api.APIClient(api.APIConfig(SECRET), transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return client

    def test_fixed_request_and_public_only_response(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=completed())
        client = self.client(handler)
        result = client.complete('Return JSON.', {'observation': 1}, json_output=True)
        request = requests[0]
        body = json.loads(request.content)
        self.assertEqual(str(request.url), 'https://api.openai.com/v1/responses')
        self.assertEqual(body['model'], 'gpt-6.1-terra')
        self.assertEqual(body['reasoning'], {'effort': 'medium'})
        self.assertIs(body['store'], False)
        self.assertEqual(body['max_output_tokens'], 4096)
        self.assertEqual(body['text']['format']['type'], 'json_object')
        self.assertEqual([m['role'] for m in body['input']], ['system', 'user'])
        self.assertNotIn('temperature', body)
        self.assertNotIn('top_p', body)
        self.assertEqual(set(result), {'output_text', 'id', 'model', 'status', 'usage',
                                      'requested_model', 'resolved_model'})
        self.assertEqual(json.loads(result['output_text']), {'ok': True})
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(result))

    def test_explicit_dated_resolution_is_reported_without_request_substitution(self):
        requests = []
        raw = completed()
        raw['model'] = api.MODEL + '-2026-10-29'
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=raw)
        result = self.client(handler).complete('Return JSON.', 'test', json_output=True)
        self.assertEqual(requests[0]['model'], api.MODEL)
        self.assertEqual(result['requested_model'], api.MODEL)
        self.assertEqual(result['resolved_model'], raw['model'])
        self.assertFalse(api._requested_family(api.MODEL + '-latest'))
        self.assertFalse(api._requested_family(api.MODEL + '-2026-99-99'))

    def test_wrong_model_and_incomplete_are_rejected_without_body(self):
        for patch in [{'model': 'another-model-' + SECRET},
                      {'status': 'incomplete', 'incomplete_details': {'reason': SECRET}},
                      {'status': 'failed', 'error': {'message': SECRET}}]:
            raw = completed()
            raw.update(patch)
            with self.subTest(patch=list(patch)):
                client = self.client(lambda request, data=raw: httpx.Response(200, json=data))
                with self.assertRaises(api.OpenAIClientError) as caught:
                    client.complete('Return JSON.', 'test')
                self.assertNotIn(SECRET, str(caught.exception))
                self.assertIsNone(caught.exception.__cause__)

    def test_refusal_and_missing_public_text(self):
        for content in [[{'type': 'refusal', 'refusal': SECRET}], []]:
            raw = completed()
            raw['output'][1]['content'] = content
            client = self.client(lambda request, data=raw: httpx.Response(200, json=data))
            with self.assertRaises(api.OpenAIClientError) as caught:
                client.complete('Return JSON.', 'test')
            self.assertNotIn(SECRET, str(caught.exception))

    def test_secret_not_in_repr_or_http_errors(self):
        client = self.client(lambda request: httpx.Response(401, text=SECRET))
        self.assertNotIn(SECRET, repr(client))
        self.assertNotIn(SECRET, repr(client.config))
        self.assertNotIn(SECRET, json.dumps(client.config.public()))
        with self.assertRaises(api.OpenAIClientError) as caught:
            client.complete('Return JSON.', 'test')
        self.assertEqual(str(caught.exception), 'OpenAI HTTP request failed (status=401)')

    def test_transport_error_is_sanitized(self):
        def handler(request):
            raise httpx.ReadTimeout(SECRET, request=request)
        client = self.client(handler)
        with self.assertRaises(api.OpenAIClientError) as caught:
            client.complete('Return JSON.', 'test')
        self.assertNotIn(SECRET, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_accidental_key_echo_is_redacted(self):
        raw = completed()
        raw['id'] = SECRET
        raw['output'][1]['content'][0]['text'] = json.dumps({'echo': SECRET})
        client = self.client(lambda request: httpx.Response(200, json=raw))
        result = client.complete('Return JSON.', 'test', json_output=True)
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertIn('[REDACTED]', result['output_text'])

    def test_config_bare_and_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test-key.txt'
            path.write_text(SECRET + '\n', encoding='utf-8')
            self.assertEqual(api.load_config(path).base_url, api.DEFAULT_BASE_URL)
            path.write_text(json.dumps({'api_key': SECRET, 'base_url': 'https://example.com'}), encoding='utf-8')
            self.assertEqual(api.load_config(path).base_url, 'https://example.com/v1')
            path.write_text('{"api_key":' + SECRET, encoding='utf-8')
            with self.assertRaises(ValueError) as caught:
                api.load_config(path)
            self.assertNotIn(SECRET, str(caught.exception))

    def test_credential_urls_rejected(self):
        for url in ['http://example.com/v1', 'https://user:pass@example.com/v1',
                    'https://example.com/v1?key=' + SECRET, 'https://example.com/' + SECRET + '/v1']:
            with self.subTest(url_kind=url.split(':')[0]):
                with self.assertRaises(ValueError) as caught:
                    api.APIConfig(SECRET, url)
                self.assertNotIn(SECRET, str(caught.exception))

    def test_model_listing_and_preflight(self):
        paths = []
        def handler(request):
            paths.append(request.url.path)
            if request.method == 'GET':
                return httpx.Response(200, json={'data': [{'id': api.MODEL, 'secret': SECRET}]})
            return httpx.Response(200, json=completed())
        result = api.preflight(self.client(handler))
        self.assertEqual(paths, ['/v1/models', '/v1/responses'])
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['reasoning_effort'], 'medium')
        self.assertNotIn(SECRET, json.dumps(result))

    def test_absent_model_never_falls_back_or_posts(self):
        methods = []
        def handler(request):
            methods.append(request.method)
            return httpx.Response(200, json={'data': [{'id': 'another-model'}]})
        with self.assertRaises(api.OpenAIClientError):
            api.preflight(self.client(handler))
        self.assertEqual(methods, ['GET'])

    def test_json_output_must_be_an_object(self):
        for text in ['[]', 'not JSON', 'null']:
            raw = completed()
            raw['output'][1]['content'][0]['text'] = text
            client = self.client(lambda request, data=raw: httpx.Response(200, json=data))
            with self.assertRaises(api.OpenAIClientError):
                client.complete('Return JSON.', 'test', json_output=True)


if __name__ == '__main__':
    unittest.main()
