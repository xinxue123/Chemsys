"""Secret-safe, independently audited OpenAI Responses transport for R7.

Provider migration only: the R6 qualitative prompt and validation contract,
R7 posterior, GP, gate and study budget remain unchanged. No hidden reasoning
content or API key is ever written to an experiment file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import httpx

from . import continuous_benchmark_data as data
from . import continuous_r6_engine as r6
from . import continuous_r6_llm as original


MODEL = 'gpt-6.1-terra'
REASONING_EFFORT = 'medium'
ENDPOINT = 'https://api.openai.com/v1'
SYSTEM = original.SYSTEM
MAX_OUTPUT_TOKENS = 8192
MAX_TRANSPORT_ATTEMPTS = 8
MAX_VALIDATION_RESPONSES = 3


@dataclass(frozen=True)
class APIConfig:
    api_key: str = field(repr=False)
    base_url: str = ENDPOINT

    def public(self):
        return {'base_url': self.base_url, 'model': MODEL,
                'reasoning_effort': REASONING_EFFORT,
                'credential_source': 'external user-supplied key file; never copied'}


def load_config(path):
    """The supplied key file is a raw secret, never a model instruction."""
    secret = Path(path).read_text(encoding='utf-8-sig').strip()
    if not secret.startswith('sk-') or any(ch.isspace() for ch in secret):
        raise ValueError('expected a raw OpenAI API key in the configured local file')
    return APIConfig(api_key=secret)


def safe_error(exc, config):
    if isinstance(exc, httpx.HTTPStatusError):
        return {'kind': 'HTTPStatusError', 'status': exc.response.status_code}
    return {'kind': type(exc).__name__,
            'message': str(exc).replace(config.api_key, '[REDACTED]')[:180]}


class APIClient:
    def __init__(self, config):
        self.config = config
        self.http = httpx.Client(
            base_url=config.base_url,
            headers={'Authorization': 'Bearer ' + config.api_key},
            timeout=httpx.Timeout(180., connect=20.),
            follow_redirects=False,
        )

    def close(self):
        self.http.close()

    def complete(self, messages):
        payload = {
            'model': MODEL,
            'reasoning': {'effort': REASONING_EFFORT},
            'max_output_tokens': MAX_OUTPUT_TOKENS,
            'text': {'format': {'type': 'json_object'}},
            'input': messages,
            'store': False,
        }
        t0 = time.perf_counter()
        response = self.http.post('/responses', json=payload)
        response.raise_for_status()
        body = response.json()
        if body.get('model') != MODEL:
            raise ValueError('actual response model differs from precommitted provider model')
        visible_parts = []
        for item in body.get('output', []):
            if item.get('type') != 'message':
                continue
            for content in item.get('content', []):
                if content.get('type') == 'output_text':
                    visible_parts.append(content.get('text', ''))
        content = '\n'.join(visible_parts)
        # No raw output array, reasoning, or headers are retained.
        return {
            'id': body.get('id'), 'created': body.get('created_at'),
            'model': body.get('model'), 'status': body.get('status'),
            'content': content, 'usage': body.get('usage', {}),
            'wall_seconds': time.perf_counter() - t0,
            'retained_raw_reasoning': False,
        }


def request(client, cache, state, t, context, protocol_sha256, root):
    """Share the same first-round request across B2/B4, then adapt online."""
    from filelock import FileLock

    public = r6.public_menu(cache, dict(state, arm='A4'), t, context)
    request_context = {
        'public': public,
        'observed_prefix': state['history'],
        'task': cache.pool['task'], 'phase': 'R7_OpenAI_qualitative_pairwise',
        'arm': 'A4', 'round': t,
        'protocol_sha256': protocol_sha256,
        'model': MODEL, 'reasoning_effort': REASONING_EFFORT,
        'system_sha256': r6.digest(SYSTEM),
    }
    request_context['context_sha256'] = r6.digest(request_context)
    directory = Path(root) / 'requests' / request_context['context_sha256']
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / '.request.lock'), timeout=900):
        context_file = directory / 'context.json'
        if context_file.exists():
            assert data.read(context_file) == request_context
        else:
            data.write(context_file, request_context)
        valid_file = directory / 'validated.json'
        if valid_file.exists():
            saved = data.read(valid_file)
            assert saved['context_sha256'] == request_context['context_sha256']
            assert saved['plan'] == original.validate(saved['raw'], public['candidate_menu'])
            return saved['plan'], _receipt(saved)

        base = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps(public, ensure_ascii=False,
                                                    allow_nan=False, separators=(',', ':'))}]
        messages = base
        received = 0
        for attempt in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
            attempt_dir = directory / f'attempt{attempt:02d}'
            body = {'model': MODEL, 'reasoning': {'effort': REASONING_EFFORT},
                    'max_output_tokens': MAX_OUTPUT_TOKENS,
                    'text': {'format': {'type': 'json_object'}},
                    'input': messages, 'store': False}
            request_file = attempt_dir / 'request.json'
            if request_file.exists():
                assert data.read(request_file) == body
            else:
                data.write(request_file, body)
            response_file = attempt_dir / 'response.json'
            transport_error = attempt_dir / 'transport_error.json'
            if transport_error.exists():
                if not data.read(transport_error)['retryable']:
                    raise RuntimeError('nonretryable OpenAI API request failure')
                continue
            if response_file.exists():
                response = data.read(response_file)
            else:
                try:
                    response = client.complete(messages)
                    data.write(response_file, response)
                except Exception as exc:
                    code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                    retry = isinstance(exc, (httpx.TimeoutException, httpx.TransportError)) or code in (429, 500, 502, 503, 504)
                    data.write(transport_error, {
                        'error': safe_error(exc, client.config), 'retryable': bool(retry)})
                    if not retry:
                        raise RuntimeError('nonretryable OpenAI API request failure') from None
                    if attempt < MAX_TRANSPORT_ATTEMPTS:
                        time.sleep(min(5 * 2 ** (attempt-1), 30))
                    continue
            received += 1
            try:
                if response['model'] != MODEL or response['status'] != 'completed':
                    raise ValueError('OpenAI response model mismatch or incomplete status')
                raw = json.loads(response['content'])
                plan = original.validate(raw, public['candidate_menu'])
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                error = type(exc).__name__ + ': ' + str(exc)[:200]
                data.write(attempt_dir / 'validation_error.json', {'error': error})
                if received >= MAX_VALIDATION_RESPONSES:
                    raise RuntimeError('OpenAI response validation limit exhausted') from None
                messages = base + [
                    {'role': 'assistant', 'content': response.get('content') or '{}'},
                    {'role': 'user', 'content': 'Validation failed: ' + error +
                     '. Correct only the JSON using the SAME observations; no experiment occurred.'},
                ]
                continue
            saved = {'raw': raw, 'plan': plan, 'accepted_attempt': attempt,
                     'received_responses': received,
                     'context_sha256': request_context['context_sha256'],
                     'response_sha256': data.filehash(response_file),
                     'response_model': response['model'], 'response_id': response['id']}
            data.write(valid_file, saved)
            return plan, _receipt(saved)
    raise RuntimeError('OpenAI transport attempt limit exhausted')


def _receipt(saved):
    return {k: saved[k] for k in ('context_sha256', 'response_sha256',
                                 'accepted_attempt', 'received_responses',
                                 'response_model', 'response_id')}
