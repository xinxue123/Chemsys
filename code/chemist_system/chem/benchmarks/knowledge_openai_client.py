"""Secret-safe Responses transport for the fixed knowledge-loop LLM.

No request/response files are written here. Callers may persist the returned
public output and allowlisted metadata, never the raw HTTP response.
"""
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx

MODEL = 'gpt-6.1-terra'
REASONING_EFFORT = 'medium'
DEFAULT_BASE_URL = 'https://api.openai.com/v1'
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class OpenAIClientError(RuntimeError):
    """A deliberately body/header-free transport or protocol error."""


def _requested_family(model):
    """Accept an explicitly reported date snapshot, without requesting a substitute."""
    if model == MODEL:
        return True
    if not isinstance(model, str) or not model.startswith(MODEL + '-'):
        return False
    suffix = model[len(MODEL) + 1:]
    try:
        return len(suffix) == 10 and date.fromisoformat(suffix).isoformat() == suffix
    except ValueError:
        return False


@dataclass(frozen=True)
class APIConfig:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL

    def __post_init__(self):
        key = self.api_key
        if not isinstance(key, str) or not key or any(c.isspace() for c in key):
            raise ValueError('OpenAI credential must be one nonempty token')
        if not isinstance(self.base_url, str):
            raise ValueError('Invalid OpenAI base URL')
        try:
            parts = urlsplit(self.base_url)
        except ValueError:
            raise ValueError('Invalid OpenAI base URL') from None
        if (parts.scheme != 'https' or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment or key in self.base_url):
            raise ValueError('OpenAI base URL must be HTTPS without credentials, query or fragment')
        base = self.base_url.rstrip('/')
        if not parts.path or parts.path == '/':
            base += '/v1'
        if not base.endswith('/v1'):
            raise ValueError('OpenAI base URL must end at the v1 API root')
        object.__setattr__(self, 'base_url', base)

    def public(self):
        return {'base_url': self.base_url, 'model': MODEL,
                'reasoning_effort': REASONING_EFFORT, 'store': False}


def load_config(path):
    """Load a bare key or JSON {api_key, base_url}; never echo file contents."""
    try:
        raw = Path(path).read_text(encoding='utf-8-sig').strip()
    except (OSError, UnicodeError):
        raise ValueError('Cannot read OpenAI credential file') from None
    if raw.startswith('{'):
        try:
            parsed = json.loads(raw)
        except ValueError:
            raise ValueError('Invalid OpenAI JSON configuration') from None
        if not isinstance(parsed, dict) or set(parsed) - {'api_key', 'base_url'}:
            raise ValueError('OpenAI JSON configuration accepts api_key and base_url only')
        return APIConfig(api_key=parsed.get('api_key'),
                         base_url=parsed.get('base_url', DEFAULT_BASE_URL))
    return APIConfig(api_key=raw)


def _usage(raw):
    """Only token counts, including reasoning-token count but no reasoning text."""
    if not isinstance(raw, dict):
        return {}
    out = {key: raw[key] for key in ('input_tokens', 'output_tokens', 'total_tokens')
           if type(raw.get(key)) is int and raw[key] >= 0}
    for group, keys in [('input_tokens_details', ('cached_tokens',)),
                        ('output_tokens_details', ('reasoning_tokens',))]:
        nested = raw.get(group)
        if isinstance(nested, dict):
            out[group] = {key: nested[key] for key in keys
                          if type(nested.get(key)) is int and nested[key] >= 0}
    return out


class APIClient:
    def __init__(self, config, *, transport=None, timeout=240.):
        self.config = config
        self.http = httpx.Client(
            base_url=config.base_url + '/',
            headers={'Authorization': 'Bearer ' + config.api_key},
            timeout=httpx.Timeout(timeout, connect=20.),
            follow_redirects=False, transport=transport)

    def __repr__(self):
        return f'APIClient(model={MODEL!r}, reasoning_effort={REASONING_EFFORT!r})'

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        self.http.close()

    def _request(self, method, path, **kwargs):
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError:
            raise OpenAIClientError('OpenAI transport request failed') from None
        if not 200 <= response.status_code < 300:
            raise OpenAIClientError(f'OpenAI HTTP request failed (status={response.status_code})')
        try:
            payload = response.json()
        except ValueError:
            raise OpenAIClientError('OpenAI returned invalid JSON') from None
        if not isinstance(payload, dict):
            raise OpenAIClientError('OpenAI returned a non-object response')
        return payload

    def models(self):
        payload = self._request('GET', 'models')
        data = payload.get('data')
        if not isinstance(data, list):
            raise OpenAIClientError('OpenAI model list has an invalid schema')
        ids = [row['id'] for row in data if isinstance(row, dict) and isinstance(row.get('id'), str)]
        return {'model': MODEL, 'available': MODEL in ids,
                'model_ids': [value.replace(self.config.api_key, '[REDACTED]') for value in ids]}

    def complete(self, system, user, *, json_output=False,
                 max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
        """Return public output and metadata; request model and effort are fixed."""
        if not isinstance(system, str) or not system.strip():
            raise ValueError('A nonempty system prompt is required')
        if not isinstance(user, str):
            try:
                user = json.dumps(user, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            except (TypeError, ValueError):
                raise ValueError('User input must be text or JSON-serializable data') from None
        if not user.strip():
            raise ValueError('A nonempty user input is required')
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('max_output_tokens must be a positive integer')
        payload = {'model': MODEL, 'reasoning': {'effort': REASONING_EFFORT},
                   'store': False, 'max_output_tokens': max_output_tokens,
                   'input': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': user}]}
        if json_output:
            payload['text'] = {'format': {'type': 'json_object'}}
        raw = self._request('POST', 'responses', json=payload)
        if not _requested_family(raw.get('model')):
            raise OpenAIClientError('OpenAI response model differs from the fixed requested model')
        if raw.get('status') != 'completed' or raw.get('error') is not None:
            raise OpenAIClientError('OpenAI response did not complete successfully')
        if not isinstance(raw.get('output'), list):
            raise OpenAIClientError('OpenAI response has invalid output')
        parts = []
        for item in raw['output']:
            if not isinstance(item, dict) or item.get('type') != 'message':
                continue  # Drop all reasoning/tool items, including summaries.
            if item.get('status', 'completed') != 'completed' or item.get('role') != 'assistant':
                raise OpenAIClientError('OpenAI output message is not a completed assistant message')
            if not isinstance(item.get('content'), list):
                raise OpenAIClientError('OpenAI output message has invalid content')
            for content in item['content']:
                if not isinstance(content, dict):
                    raise OpenAIClientError('OpenAI output content has invalid schema')
                if content.get('type') == 'refusal':
                    raise OpenAIClientError('OpenAI response refused the request')
                if content.get('type') == 'output_text':
                    if not isinstance(content.get('text'), str):
                        raise OpenAIClientError('OpenAI public output is not text')
                    parts.append(content['text'])
        public = '\n'.join(parts).replace(self.config.api_key, '[REDACTED]')
        if not public.strip():
            raise OpenAIClientError('OpenAI response contains no public output text')
        if json_output:
            try:
                parsed = json.loads(public)
            except ValueError:
                raise OpenAIClientError('OpenAI public output is not valid JSON') from None
            if not isinstance(parsed, dict):
                raise OpenAIClientError('OpenAI public output must be a JSON object')
        response_id = raw.get('id')
        if not isinstance(response_id, str):
            raise OpenAIClientError('OpenAI response is missing an identifier')
        return {'output_text': public, 'id': response_id.replace(self.config.api_key, '[REDACTED]'),
                'model': raw['model'], 'requested_model': MODEL, 'resolved_model': raw['model'],
                'status': 'completed', 'usage': _usage(raw.get('usage'))}


def preflight(client, *, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """Explicit opt-in network check: model listing, then one minimal Responses call."""
    listing = client.models()
    if not listing['available']:
        raise OpenAIClientError('The fixed requested model is absent from the account model list')
    result = client.complete('Return only the JSON object {"ok":true}.',
                             'Confirm this transport check with the requested JSON.',
                             json_output=True, max_output_tokens=max_output_tokens)
    if json.loads(result['output_text']) != {'ok': True}:
        raise OpenAIClientError('OpenAI preflight returned an unexpected public result')
    return {'status': 'PASS', 'model': MODEL, 'reasoning_effort': REASONING_EFFORT,
            'model_available': True, 'response': result}
