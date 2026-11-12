"""Model-only transport adaptation; preserves the original prompt and validator."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'evidence/deepseek'
BASE = HERE / 'new'
MODEL = 'deepseek-flash'
EFFORT = 'high'  # DeepSeek exposes low/high/max, not the original model's medium.
MAXIMUM_SCHEMA_ATTEMPTS = 2
MAXIMUM_OUTPUT_TOKENS = 32768


def digest(value):
    from chemical_hypothesis_bo.hypotheses import digest as original
    return original(value)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == value, 'Attempted overwrite of sealed API record'
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def connection():
    raw = Path(os.environ['CHEM_DEEPSEEK_CONFIG']).read_text(encoding='utf-8-sig')
    base = re.search(r'(?im)^\s*BASE_URL\s*[:=]\s*(.+?)\s*$', raw).group(1).strip().strip('\"\x27').rstrip('/')
    key = re.search(r'(?im)^\s*DEEPSEEK_API_KEY\s*[:=]\s*(.+?)\s*$', raw).group(1).strip().strip('\"\x27')
    url = urlsplit(base)
    assert url.scheme == 'https' and url.hostname == 'api.deepseek.com'
    assert not url.username and not url.password and not url.query and not url.fragment
    return base, key


def complete(system, public, record):
    base, key = connection()
    payload = dict(model=MODEL, reasoning_effort=EFFORT, thinking=dict(type='enabled'),
                   max_tokens=MAXIMUM_OUTPUT_TOKENS,
                   messages=[dict(role='system', content=system),
                             dict(role='user', content=json.dumps(public, ensure_ascii=False))])
    if 'json' in (system + json.dumps(public, ensure_ascii=False)).casefold():
        payload['response_format'] = dict(type='json_object')
    # Credentials, private reasoning text, and HTTP headers are never persisted.
    write(record / 'wire_payload.json', payload)
    for retry in range(4):
        req = urllib.request.Request(base + '/chat/completions',
                                     data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
                                     method='POST')
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as error:
            write(record / f'transport_error{retry + 1}.json',
                  dict(type='HTTPError', code=error.code, retry=retry + 1, UTC=datetime.now(timezone.utc).isoformat()))
            if error.code not in (429, 500, 502, 503, 504) or retry == 3:
                raise RuntimeError(f'DeepSeek HTTP status {error.code}') from None
            time.sleep(3 * (retry + 1))
    choice = data['choices'][0]
    output = choice['message'].get('content')
    result = dict(id=data['id'], resolved_model=data.get('model'), usage=data.get('usage', {}),
                  output_text=output or '', finish_reason=choice.get('finish_reason'),
                  elapsed_seconds=time.monotonic() - started, created=data.get('created'),
                  endpoint=base, thinking_enabled=True, reasoning_effort=EFFORT)
    write(record / 'api_result.json', result)
    return result


def resolve_own_query_references(raw, public):
    """Resolve only bare own-query numbers to their explicit candidate IDs.

    Used after two complete strict-schema responses have failed. No scope,
    forecast, claim or candidate is altered and no new model call is made.
    """
    observations=public.get('own_observations',[])
    lookup={str(x['query']):x['candidate_id'] for x in observations if 'query' in x}
    if len(lookup)!=len(observations) or not isinstance(raw,dict):
        return None,[]
    normalized=deepcopy(raw);changes=[]
    for note in normalized.get('hypotheses',[]):
        if not isinstance(note,dict) or not isinstance(note.get('supporting_observation_ids'),list):
            continue
        for i,value in enumerate(note['supporting_observation_ids']):
            if type(value) not in (str,int) or not re.fullmatch(r'[1-9][0-9]*',str(value)):
                continue
            key=str(value)
            if key in lookup:
                note['supporting_observation_ids'][i]=lookup[key]
                changes.append(dict(hypothesis_id=note.get('id'),query_reference=value,candidate_id=lookup[key]))
    return (normalized if changes else None),changes


def request(namespace, stage, system, public, validator):
    assert not os.environ.get('CHEM_DEEPSEEK_OFFLINE'), 'API disabled during numerical qualification'
    assert '\\' not in namespace and '..' not in namespace.split('/')
    context = dict(system=system, public=public, model=MODEL, effort=EFFORT, stage=stage)
    key = digest(context)
    folder = BASE / 'requests' / namespace / stage
    done = folder / 'validated.json'
    if done.exists():
        saved = read(done)
        assert saved['context_sha256'] == key, 'Cached response belongs to a different context'
        if 'normalized_raw' in saved:
            normalized,changes=resolve_own_query_references(saved['raw'],public)
            assert normalized==saved['normalized_raw'] and changes==saved['reference_resolution']
        assert saved['plan'] == validator(saved.get('normalized_raw',saved['raw']))
        return saved
    error = None
    completed_schema_attempts = 0
    truncations = 0
    # Length-truncated generations did not supply a complete schema response.
    # Keep them as billed transport failures; permit at most two COMPLETE schema
    # responses, with at most four length retries. Legacy pilot attempts stay sealed.
    for attempt in range(1, MAXIMUM_SCHEMA_ATTEMPTS + 4 + 1):
        target = folder / f'attempt{attempt}'
        payload = public if completed_schema_attempts == 0 else dict(original=public, schema_repair_only=error)
        if (target / 'request.json').exists():
            prior = read(target / 'request.json')
            assert prior['context_sha256'] == key and prior['system'] == system
            assert prior['public'] == public or prior['public'].get('original') == public
            payload = prior['public']
        else:
            write(target / 'request.json', dict(system=system, public=payload, model=MODEL, effort=EFFORT,
                                              stage=stage, attempt=attempt, context_sha256=key))
        result = read(target / 'api_result.json') if (target / 'api_result.json').exists() else complete(system, payload, target)
        if result.get('finish_reason') == 'length':
            truncations += 1
            write(target / 'truncation.json', dict(finish_reason='length',
                                                  complete_schema_response=False, billed_and_retained=True))
            if truncations >= 4:
                raise ValueError('Four output truncations: ' + namespace + '/' + stage)
            continue
        completed_schema_attempts += 1
        normalization={}
        try:
            raw = json.loads(result['output_text'])
            plan = validator(raw)
        except (ValueError, TypeError, KeyError) as exc:
            error = dict(error=str(exc), repair_scope='schema/input validity only; no new observations')
            write(target / 'error.json', error)
            if completed_schema_attempts >= MAXIMUM_SCHEMA_ATTEMPTS:
                normalized,changes=resolve_own_query_references(raw,public) if error['error']=='Only own observed support IDs allowed' else (None,[])
                if normalized is None:
                    break
                try:
                    plan=validator(normalized)
                except (ValueError,TypeError,KeyError) as resolution_error:
                    write(target/'reference_resolution_error.json',dict(error=str(resolution_error),mapping=changes))
                    break
                normalization=dict(normalized_raw=normalized,reference_resolution=changes)
                write(target/'reference_resolution.json',dict(mapping=changes,other_fields_unchanged=True,
                                                              extra_model_calls=0,strict_attempts=completed_schema_attempts))
            else:
                continue
        saved = dict(context_sha256=key, raw=raw, plan=plan, attempt=attempt,
                     completed_schema_attempt=completed_schema_attempts, output_truncations=truncations,
                     response_id=result['id'], resolved_model=result['resolved_model'], usage=result['usage'],
                     transport_type='deepseek_chat_completions_api',
                     sealed_UTC=datetime.now(timezone.utc).isoformat(),
                     request=(target / 'request.json').relative_to(ROOT).as_posix(),**normalization)
        write(done, saved)
        print(json.dumps(dict(event='API_VALIDATED', namespace=namespace, stage=stage,
                              attempt=attempt, tokens=result['usage'].get('total_tokens'),
                              elapsed_seconds=round(result['elapsed_seconds'], 2))), flush=True)
        return saved
    raise ValueError('Two schema attempts failed: ' + namespace + '/' + stage)
