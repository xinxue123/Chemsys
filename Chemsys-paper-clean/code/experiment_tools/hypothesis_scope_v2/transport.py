"""Stateless API requests with a sealed record of each attempt.

The credential is read only through load_config, from a path in CHEM_OPENAI_CONFIG.
It is never copied, logged or written to any record.
"""
from datetime import datetime, timezone
import json
import os
from chem.benchmarks.knowledge_openai_client import APIClient, load_config
from chemical_hypothesis_bo.hypotheses import digest
from kernel_contrast_qlogei.study import ROOT, read, write
from . import VARIANT, MODEL, EFFORT, MAXIMUM_SCHEMA_ATTEMPTS, MAXIMUM_OUTPUT_TOKENS

BASE = ROOT/'experiments/runs'/VARIANT


def wants_json_mode(system, public):
    """The API requires the literal word in the input for JSON mode."""
    return 'json' in (system + json.dumps(public, ensure_ascii=False)).casefold()


def request(namespace, stage, system, public, validator):
    context = dict(system=system, public=public, model=MODEL, effort=EFFORT, stage=stage)
    key = digest(context)
    folder = BASE/'requests'/namespace/stage
    done = folder/'validated.json'
    if done.exists():
        saved = read(done)
        assert saved['context_sha256'] == key, 'Sealed response belongs to a different context'
        assert saved['plan'] == validator(saved['raw'])
        return saved
    config = load_config(os.environ['CHEM_OPENAI_CONFIG'])
    error = None
    for attempt in range(1, MAXIMUM_SCHEMA_ATTEMPTS + 1):
        folder_a = folder/f'attempt{attempt}'
        payload = public if attempt == 1 else dict(original=public, schema_repair_only=error)
        if not (folder_a/'api_result.json').exists():
            write(folder_a/'request.json', dict(system=system, public=payload, model=MODEL, effort=EFFORT,
                                                stage=stage, attempt=attempt, context_sha256=key))
            with APIClient(config, timeout=240.) as client:
                result = client.complete(system, payload, json_output=wants_json_mode(system, payload),
                                         max_output_tokens=MAXIMUM_OUTPUT_TOKENS)
            write(folder_a/'api_result.json', result)
        result = read(folder_a/'api_result.json')
        try:
            raw = json.loads(result['output_text'])
            plan = validator(raw)
        except (ValueError, TypeError, KeyError) as exc:
            error = dict(error=str(exc), repair_scope='schema/input validity only; no new observations')
            write(folder_a/'error.json', error)
            continue
        saved = dict(context_sha256=key, raw=raw, plan=plan, attempt=attempt, response_id=result['id'],
                     resolved_model=result['resolved_model'], usage=result['usage'],
                     transport_type='openai_responses_api', sealed_UTC=datetime.now(timezone.utc).isoformat(),
                     request=(folder_a/'request.json').relative_to(ROOT).as_posix())
        write(done, saved)
        return saved
    raise ValueError('Two schema attempts failed: ' + namespace + '/' + stage)
