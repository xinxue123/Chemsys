"""Isolated R7-S OpenAI 6.1 Terra/medium study, preserving frozen BO code.

Wraps the pre-existing R7 step/decision/audit implementation. Only the LLM
provider and the protocol/run paths are replaced. Old DeepSeek runs, R7 B0
records, caches and source files remain unmodified.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

from . import continuous_benchmark_data as data
from . import continuous_benchmark_engine as base
from . import continuous_r6_engine as r6
from . import continuous_r7_engine as r7
from . import continuous_r7_study as frozen
from . import continuous_r7_openai_llm as llm


ROOT = Path('data/runs/continuous_r7_openai_terra_medium_20261030')
OUT = Path('research/continuous_r7_openai_terra_medium_20261030')
TASKS = frozen.TASKS
ARMS = r7.ARMS


def _paths():
    return sorted(set(frozen._paths() + [Path(__file__), Path(llm.__file__),
                                         Path('continuous_r7_openai.py')]))


def qualify():
    return frozen.qualify()


def prepare():
    if (OUT / 'protocol.json').exists():
        return check()
    source = qualify()
    if source['status'] != 'PASS':
        raise FileNotFoundError('Original SnAr/HPLC pool and oracle are required')
    protocol = {
        'name': 'R7_S_OpenAI_gpt_6_1_terra_medium_passive_covariance',
        'tasks': list(TASKS), 'arms': list(ARMS),
        'initial_queries': 4, 'additional_queries': 16,
        'GP': 'unchanged original SingleTaskGP and MC512 qLogEI',
        'LLM': {
            'provider': 'OpenAI API',
            'endpoint': llm.ENDPOINT + '/responses',
            'model': llm.MODEL,
            'reasoning_effort': llm.REASONING_EFFORT,
            'max_output_tokens': llm.MAX_OUTPUT_TOKENS,
            'format': 'json_object', 'store': False,
            'system_prompt_sha256': r6.digest(llm.SYSTEM),
            'frozen_first_call_rounds': list(r7.CALL_ROUNDS),
            'adaptive_call_rounds': list(r7.ONLINE_CALL_ROUNDS),
            'identical_first_round_request_for_B2_B4': True,
            'accepted_visible_JSON_only': True,
            'credential_not_copied': True,
        },
        'knowledge': {
            'representation': 'anchored pairwise sign event',
            'propagation': 'joint original GP posterior covariance',
            'evidence': 'passive prequential score; rollback if contradicted',
            'active_probes': 0, 'gate_exceptions': 0,
        },
        'data_sha256': {v['task']: {'pool': v['input_sha256'], 'oracle': v['oracle_sha256']}
                        for v in source['tasks']},
        'sources_sha256': {p.as_posix(): data.filehash(p) for p in _paths()},
        'source_refs': {'original_main': 'cf5c01b32af8e8417b66e4aa9b314c8acb55ef5a',
                        'original_R7_PR2': '92621378e7972d4353b07cf96fa3f9b02cd6ee56'},
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'no_prior_performance_claim': True,
    }
    protocol['protocol_sha256'] = r6.digest(protocol)
    data.write(OUT / 'protocol.json', protocol)
    return protocol


def check():
    protocol = data.read(OUT / 'protocol.json')
    assert protocol['protocol_sha256'] == r6.digest({
        k: v for k, v in protocol.items() if k != 'protocol_sha256'})
    assert protocol['sources_sha256'] == {p.as_posix(): data.filehash(p) for p in _paths()}, 'source drift'
    for task, hashes in protocol['data_sha256'].items():
        assert data.filehash(data.ROOT / task / 'pool.json') == hashes['pool']
        assert data.filehash(data.ROOT / task / 'oracle.npz') == hashes['oracle']
    assert protocol['LLM']['model'] == llm.MODEL
    assert protocol['LLM']['reasoning_effort'] == llm.REASONING_EFFORT
    return protocol


@contextmanager
def _isolated_runtime():
    """Scope the frozen study's globals to a separate R7/OpenAI run tree."""
    names = ('ROOT', 'OUT', '_runtime', 'check')
    previous = {name: getattr(frozen, name) for name in names}
    try:
        frozen.ROOT = ROOT
        frozen.OUT = OUT
        frozen._runtime = lambda: (data, base, llm, llm)
        frozen.check = check
        yield
    finally:
        for name, value in previous.items():
            setattr(frozen, name, value)


def run_one(task, arm, seed=0, config=''):
    if task not in TASKS or arm not in ARMS:
        raise ValueError('unknown R7 task or arm')
    if arm != 'B0' and not config:
        raise ValueError('non-B0 R7 arms require --key-file with a local API key')
    with _isolated_runtime():
        return frozen.run_one(task, arm, seed, config)


def status():
    with _isolated_runtime():
        return frozen.status()


def audit():
    with _isolated_runtime():
        report = frozen.audit()
    inspected = []
    for f in sorted((ROOT / 'requests').glob('*/validated.json')):
        row = data.read(f)
        assert row['response_model'] == llm.MODEL
        request = data.read(f.parent / f'attempt{row["accepted_attempt"]:02d}' / 'request.json')
        response = data.read(f.parent / f'attempt{row["accepted_attempt"]:02d}' / 'response.json')
        assert request['model'] == llm.MODEL
        assert request['reasoning']['effort'] == llm.REASONING_EFFORT
        assert response['model'] == llm.MODEL and response['retained_raw_reasoning'] is False
        assert 'api_key' not in request and 'Authorization' not in request
        inspected.append({'context_sha256': row['context_sha256'], 'model': llm.MODEL,
                          'reasoning_effort': llm.REASONING_EFFORT,
                          'status': 'PASS'})
    return dict(report, provider_responses=inspected,
                provider_responses_audited=len(inspected))


def report():
    with _isolated_runtime():
        result = frozen.report()
    return dict(result, model=llm.MODEL, reasoning_effort=llm.REASONING_EFFORT,
                protocol_sha256=check()['protocol_sha256'])
