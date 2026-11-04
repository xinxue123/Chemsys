from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from knowledge_rank_v9 import llm as native
from paper_v12_subagent import transport as t
from paper_v12_subagent import study

ROOT = Path(__file__).resolve().parents[4]


def evidence(agent):
    return dict(spawn_request=dict(task_name='isolated_' + agent, model=t.MODEL,
                                  reasoning_effort=t.REASONING_EFFORT, fork_turns='none'),
                spawn_result=dict(agent_id=agent, task_name='/root/isolated_' + agent))


def publish(request, raw, agent):
    t.assign(request, agent, evidence(agent))
    source = request.parent / 'authored.json'
    source.write_text(raw, encoding='utf-8')
    return t.submit(request, source, agent)


def validator(raw):
    if not isinstance(raw, dict) or set(raw) != {'choose'} or raw['choose'] not in ('A', 'B'):
        raise ValueError('Choose exactly A or B')
    return dict(selected_id=raw['choose'])


class TransportTests(unittest.TestCase):
    def make(self, folder):
        b = Path(folder) / 'paper_v12_subagent' / 's0' / 'c1' / 'memory'
        return b, t.FileQueueClient(b, ('Reizman_Suzuki_Case_1', 0, 'memory'), wait_seconds=0)

    def call(self, client, public=None):
        return t.request(client, 'Return public JSON only.', public or {'choices': ['A', 'B']},
                         client.base / 'requests', validator)

    def test_native_context_exact_pending_resume_and_cached_receipt(self):
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            b, client = self.make(folder)
            with self.assertRaises(t.PendingRequest) as caught:
                self.call(client)
            path = caught.exception.path
            request = t.read(path)
            context = dict(system='Return public JSON only.', public={'choices': ['A', 'B']},
                           model=t.MODEL, reasoning_effort=t.REASONING_EFFORT)
            self.assertEqual(request['context_sha256'], native.digest(context))
            self.assertEqual(request['public'], context['public'])
            self.assertFalse((path.parent / 'assignment.json').exists())
            publish(path, '{"choose":"B"}\n', 'agent_1')
            result = self.call(client)
            self.assertEqual(result['plan'], {'selected_id': 'B'})
            self.assertEqual(result['usage'], {})
            self.assertFalse(result['usage_available'])
            self.assertEqual(result['backend'], 'codex_subagent')
            self.assertTrue(result['response_id'].startswith('subagent:'))
            self.assertNotIn('resolved_model', result['transport_provenance'])
            snapshot = {p: p.read_bytes() for p in b.rglob('*.json')}
            self.assertEqual(self.call(client), result)
            self.assertEqual({p: p.read_bytes() for p in b.rglob('*.json')}, snapshot)
            self.assertEqual(t.read(path.parent / 'validation.json')['status'], 'VALIDATED')

    def test_schema_only_repairs_are_bounded_and_raw_attempts_immutable(self):
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            b, client = self.make(folder)
            originals = []
            for attempt in range(1, 4):
                with self.assertRaises(t.PendingRequest) as caught:
                    self.call(client)
                path = caught.exception.path
                request = t.read(path)
                self.assertEqual(request['attempt'], attempt)
                if attempt > 1:
                    self.assertEqual(request['public']['original'], {'choices': ['A', 'B']})
                    self.assertIn('schema_repair_only', request['public'])
                publish(path, 'bad JSON' if attempt == 1 else '{"choose":"C"}', 'agent_' + str(attempt))
                originals.append((path.parent / 'response.raw.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'bounded schema-only repair'):
                self.call(client)
            self.assertEqual(len(list((b / 'queue').glob('*/attempt*/request.json'))), 3)
            for i, path in enumerate(sorted((b / 'queue').glob('*/attempt*/response.raw.json'))):
                self.assertEqual(path.read_bytes(), originals[i])
                self.assertEqual(t.read(path.parent / 'validation.json')['status'], 'SCHEMA_REJECTED')

    def test_tampering_and_agent_reuse_are_rejected(self):
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            b, client = self.make(folder)
            with self.assertRaises(t.PendingRequest):
                self.call(client)
            first = client.active
            publish(first, '{"choose":"C"}', 'agent_same')
            with self.assertRaises(t.PendingRequest):
                self.call(client)
            with self.assertRaisesRegex(ValueError, 'fresh agent'):
                t.assign(client.active, 'agent_same', evidence('agent_same'))
            with self.assertRaises(ValueError):
                t.assign(client.active, 'bad', dict(spawn_request=dict(model=t.MODEL,
                         reasoning_effort='medium', fork_turns='all', task_name='bad'),
                         spawn_result=dict(agent_id='bad')))
            publish(client.active, '{"choose":"A"}', 'agent_new')
            (client.active.parent / 'response.raw.json').write_text('{"choose":"B"}', encoding='utf-8')
            with self.assertRaises(t.ReceiptDrift):
                self.call(client)

    def test_actual_spawn_tool_result_task_name_is_preserved(self):
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            _, client = self.make(folder)
            with self.assertRaises(t.PendingRequest):
                self.call(client)
            agent = '/root/isolated_actual'
            receipt = dict(spawn_request=dict(task_name='isolated_actual', model=t.MODEL,
                           reasoning_effort=t.REASONING_EFFORT, fork_turns='none'),
                           spawn_result=dict(task_name=agent))
            saved = t.assign(client.active, agent, receipt)
            self.assertEqual(saved['spawn_result'], {'task_name': agent})
            source = client.active.parent / 'authored.json'
            source.write_text('{"choose":"A"}', encoding='utf-8')
            t.submit(client.active, source, agent)
            self.assertEqual(self.call(client)['plan'], {'selected_id': 'A'})

    def test_native_request_behavior_matches_frozen_wrapper(self):
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            b, client = self.make(folder)
            raws = ['{"choose":"C"}', '{"choose":"A"}']
            seen = []
            class NativeClient:
                def complete(self, system, public, **kwargs):
                    seen.append(public)
                    return dict(output_text=raws[len(seen)-1], id='synthetic', usage={})
            expected = t._NATIVE_REQUEST(NativeClient(), 'Return public JSON only.', {'choices': ['A', 'B']},
                                         Path(folder) / 'native', validator)
            for i, raw in enumerate(raws):
                with self.assertRaises(t.PendingRequest):
                    self.call(client)
                self.assertEqual(t.read(client.active)['public'], seen[i])
                publish(client.active, raw, 'agent_' + str(i))
            actual = self.call(client)
            for key in ('context_sha256', 'raw', 'plan', 'attempts', 'usage'):
                self.assertEqual(actual[key], expected[key])

    def test_all_68_jobs_have_separate_namespaces(self):
        self.assertEqual(len(study.JOBS), 68)
        self.assertEqual(len(set(study.JOBS)), 68)
        self.assertEqual(len({study.base(ROOT, *j) for j in study.JOBS}), 68)
        for task in study.original.FOCUS:
            for seed in range(4):
                self.assertIn((task, seed, 'memory'), study.JOBS)
        with self.assertRaises(ValueError):
            study.base(ROOT, 'HPLC', 0, 'bo')

    def test_opaque_export_discloses_only_exact_public_packet(self):
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            root = Path(folder)
            b = root / 'experiments/runs/paper_v12_subagent/s0/c1/memory'
            client = t.FileQueueClient(b, ('Reizman_Suzuki_Case_1', 0, 'memory'), wait_seconds=0)
            with self.assertRaises(t.PendingRequest):
                self.call(client)
            receipt = t.export_request(root, client.active)
            packet = t.read(root / receipt['packet_relpath'])
            self.assertEqual(packet, {'system': 'Return public JSON only.', 'public': {'choices': ['A', 'B']}})
            self.assertEqual(len(Path(receipt['packet_relpath']).parent.name), 64)
            self.assertFalse((root / receipt['output_relpath']).exists())
            self.assertEqual(receipt, t.export_request(root, client.active))

    def test_cached_extractor_is_identical_and_compiles_once(self):
        insight = dict(comment='Synthetic', keywords='test', hypotheses=[
            dict(strategy='second', rationale='test', confidence='medium', parameter_sets=[{'candidate_id': 'B'}]),
            dict(strategy='first', rationale='test', confidence='high', parameter_sets=[{'candidate_id': 'A'}]),
            dict(strategy='tie', rationale='test', confidence='high', parameter_sets=[{'candidate_id': 'C'}])])
        study.v12.llm.compiled_extractor.cache_clear()
        expected = native.official_extract(ROOT, insight)
        self.assertEqual(study._cached_v9_extract(ROOT, insight), expected)
        self.assertEqual(study._cached_v9_extract(ROOT, insight), expected)
        self.assertEqual(study.v12.llm.compiled_extractor.cache_info().misses, 1)


if __name__ == '__main__':
    unittest.main()
