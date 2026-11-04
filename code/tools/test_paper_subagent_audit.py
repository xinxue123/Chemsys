"""Offline mutation checks for request isolation and exact output custody."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import paper_subagent_audit as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def assignment(agent_id='synthetic-agent'):
    return dict(backend=audit.BACKEND, requested_model='gpt-6.1-terra',
                requested_reasoning_effort='medium', fork_turns='none',
                context_policy=audit.CONTEXT_POLICY, agent_id=agent_id,
                model_verification='local_orchestrator_spawn_receipt_not_provider_attestation',
                spawn_request=dict(model='gpt-6.1-terra', reasoning_effort='medium',
                                   fork_turns='none', task_name='synthetic_request'),
                spawn_result=dict(agent_id=agent_id))


class AgentProvenanceTests(unittest.TestCase):
    protocol = dict(model='gpt-6.1-terra', reasoning_effort='medium')

    def test_completed_input_custody_detects_changed_and_missing_bytes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'case/runs/native'
            for path in (run / 'state.json', run / 'initial_commit.json', root / 'case/launch.json'):
                write(path, {'frozen': True})
            write(run / 'rounds/01_commit.json', {'selected': 'p1'})
            gp = run / 'rounds/01_GP.npz'
            gp.write_bytes(b'saved GP arrays')
            write(root / 'case/requests/key/context.json', {'public': {}})
            write(root / 'case/queue/key/attempt1/export.json',
                  {'packet_relpath': 'opaque/input.json', 'output_relpath': 'opaque/output.json'})
            write(root / 'opaque/input.json', {'system': 'exact', 'public': {}})
            write(root / 'opaque/output.json', {'selected': 'p1'})
            snapshot = audit.completed_input_snapshot(root, run)
            audit.verify_input_snapshot(root, snapshot)
            self.assertIn('opaque/output.json', snapshot)
            gp.write_bytes(b'changed GP arrays')
            with self.assertRaises(AssertionError):
                audit.verify_input_snapshot(root, snapshot)
            gp.write_bytes(b'saved GP arrays')
            (root / 'opaque/output.json').unlink()
            with self.assertRaises(AssertionError):
                audit.verify_input_snapshot(root, snapshot)

    def test_actual_saved_spawn_settings_and_receipt_are_required(self):
        good = assignment()
        self.assertEqual(audit.validate_assignment(good, self.protocol), 'synthetic-agent')
        mutations = [('model', 'gpt-6.1-sol'), ('reasoning_effort', 'high'), ('fork_turns', 'all')]
        for key, bad in mutations:
            changed = deepcopy(good)
            changed['spawn_request'][key] = bad
            with self.assertRaises(AssertionError):
                audit.validate_assignment(changed, self.protocol)
        changed = deepcopy(good)
        changed['spawn_result']['agent_id'] = 'different-agent'
        with self.assertRaises(AssertionError):
            audit.validate_assignment(changed, self.protocol)

    def test_fresh_agent_per_attempt_and_separate_local_identity(self):
        records = [dict(response_id='subagent:a', agent_id='agent-a'),
                   dict(response_id='subagent:b', agent_id='agent-b')]
        self.assertEqual(audit.validate_unique_records(records), 2)
        for changed in ([records[0], dict(records[1], agent_id='agent-a')],
                        [records[0], dict(records[1], response_id='subagent:a')],
                        [dict(records[0], response_id='resp_http')]):
            with self.assertRaises(AssertionError):
                audit.validate_unique_records(changed)
        with self.assertRaises(AssertionError):
            audit.validate_unique_records(records, {'subagent:a'})

    def test_schema_repairs_include_only_original_public_and_recorded_error(self):
        context = dict(system='frozen system', public={'query': 1, 'candidate_menu': []})
        self.assertEqual(audit.exact_attempt_request(context, 1), context)
        self.assertEqual(audit.exact_attempt_request(context, 2, 'invalid schema'),
                         dict(system=context['system'], public=dict(original=context['public'], schema_repair_only='invalid schema')))
        for number, error in ((1, 'extra'), (2, None), (4, 'invalid schema')):
            with self.assertRaises(AssertionError):
                audit.exact_attempt_request(context, number, error)

    def test_transport_continuation_cannot_resample_an_existing_response(self):
        a=assignment();a['spawn_request']['message']='identical registered file instruction'
        before=dict(original_agent=a['agent_id'],request_sha256='request',assignment_sha256='assignment',
                    observed_tool_status='interrupted',output_exists=False,response_raw_exists=False,provenance_exists=False)
        after=dict(agent_id=a['agent_id'],assignment_unchanged=True,scientific_response_submitted_yet=False,
                   output_exists=True,output_sha256='output')
        followup=dict(target=a['agent_id'],message=a['spawn_request']['message'])
        audit.validate_continuation_record(before,after,followup,a,{'request_sha256':'request'},'assignment','output')
        for key in ('output_exists','response_raw_exists','provenance_exists'):
            with self.assertRaises(AssertionError):
                audit.validate_continuation_record(dict(before,**{key:True}),after,followup,a,{'request_sha256':'request'},'assignment','output')

    def test_transport_continuation_cannot_add_context_or_change_agent(self):
        a=assignment();a['spawn_request']['message']='identical registered file instruction'
        before=dict(original_agent=a['agent_id'],request_sha256='request',assignment_sha256='assignment',
                    observed_tool_status='interrupted',output_exists=False,response_raw_exists=False,provenance_exists=False)
        after=dict(agent_id=a['agent_id'],assignment_unchanged=True,scientific_response_submitted_yet=False,
                   output_exists=True,output_sha256='output')
        followup=dict(target=a['agent_id'],message=a['spawn_request']['message'])
        for changed in (dict(followup,target='another-agent'),dict(followup,message='use a better candidate')):
            with self.assertRaises(AssertionError):
                audit.validate_continuation_record(before,after,changed,a,{'request_sha256':'request'},'assignment','output')

    def fixture(self, root):
        native = root / 'requests' / 'context' / 'attempt1'
        queue = root / 'queue' / 'context' / 'attempt1'
        context = dict(system='return public JSON', public={'query': 1},
                       model=self.protocol['model'], reasoning_effort=self.protocol['reasoning_effort'])
        expected = audit.exact_attempt_request(context, 1)
        packet = dict(expected, context_sha256=audit.native.study.digest(context),
                      requested_model=self.protocol['model'],
                      requested_reasoning_effort=self.protocol['reasoning_effort'],
                      schema_version=1, backend=audit.BACKEND, attempt=1, job=['synthetic', 0, 'memory'],
                      json_output=True, max_output_tokens=6000, fork_turns='none', context_policy=audit.CONTEXT_POLICY)
        request_hash = audit.native.study.digest(packet)
        packet['request_sha256'] = request_hash
        write(native / 'request.json', expected)
        write(queue / 'request.json', packet)
        assigned = dict(assignment(), request_sha256=request_hash)
        write(queue / 'assignment.json', assigned)
        raw_text = '{"summary":"synthetic public response", "actions":[]}\n'
        (queue / 'response.raw.json').write_text(raw_text, encoding='utf-8', newline='')
        provenance = {key: assigned[key] for key in ('backend', 'agent_id', 'requested_model',
                     'requested_reasoning_effort', 'fork_turns', 'context_policy', 'model_verification')}
        provenance.update(request_sha256=request_hash, raw_output_sha256=audit.file_sha(queue / 'response.raw.json'),
                          assignment_sha256=audit.file_sha(queue / 'assignment.json'),
                          response_id_kind='local_content_receipt', context_sha256=packet['context_sha256'], usage_available=False)
        provenance['response_id'] = 'subagent:' + audit.native.study.digest({k: provenance[k] for k in
                                    ('request_sha256', 'raw_output_sha256', 'assignment_sha256')})
        write(queue / 'provenance.json', provenance)
        write(native / 'response.json', dict(output_text=raw_text, id=provenance['response_id'], usage={},
              backend=audit.BACKEND, usage_available=False, transport_provenance=provenance))
        return native, queue, context

    def test_literal_output_custody_rejects_mutation_and_fabricated_metadata(self):
        with TemporaryDirectory(prefix='subagent_audit_') as tmp:
            root = Path(tmp)
            native, queue, context = self.fixture(root)
            result = audit.audit_attempt(root, native, queue, context, 1, self.protocol)
            self.assertEqual(result['agent_id'], 'synthetic-agent')
            original = audit.read(native / 'response.json')
            for altered in (dict(original, output_text='{}'), dict(original, usage={'input_tokens': 1}),
                            dict(original, resolved_model='gpt-6.1-terra')):
                write(native / 'response.json', altered)
                with self.assertRaises(AssertionError):
                    audit.audit_attempt(root, native, queue, context, 1, self.protocol)
            write(native / 'response.json', original)
            (queue / 'response.raw.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(AssertionError):
                audit.audit_attempt(root, native, queue, context, 1, self.protocol)

    def test_independent_removal_rules_keep_their_distinct_transitions(self):
        # These are the independent rules used by the delegated native auditor.
        state = audit.native.study.v12.engine.kf.initial()
        for t in range(1, 6):
            updated, _ = audit.native.independent_fixed_kf(state, .95, .2, False, t)
            self.assertEqual((updated['status'], updated['mean'], updated['variance'], updated['weight']),
                             ('active', 0., .25, .4))
            self.assertEqual(updated['local_updates'], t)
            state = updated
        self.assertFalse(audit.native.independent_initial_edit(None, {'edit_calls': [1]}, None, 4)['allowed'])

    def test_opaque_prompt_rejects_hidden_arm_and_seed_hints(self):
        with TemporaryDirectory(prefix='opaque_') as tmp:
            root=Path(tmp).resolve();native,queue,context=self.fixture(root)
            packet=audit.read(queue/'request.json');assigned=audit.read(queue/'assignment.json')
            opaque=audit.native.study.digest(dict(purpose='isolated_public_packet',request_sha256=packet['request_sha256']))
            folder=root/'experiments/agent_io/paper_v12_subagent'/opaque
            input_file=folder/'input.json';output_file=folder/'output.json'
            expected=dict(system=packet['system'],public=packet['public']);write(input_file,expected)
            output_file.write_bytes((queue/'response.raw.json').read_bytes())
            template=root/'instructions.txt';template.write_text('Read {input_path}. Write {output_path}.',encoding='utf-8')
            protocol=dict(self.protocol,decision_instructions_path='instructions.txt',decision_instructions_sha256=audit.file_sha(template))
            assigned['spawn_request']['message']=template.read_text(encoding='utf-8').format(input_path=input_file.as_posix(),output_path=output_file.as_posix())
            export=dict(exposure_policy='opaque_exact_system_public_packet',request_sha256=packet['request_sha256'],
                packet_relpath=input_file.relative_to(root).as_posix(),output_relpath=output_file.relative_to(root).as_posix(),
                packet_sha256=audit.file_sha(input_file),packet_payload_sha256=audit.native.study.digest(expected))
            write(queue/'export.json',export)
            audit.validate_public_export(root,queue,packet,assigned,protocol)
            hidden=deepcopy(assigned);hidden['spawn_request']['message']+=' You are the no_KF seed3 arm.'
            with self.assertRaises(AssertionError):audit.validate_public_export(root,queue,packet,hidden,protocol)
            write(input_file,dict(expected,seed=3))
            with self.assertRaises(AssertionError):audit.validate_public_export(root,queue,packet,assigned,protocol)


if __name__ == '__main__':
    unittest.main()
