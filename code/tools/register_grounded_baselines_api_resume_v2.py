"""Append the verified pre-generation JSON-mode correction; preserve v1 evidence."""
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import grounded_baselines_api_dispatch_v2 as d


def main():
    if d.REG.exists():
        print('Transport correction is already registered.')
        return
    original = d.read(d.BASE/'api_transport_registration.json')
    for path,h in original['source_sha256'].items():assert d.sha(d.ROOT/path) == h,path
    diagnostic = d.read(d.OPS/'json_mode_diagnostic.json')
    assert diagnostic['HTTP_status'] == 400 and diagnostic['error']['param'] == 'input'
    assert 'must contain the word' in diagnostic['error']['message']
    errors = list(d.BASE.glob('requests/*/*/attempt*/api_transport_error.json'))
    assert len(errors) == 4
    archived = {}
    affected = []
    copied = []
    for index,error in enumerate(errors):
        af = error.parent
        assert af.resolve().is_relative_to(d.BASE.resolve())
        req = d.read(af/'request.json')
        assert req['task'].endswith('_lgbo') and 'json' not in json.dumps(dict(system=req['system'],public=req['public'])).casefold()
        assert d.read(error)['message'] == 'OpenAI HTTP request failed (status=400)'
        assert not any((af/name).exists() for name in ('api_result.json','receipt.json','completion.json','response.raw.json'))
        assert not (d.ROOT/d.read(af/'export.json')['output']).exists()
        for name in ('assignment.json','api_call_started.json','api_transport_error.json'):
            source = af/name
            # Keep archive paths short enough for Windows native path limits.
            target = d.OPS/'bad400'/str(index)/name
            target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists()
            shutil.copyfile(source,target)
            assert d.sha(source) == d.sha(target)
            archived[target.relative_to(d.ROOT).as_posix()] = d.sha(target)
            copied.append((source,target))
        affected.append(af.relative_to(d.ROOT).as_posix())
    # All three records were copied and checked before removing only their exact,
    # verified original paths; no scientific request or result is removed.
    for source,target in copied:
        assert source.resolve().is_relative_to(d.BASE.resolve()) and d.sha(source) == d.sha(target)
        source.unlink()
    receipts = {path.relative_to(d.ROOT).as_posix():d.sha(path) for path in d.BASE.glob('requests/*/*/attempt*/receipt.json')}
    assert len(receipts) == 556
    names = ('grounded_baselines_api_dispatch_v2.py','audit_grounded_baselines_api_v2.py',
             'report_grounded_baselines_api_v2.py','register_grounded_baselines_api_resume_v2.py',
             'test_grounded_baselines_api_transport_v2.py')
    reg = dict(original)
    reg.update(status='REGISTERED_PRE_GENERATION_HTTP400_FORMAT_CORRECTION',
        created_UTC=datetime.now(timezone.utc).isoformat(),supersedes_for_new_calls='api_transport_registration.json',
        previous_registration_sha256=d.sha(d.BASE/'api_transport_registration.json'),
        json_output_rule='Request json_object only when the unchanged scientific input contains the json literal; otherwise use default text output and parse the mandated JSON object.',
        affected_rejected_requests=affected,rejected_calls_produced_no_model_responses=True,
        archived_rejected_HTTP_files_sha256=archived,existing_response_receipts_sha256=receipts,
        diagnostic_sha256=d.sha(d.OPS/'json_mode_diagnostic.json'),
        source_sha256={'code/tools/'+name:d.sha(d.ROOT/'code/tools'/name) for name in names})
    reg['wire_parameters'] = dict(original['wire_parameters'])
    reg['wire_parameters'].pop('text')
    d.write_once(d.REG,reg)
    for path in reg['source_sha256']:
        target = d.BASE/'api_source_snapshot'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((d.ROOT/path).read_bytes())
    print(json.dumps(dict(status=reg['status'],rejected_calls_archived=len(affected),responses_preserved=len(receipts))))


if __name__ == '__main__':main()
