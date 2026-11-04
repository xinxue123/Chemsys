"""Resume registered work after the tool runtime lost its worker processes.

No scientific source, saved response, completed trajectory or budget is changed.
Native checkpoint/commit handling and the already registered RBO projection are
reused. FileLock releases dead processes' operating-system locks automatically.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/chemist_system'), str(ROOT/'code/experiment_tools'),
               str(ROOT/'code/tools')]
from v13_late_validation import study, queue
import v13_late_rbo_compat as compat


def worker(job, protocol):
    if job['arm'] == 'rbo':
        with patch.object(queue, 'request', side_effect=compat.request):
            return study.run_job(job, protocol)
    return study.run_job(job, protocol)


def main():
    protocol = study.check(True)
    b = study.base()
    original_compat = queue.read(b/'RBO_empty_sidecar_compat_registration.json')
    if original_compat['adapter_sha256'] != queue.file_sha(compat.__file__):
        raise ValueError('Registered compatibility source changed')
    complete, remaining, states = [], [], {}
    immutable = {}
    for job in protocol['jobs']:
        folder = b/job['id']/'runs'/job['id']
        sf = folder/'state.json'
        s = queue.read(sf) if sf.exists() else None
        states[job['id']] = dict(status=s['status'] if s else 'WAITING',
            queries=len(s['history']) if s else 0,
            path=sf.relative_to(ROOT).as_posix() if s else None,
            sha256=queue.file_sha(sf) if s else None)
        if s and s['status'] == 'COMPLETE':
            complete.append(job['id'])
            immutable[sf.relative_to(ROOT).as_posix()] = queue.file_sha(sf)
        else:
            remaining.append(job)
        for cp in (folder/'rounds').glob('*_commit.json'):
            immutable[cp.relative_to(ROOT).as_posix()] = queue.file_sha(cp)
    for name in ('request.json', 'context.json', 'invocation.json',
                 'assignment.json', 'response.raw.json', 'response_receipt.json'):
        for path in (b/'requests').rglob(name):
            immutable[path.relative_to(ROOT).as_posix()] = queue.file_sha(path)
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        cause='Tool runtime sessions disappeared and no Python worker process remained',
        original_protocol_sha256=protocol['protocol_sha256'],
        adapter_sha256=queue.file_sha(__file__),
        original_compat_sha256=original_compat['adapter_sha256'],
        completed_trajectories_preserved=complete,
        resumed_jobs=[j['id'] for j in remaining],
        states_before=states, immutable_existing_files_sha256=immutable,
        saved_responses_reused_without_new_sampling=True,
        scientific_sources_and_budgets_unchanged=True, workers=6)
    queue.write_once(b/'runtime_resume_registration.json', registration)
    errors = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(worker, j, protocol): j for j in remaining}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                print(json.dumps(dict(job=job['id'], result=result)), flush=True)
            except Exception as exc:
                errors.append(dict(job=job['id'], type=type(exc).__name__, message=str(exc)))
                print(json.dumps(errors[-1]), flush=True)
    for name, value in immutable.items():
        if queue.file_sha(ROOT/name) != value:
            raise ValueError('Pre-reset immutable evidence changed: '+name)
    study.check(True)
    final = {}
    for job in protocol['jobs']:
        sf = b/job['id']/'runs'/job['id']/'state.json'
        s = queue.read(sf)
        final[job['id']] = dict(status=s['status'], queries=len(s['history']),
                               state_sha256=queue.file_sha(sf))
    done = not errors and all(s['status']=='COMPLETE' and s['queries']==20 for s in final.values())
    receipt = dict(status='COMPLETE' if done else 'INCOMPLETE',
        operational_resume=True, errors=errors, final=final,
        completed_before_reset=len(complete), original_completed_states_unchanged=True,
        preexisting_commits_packets_and_raw_responses_unchanged=True,
        scientific_sources_and_budgets_unchanged=True,
        registration_sha256=queue.file_sha(b/'runtime_resume_registration.json'))
    queue.write_once(b/'runtime_resume_completion.json', receipt)
    # The original executors were interrupted before writing their end receipts.
    # These explicitly identify the operational resume, rather than inventing
    # an original uninterrupted native-batch success.
    queue.write_once(b/'batch_receipt.json', receipt)
    queue.write_once(b/'RBO_compat_batch_receipt.json', dict(
        operational_resume=True, original_compat_registration_unchanged=True,
        results=[dict(job=j['id'], **final[j['id']]) for j in protocol['jobs'] if j['arm']=='rbo']))
    print(json.dumps(dict(status=receipt['status'], complete=sum(s['status']=='COMPLETE' for s in final.values()))), flush=True)


if __name__ == '__main__':
    main()
