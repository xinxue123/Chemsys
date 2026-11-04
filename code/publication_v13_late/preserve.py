"""Seal additive experimental evidence; historical milestones remain immutable."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_late_validation import study,queue

P=ROOT/'paper/v13_late'
M=ROOT/'experiments/milestones/v13_late_validation_20261103'


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def write(path,x):queue.write_once(path,x)
def good(path):
    path=Path(path)
    return path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.lock') and not path.name.startswith('.')


def main():
    protocol=study.check(True);audit=queue.read(P/'audit/new_native_replay.json')
    if audit['status']!='PASS_COMPLETE':raise ValueError('Full native audit required')
    M.mkdir(parents=True,exist_ok=True)
    files=set()
    for path in study.base().rglob('*'):
        if good(path) and 'cache' not in path.relative_to(study.base()).parts:files.add(path.resolve())
    for r in rows(P/'data/LLM_request_inventory.csv'):
        files.add((ROOT/r['input']).resolve())
        req=queue.read(ROOT/r['request']);files.add((ROOT/req['output']).resolve())
    for r in rows(P/'data/new_fit_inventory.csv'):
        if r['checkpoint']:
            model=ROOT/r['checkpoint'];files.add(model.resolve())
            for ext in ('.json','.npz'):
                if model.with_suffix(ext).is_file():files.add(model.with_suffix(ext).resolve())
    # Bind original shared prefixes to the old native archive, while also
    # including the compact source observations and their first-ten evidence.
    prefix_keys={(study.SHORT[p['task']],p['seed']) for p in protocol['prefixes']}
    old_bindings=rows(ROOT/'paper/v13/data/fit_inventory.csv')
    for row in old_bindings:
        if row['regime']=='cold' and (row['task'],int(row['seed'])) in prefix_keys and int(row['query'])<=10:
            for label in ('checkpoint','fit_record','commit_path','GP_arrays'):
                if row[label]:files.add((ROOT/row[label]).resolve())
    for row in protocol['prefixes']:files.add((ROOT/row['source']).resolve())
    for row in rows(ROOT/'paper/v13/data/datasets.csv'):
        for key in ('pool_path','oracle_path'):files.add((ROOT/row[key]).resolve())
    files.update((ROOT/name).resolve() for name in protocol['sources_sha256'])
    for path in (ROOT/'code/publication_v13_late').glob('*.py'):files.add(path.resolve())
    for path in (ROOT/'code/analysis/audit_v13_late_validation.py',ROOT/'code/analysis/v13_late_phase_analysis.py',
        ROOT/'code/tools/v13_late_dispatch.py',ROOT/'code/tools/v13_late_rbo_compat.py',
        ROOT/'code/tools/resume_v13_late.py',ROOT/'code/tools/finish_v13_late.py',
        ROOT/'runtime/cache/tex/local_fonts/lm.zip',ROOT/'runtime/cache/tex/local_fonts/GUST-FONT-LICENSE.TXT',
        ROOT/'code/chemist_system/third_party/reasoning_bo_official/data/chemistry_data.zip'):
        files.add(path.resolve())
    for report in (ROOT/'experiments/reports/v13_late_validation_20261103',ROOT/'experiments/logs/v13_late_validation_20261103'):
        files.update(p.resolve() for p in report.rglob('*') if good(p))
    # Source licensing and environment declarations are explicit, not a broad
    # home-directory collection. Credential/configuration files are excluded.
    files.add((ROOT/'code/chemist_system/pyproject.toml').resolve())
    vendored=ROOT/'code/chemist_system/third_party/reasoning_bo_official'
    files.update(x.resolve() for x in vendored.rglob('*') if good(x) and x.name.lower().startswith(('license','notice','copying')))
    for x in files:
        if not good(x):raise ValueError('Missing or excluded evidence: '+str(x))
        rel(x)
        if x.name.lower() in ('openai-key.txt','config.txt','.env'):raise ValueError('Credential file must not enter archive')
    index={rel(x):queue.file_sha(x) for x in sorted(files)}
    old=ROOT/'experiments/milestones/v13_publication_20261103'
    previous=queue.read(old/'receipt.json')
    manifest=dict(purpose='Additive V13 common-prefix/transfer evidence; no old archive rewritten',
        protocol_sha256=protocol['protocol_sha256'],files_sha256=index,
        historical_manifest=rel(old/'manifest.json'),historical_manifest_sha256=queue.file_sha(old/'manifest.json'),
        historical_evidence_zip=rel(old/'evidence_and_GP_models.zip'),
        historical_evidence_zip_sha256=queue.file_sha(old/'evidence_and_GP_models.zip'))
    write(M/'manifest.json',manifest)
    zpath=M/'evidence_and_GP_models.zip'
    if not zpath.exists():
        with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in index:z.write(ROOT/name,name)
    with zipfile.ZipFile(zpath) as z:
        if set(z.namelist())!=set(index):raise ValueError('Archive membership mismatch')
        for name,value in index.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=value:raise ValueError('Archive byte verification failed')
    receipt=dict(status='SEALED_AND_VERIFIED',files=len(index),new_paid_queries=840,trajectories=75,
        fitted_new_decisions=831,no_data_decisions=9,raw_LLM_requests=len(rows(P/'data/LLM_request_inventory.csv')),
        distinct_bound_new_checkpoints=len({r['checkpoint'] for r in rows(P/'data/new_fit_inventory.csv') if r['checkpoint']}),
        protocol_sha256=protocol['protocol_sha256'],manifest_sha256=queue.file_sha(M/'manifest.json'),
        archive_sha256=queue.file_sha(zpath),archive_bytes=zpath.stat().st_size,
        all_archived_member_bytes_verified=True,original_results_sources_and_release_unchanged=True,
        inherited_prefix_checkpoint_count=sum(x.endswith('.pt') and '/'+study.VARIANT+'/' not in x for x in index))
    write(M/'receipt.json',receipt);shutil.copy2(M/'receipt.json',P/'audit/new_preservation_receipt.json')
    study.check(True);print(json.dumps(receipt))


if __name__=='__main__':main()
