"""Seal the completed warm-start evidence before a separate cold-start study."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/experiment_tools'), str(ROOT / 'code/chemist_system')]
from v13_initialization_seeds import study
from state_split_v13_experiment import transport


def main():
    protocol = study.check(ROOT, True)
    report = ROOT / 'experiments/reports/v13_seed123_20261102'
    audit = transport.read(report / 'audit.json')
    if (audit['status'], audit['complete_trajectories'], audit['queries_replayed']) != ('PASS_COMPLETE', 24, 384):
        raise ValueError('Archive requires the fully verified completed study')
    target = ROOT / 'experiments/milestones/v13_warmstart_20261102'
    receipt_path = target / 'receipt.json'
    if receipt_path.exists():
        receipt = transport.read(receipt_path)
        for name, sha in receipt['artifact_sha256'].items():
            if transport.file_sha(target / name) != sha:
                raise ValueError('Milestone archive changed')
        print(json.dumps(receipt))
        return
    files = set()
    folders = [study.directory(ROOT), study.FROZEN_BASE(ROOT), report,
               ROOT / 'experiments/reports/v13_state_split_experiment_20261102']
    for folder in folders:
        if not folder.is_dir():
            raise ValueError('Missing completed evidence directory: ' + str(folder))
        for path in folder.rglob('*'):
            if path.is_file() and not any(p in ('cache', '__pycache__') for p in path.relative_to(folder).parts):
                files.add(path.resolve())
    for name in protocol['sources_sha256']:
        files.add((ROOT / name).resolve())
    for name in protocol['protected_sha256']:
        files.add((ROOT / name).resolve())
    for name in ('v13_multiseed_results.py', 'v13_multiseed_dispatch.py',
                 'v13_multiseed_seal_windows.py', 'v13_results.py', 'v13_report_zh.py',
                 'archive_v13_warmstart.py'):
        files.add(ROOT / 'code/tools' / name)
    # The raw agent packets are already committed in request folders. Include
    # exported packets as well so transport receipts can be checked in isolation.
    for folder in (study.directory(ROOT), study.FROZEN_BASE(ROOT)):
        for path in folder.rglob('export.json'):
            value = transport.read(path)
            for key in ('input', 'output'):
                exported = (ROOT / value[key]).resolve()
                if exported.is_file():
                    files.add(exported)
    names = {p.relative_to(ROOT).as_posix(): p for p in files}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Frozen 4+16 evidence; cold-start experiments must use a separate namespace',
        scientific_status='Execution verified; no stable overall performance advantage established',
        protocol_sha256=protocol['protocol_sha256'],
        file_count=len(names), byte_count=sum(p.stat().st_size for p in names.values()),
        files_sha256={name: transport.file_sha(path) for name, path in sorted(names.items())},
        omitted='Reusable native model caches; original caches and all historical directories remain in place')
    target.mkdir(parents=True, exist_ok=True)
    transport.write_once(target / 'manifest.json', manifest)
    archive = target / 'evidence.zip'
    temporary = target / 'evidence.zip.partial'
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, path in sorted(names.items()):
            if transport.file_sha(path) != manifest['files_sha256'][name]:
                raise ValueError('Source changed during archiving')
            z.write(path, name)
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != set(names):
            raise ValueError('Incomplete archive')
        for name in z.namelist():
            if hashlib.sha256(z.read(name)).hexdigest() != manifest['files_sha256'][name]:
                raise ValueError('Archived bytes failed verification')
    receipt = dict(status='SEALED_AND_VERIFIED', file_count=manifest['file_count'],
        archive_bytes=archive.stat().st_size, original_files_modified=False,
        artifact_sha256={name: transport.file_sha(target / name) for name in ('manifest.json', 'evidence.zip')})
    transport.write_once(receipt_path, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
