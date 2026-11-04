"""Seal completed V9 evidence without editing sources, runs, or historical controls.

The archive contains explicitly selected evidence, not a recursive workspace backup.
Credentials and the Python environment are excluded. Oracle files are copied as
opaque bytes for custody; this utility never loads candidate objective arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[2]
RUN = Path('experiments/runs/knowledge_loop_20261101/v9_kf_rank')
REPORT = Path('experiments/reports/knowledge_loop_20261101')
DEST = Path('experiments/milestones/v9_kf_rank_20261101')
FORBIDDEN = {'config.txt', 'openai-key.txt', '.env'}


def digest_file(path):
    h = sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def selected_files(root):
    protocol = read(root / RUN / 'protocol.json')
    paths = set(protocol['source_sha256']) | set(protocol['protected_V8_sha256'])
    protected = dict(protocol['source_sha256'], **protocol['protected_V8_sha256'])
    for item in protocol['inputs']:
        paths.update(item['files_sha256'])
        protected.update(item['files_sha256'])
    for name, expected in protected.items():
        if digest_file(root / name) != expected:
            raise ValueError('Frozen evidence changed: ' + name)
    protected[protocol['design_path']] = protocol['design_sha256']
    if digest_file(root / protocol['design_path']) != protocol['design_sha256']:
        raise ValueError('Frozen V9 design changed')
    for folder in (root / RUN, root / 'runtime/knowledge_loop/snapshots/v9_kf_rank'):
        paths.update(p.relative_to(root).as_posix() for p in folder.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.lock')
    paths.update(p.relative_to(root).as_posix() for p in (root / REPORT).glob('v9*') if p.is_file())
    paths.update(p.relative_to(root).as_posix() for p in
                 (root / 'code/experiment_tools/knowledge_rank_v9/tests').glob('*.py'))
    paths.update(['docs/knowledge_rank_v9_design_20261101.txt',
                  'docs/knowledge_rank_v9_run_20261101.txt', 'requirements-bo.txt',
                  'code/tools/audit_knowledge_rank_v9.py',
                  'code/tools/test_audit_knowledge_rank_v9.py',
                  'code/tools/export_knowledge_rank_v9_diagnostics.py',
                  'code/tools/seal_knowledge_rank_v9.py'])
    completion = read(root / REPORT / 'v9_completion_receipt.json')
    for name, expected in completion['files_sha256'].items():
        if digest_file(root / REPORT / name) != expected:
            raise ValueError('Delivered artifact changed: ' + name)
    for task in protocol['tasks']:
        for arm in protocol['arms']:
            state = read(root / RUN / 'runs' / f'{task}_{arm}' / 'state.json')
            if state['status'] != 'COMPLETE' or len(state['history']) != 20 or len(state['decisions']) != 16:
                raise ValueError('Incomplete trajectory: ' + task + '/' + arm)
    for name in paths:
        path = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.name.lower() in FORBIDDEN
                or not path.resolve().is_relative_to(root.resolve()) or not path.is_file()):
            raise ValueError('Unsafe/missing archive member: ' + name)
    return sorted(paths), protocol, completion


def verify(root=ROOT, *, compare_live=True):
    folder = root / DEST
    receipt = read(folder / 'seal_receipt.json')
    manifest_path, archive_path = folder / 'manifest.json', folder / 'evidence.zip'
    if digest_file(manifest_path) != receipt['manifest_sha256']:
        raise ValueError('Manifest changed')
    if digest_file(archive_path) != receipt['archive_sha256']:
        raise ValueError('Archive changed')
    manifest = read(manifest_path)
    with zipfile.ZipFile(archive_path, 'r') as archive:
        if set(archive.namelist()) != set(manifest['files']) | {'SEAL_MANIFEST.json'}:
            raise ValueError('Archive inventory changed')
        if archive.read('SEAL_MANIFEST.json') != manifest_path.read_bytes():
            raise ValueError('Embedded manifest changed')
        for name, meta in manifest['files'].items():
            data = archive.read(name)
            if len(data) != meta['bytes'] or sha256(data).hexdigest() != meta['sha256']:
                raise ValueError('Archived member changed: ' + name)
    drift = []
    if compare_live:
        drift = [name for name, meta in manifest['files'].items()
                 if not (root / name).is_file() or digest_file(root / name) != meta['sha256']]
    return dict(status='PASS_ARCHIVE_AND_LIVE' if compare_live and not drift else
                'PASS_ARCHIVE_LIVE_DRIFT' if compare_live else 'PASS_ARCHIVE',
                files=len(manifest['files']), archive_sha256=receipt['archive_sha256'],
                archive_bytes=archive_path.stat().st_size, live_drift=drift,
                scientific_majority_goal_achieved=False)


def seal(root=ROOT):
    folder = root / DEST
    if (folder / 'seal_receipt.json').exists():
        return verify(root)
    if folder.exists():
        raise ValueError('Incomplete seal exists; inspect it, never silently overwrite evidence')
    paths, protocol, completion = selected_files(root)
    files = {name: dict(bytes=(root / name).stat().st_size, sha256=digest_file(root / name)) for name in paths}
    milestone = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
                               capture_output=True, text=True).stdout.strip()
    manifest = dict(status='SEALED_REFERENCE_BASELINE', created_utc=datetime.now(timezone.utc).isoformat(),
                    experiment_core_commit=protocol['code_milestone'], delivery_commit=milestone,
                    protocol_sha256=protocol['protocol_sha256'], tasks=protocol['tasks'],
                    arms=protocol['arms'], seed=0, initial=4, queries=16,
                    current_runs=24, readonly_V8_controls=8,
                    V9_vs_BO=dict(wins=2, ties=4, losses=2), scientific_majority_goal_achieved=False,
                    completion=completion,
                    environment={name: importlib.metadata.version(name)
                                 for name in ('torch', 'botorch', 'gpytorch', 'numpy', 'scipy')},
                    custody='Content-addressed copy; original files unchanged; no OS immutable-bit claim.',
                    exclusions=['credentials', 'Python environment', 'lock files', '__pycache__'],
                    oracle_handling='Copied opaque bytes only; no new target queried or decoded.', files=files)
    folder.mkdir(parents=True, exist_ok=False)
    blob = encode(manifest)
    (folder / 'manifest.json').write_bytes(blob)
    archive_path = folder / 'evidence.zip'
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, meta in files.items():
            data = (root / name).read_bytes()
            if sha256(data).hexdigest() != meta['sha256']:
                raise ValueError('Source changed during seal: ' + name)
            archive.writestr(name, data)
        archive.writestr('SEAL_MANIFEST.json', blob)
    receipt = dict(status='SEALED', manifest_sha256=sha256(blob).hexdigest(),
                   archive_sha256=digest_file(archive_path), archive_bytes=archive_path.stat().st_size,
                   files=len(files), original_bytes=sum(m['bytes'] for m in files.values()),
                   source_commit=milestone, protocol_sha256=protocol['protocol_sha256'])
    (folder / 'seal_receipt.json').write_bytes(encode(receipt))
    result = verify(root)
    (folder / 'verification.json').write_bytes(encode(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['seal', 'verify'])
    parser.add_argument('--archive-only', action='store_true', help='Verify bundle without requiring unchanged live files')
    args = parser.parse_args()
    value = seal() if args.action == 'seal' else verify(compare_live=not args.archive_only)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if value['live_drift']:
        raise SystemExit(1)
