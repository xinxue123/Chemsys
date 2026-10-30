"""Read-only delivery checks and an artifact inventory for Chapter 5."""
from __future__ import annotations
import ast
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .evaluation import engine_fingerprint
from .replay import write_json_atomic


def verify_delivery(project_directory):
    root = Path(project_directory).resolve()
    research = root / 'research/chapter5'
    engine = engine_fingerprint()
    suites, counters = {}, {}
    inventory = {}

    def record(path):
        key = path.relative_to(root).as_posix()
        inventory[key] = {'sha256': sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

    for name in ('math_development', 'math_test', 'summit_development', 'summit_test',
                 'assistant_development', 'assistant_test'):
        folder = root / 'data/runs/chapter5' / name
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        audit = json.loads((folder / 'independent_audit.json').read_text(encoding='utf-8'))
        if (audit['status'] != 'PASS' or audit['engine_sha256'] != engine
                or audit['manifest_sha256'] != sha256((folder / 'manifest.json').read_bytes()).hexdigest()
                or audit['auditor_sha256'] != sha256((root / 'chem/decision/reporting.py').read_bytes()).hexdigest()):
            raise ValueError(f'audit is missing, stale or invalid: {name}')
        if audit['counters']['campaigns'] != manifest['planned_campaigns']:
            raise ValueError(f'incomplete campaign population: {name}')
        for filename, expected in audit['artifact_hashes'].items():
            if sha256((folder / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f'raw evidence changed after audit: {name}/{filename}')
        if name.endswith('_test'):
            freeze = json.loads((folder / 'freeze_receipt.json').read_text(encoding='utf-8'))
            if freeze['engine_sha256'] != engine or freeze['manifest_sha256'] != audit['manifest_sha256']:
                raise ValueError(f'frozen test definition changed: {name}')
            for key, value in audit['counters'].items():
                counters[key] = counters.get(key, 0) + value
        suites[name] = {'campaigns': audit['counters']['campaigns'],
                        'stopped': sum(r['status'] == 'STOPPED' for r in audit['rows']),
                        'independent_b4_decisions': audit['counters']['independent_b4_exhaustive_decisions'],
                        'independent_gp_refits': len(audit['independent_gp_checks']),
                        'audit_sha256': sha256((folder / 'independent_audit.json').read_bytes()).hexdigest()}
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                record(path)

    environment_path = research / 'environment_reproduction.json'
    environment = json.loads(environment_path.read_text(encoding='utf-8'))
    lock = research / 'requirements-research-windows-py310.lock'
    if (environment['status'] != 'PASS' or environment['inherits_system_site_packages']
            or environment['dependency_lock_sha256'] != sha256(lock.read_bytes()).hexdigest()
            or environment['engine_sha256'] != engine):
        raise ValueError('clean environment qualification is stale or invalid')
    locked = []
    for line in lock.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, version = line.split('==')
        if importlib.metadata.version(name) != version:
            raise ValueError(f'locked installed distribution differs: {name}')
        locked.append(name)
    junit = root / 'data/runs/chapter5/clean_environment_tests.xml'
    suite = ET.parse(junit).find('.//testsuite')
    pytest = {k: int(suite.attrib[k]) for k in ('tests', 'failures', 'errors', 'skipped')}
    if pytest['tests'] != 62 or any(pytest[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('the complete clean-environment tests did not pass')
    if len(environment['independent_gp_checks']) != 26 or any(
            r['max_absolute_error'] != 0. for r in environment['independent_gp_checks']):
        raise ValueError('clean-environment numerical reproduction differs')

    python_files = [*sorted((root / 'chem').rglob('*.py')), *sorted((root / 'tests').rglob('*.py')),
                    *sorted((root / 'scripts').glob('chapter5_*.py'))]
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        record(path)
    linked_files = 0
    for doc in sorted((root / 'docs').glob('chapter5*.md')):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if target.startswith(('https://', 'http://', '#')):
                continue
            target = target.strip('<>').split('#', 1)[0]
            path = Path(target)
            if not path.is_absolute():
                path = doc.parent / path
            if not path.exists():
                raise ValueError(f'broken chapter link: {doc.name} -> {target}')
            linked_files += 1
        record(doc)
    figures = json.loads((research / 'figure_receipt.json').read_text(encoding='utf-8'))
    for relative, expected in figures['source_sha256'].items():
        if sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('figure sources changed after rendering')
    for filename, expected in figures['figure_sha256'].items():
        if sha256((research / 'figures' / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'exported figure changed: {filename}')
    material = json.loads((research / 'material_interface_demo.json').read_text(encoding='utf-8'))
    if material['review_status'] != 'HOLD' or material['recommendations'] or material['actual_oracle_queries']:
        raise ValueError('unqualified material evidence was promoted to an experimental recommendation')
    assistant = json.loads((research / 'assistant_execution_receipt.json').read_text(encoding='utf-8'))
    test_assistant = assistant['tracks']['assistant_test']
    if (test_assistant['logical_events'] != 240 or test_assistant['unique_responses'] != 174
            or len(test_assistant['failed_events']) != counters['failed_llm_logical_events']):
        raise ValueError('assistant event accounting differs')
    for path in sorted(research.rglob('*')):
        if path.is_file() and path.name != 'validation_receipt.json':
            record(path)
    for path in (junit, root / 'README.md', root / 'pyproject.toml', root / '.gitignore'):
        record(path)
    manifest_path = root / 'data/runs/chapter5/delivery_artifacts.json'
    write_json_atomic(manifest_path, {'schema_version': 1, 'files': inventory,
                                    'scope': 'source, documents, six suites, predictions, requests, oracle caches and receipts; no environments or credentials'})
    return {'schema_version': 3, 'status': 'PASS', 'scope': 'Chapter 5 delivery and evidence integrity',
            'engine_sha256': engine, 'test_counters': counters, 'suites': suites,
            'test_campaigns': sum(v['campaigns'] for k, v in suites.items() if k.endswith('_test')),
            'development_campaigns': sum(v['campaigns'] for k, v in suites.items() if k.endswith('_development')),
            'test_stopped_campaigns': sum(v['stopped'] for k, v in suites.items() if k.endswith('_test')),
            'pytest': pytest, 'clean_environment_rebuild_qualified': True, 'locked_distributions': len(locked),
            'utf8_ast_checked_python_files': len(python_files), 'checked_local_document_links': linked_files,
            'exported_figures': 3, 'formats_per_figure': 3, 'png_visual_review': 'all inspected; paired-effect legend moved outside plot area',
            'artifact_inventory': manifest_path.relative_to(root).as_posix(),
            'artifact_inventory_sha256': sha256(manifest_path.read_bytes()).hexdigest(),
            'inventoried_files': len(inventory), 'inventoried_bytes': sum(v['bytes'] for v in inventory.values()),
            'scientific_superiority_over_B3_established': False,
            'material_scientific_replay_qualified': False, 'material_interface_status': 'HOLD',
            'llm_backend_snapshot_verifiable': False, 'llm_context_isolation': False,
            'interpretation': 'Implementation and planned local experiments delivered; negative results and evidence limits retained.'}
