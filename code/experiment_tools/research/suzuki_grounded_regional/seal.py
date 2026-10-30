"""Seal the completed local experiment without copying caches or credentials."""
from hashlib import sha256
from pathlib import Path
import json

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE / 'data/runs/suzuki_grounded_regional'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def filehash(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def entry(path, base=BASE):
    return {'path': path.relative_to(base).as_posix(), 'bytes': path.stat().st_size, 'sha256': filehash(path)}


def main():
    protocol = read(OUT / 'protocol.json')
    validation = read(OUT / 'validation_receipt.json')
    result = read(OUT / 'results.json')
    analysis = read(OUT / 'analysis.json')
    assert validation['integrity_status'] == 'PASS' and validation['completed'] == 48
    assert validation['results_sha256'] == filehash(OUT / 'results.json')
    assert validation['report_source_sha256'] == filehash(BASE / 'chem/benchmarks/grounded_regional_suzuki_report.py')
    avg = result['summary']['primary']['all_conditions']
    assert abs(analysis['primary_pairs']['R2_conflict_fallback']['G3_original']['mean_delta'] -
        (avg['R2_conflict_fallback']['SearchAUC_mean'] - avg['G3_original']['SearchAUC_mean'])) < 1e-10
    source_paths = [Path(p) if Path(p).is_absolute() else BASE / p for p in protocol['scientific_sha256']]
    for path, expected in zip(source_paths, protocol['scientific_sha256'].values()):
        assert filehash(path) == expected
    source_paths += [BASE / 'chem/benchmarks/grounded_regional_suzuki_report.py',
                     BASE / 'docs/suzuki_grounded_regional.md']
    skip = {'cache', 'audit_cache', 'qualification_cache', '__pycache__'}
    paths = source_paths + [p for root in (ROOT, OUT) for p in root.rglob('*')
        if p.is_file() and not any(part in skip for part in p.parts)
        and p.name not in ('release_inventory.json', 'release_receipt.json')]
    trees = {}
    for name in ('cache', 'audit_cache', 'qualification_cache'):
        folder = ROOT / name
        rows = [entry(p, folder) for p in sorted(folder.rglob('*')) if p.is_file()]
        trees[name] = {'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
            'sha256': sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()}
    inventory = {'protocol_sha256': protocol['protocol_sha256'],
        'files': [entry(p) for p in sorted(set(paths))], 'cache_trees': trees}
    save(OUT / 'release_inventory.json', inventory)
    receipt = {'integrity_status': 'PASS', 'completed': 48, 'planned': 48,
        'added_queries': 768, 'logical_observations': 960, 'valid_LLM_refreshes': 141,
        'component_tests': validation['component_tests'], 'operator_compiles_in_loop': 0,
        'inventory_files': len(inventory['files']), 'cache_trees': trees,
        'inventory_sha256': filehash(OUT / 'release_inventory.json'),
        'validation_sha256': filehash(OUT / 'validation_receipt.json'),
        'results_sha256': filehash(OUT / 'results.json'),
        'visual_QA': ['all_conditions.png', 'case3_case4_assistant_initial.png'],
        'scientific_result': 'R2 improves mean SearchAUC over G3 by0.2853pp but remains below Gate_only by0.3710pp; R3 does not provide an overall increment. Single-seed descriptive evidence.'}
    save(OUT / 'release_receipt.json', receipt)
    print(json.dumps({'status': 'PASS', 'completed': 48, 'inventory_files': receipt['inventory_files'],
        'inventory_sha256': receipt['inventory_sha256'], 'cache_files': {k: v['files'] for k, v in trees.items()}}))


if __name__ == '__main__':
    main()
