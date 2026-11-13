"""Paper-scale empirical domains, with outcomes isolated from decision inputs."""
from collections import Counter
from itertools import product
from pathlib import Path
import csv
import io
import json
import zipfile

import numpy as np

from .reaction_transfer_data import read, write, filehash
from .suzuki_domain import digest

UPSTREAM = Path('third_party/reasoning_bo_official')
ROOT = Path('data/benchmarks/reasoning_bo')
OUT = Path('research/reasoning_bo_comparison')
SPECS = {
    'Perera_Suzuki': ('suzuki.csv', 'suzuki_config.json', 3696, 'Suzuki-Miyaura'),
    'Doyle_Buchwald_Hartwig': ('Buchwald_Hartwig.csv', 'buchwald_config.json', 3955, 'Buchwald-Hartwig'),
}


def prepare():
    archive = UPSTREAM/'data/chemistry_data.zip'
    provenance = read(UPSTREAM/'SOURCE_PROVENANCE.json')
    profiles = []
    with zipfile.ZipFile(archive) as handle:
        for task, (member, config_name, count, reaction) in SPECS.items():
            source_rows = list(csv.DictReader(io.StringIO(handle.read(member).decode('utf-8-sig'))))
            rows = [{('Arylhalide' if k == 'Aryl halide' else k.lstrip('\ufeff')):v
                for k,v in row.items()} for row in source_rows]
            axes = [key for key in rows[0] if key not in ('yield','entry')]
            assert len(rows) == count
            assert all(set(row) == set(axes+['yield','entry']) for row in rows)
            for row in rows:
                assert all(row[a] and row[a] == row[a].strip() for a in axes)
                assert np.isfinite(float(row['yield'])) and 0 <= float(row['yield']) <= 100
            rows.sort(key=lambda r: tuple(r[a] for a in axes))
            points = [{a: row[a] for a in axes} for row in rows]
            assert len({tuple(row[a] for a in axes) for row in rows}) == count
            levels = {a: sorted({row[a] for row in rows}) for a in axes}
            pool = {'task_id': task, 'reaction_class': reaction, 'axes': axes,
                'ids': [f'p{i:05d}' for i in range(count)], 'points': points,
                'levels': levels, 'fixed_substrates': {},
                'objective_unit': 'UV chromatographic product area percent' if task == 'Perera_Suzuki'
                    else 'reported experimental reaction yield percent',
                'domain_definition': 'Entire author-bundled measured table; substrates ARE optimization variables.',
                'source_archive_sha256': filehash(archive), 'source_member': member}
            folder = ROOT/task
            write(folder/'pool.json', pool)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/'oracle.npz', values=np.array([float(r['yield'])/100 for r in rows]))
            config = read(UPSTREAM/'src/config'/config_name)
            original_axes = {r['display_name']: r for r in config['parameter_definitions']}
            if 'Aryl halide' in original_axes:
                original_axes['Arylhalide'] = original_axes.pop('Aryl halide')
            config_changes = []
            config['parameter_definitions'] = []
            for axis in axes:
                item = original_axes[axis].copy()
                old = set(item['bounds'])
                config_changes.append({'axis': axis, 'added': sorted(set(levels[axis])-old),
                    'removed': sorted(old-set(levels[axis]))})
                item.update(display_name=axis, bounds=levels[axis])
                config['parameter_definitions'].append(item)
            # Neither input file establishes isolated yield; do not amplify that source claim.
            config['target']['description'] = pool['objective_unit']+'; maximize the measured table value'
            config['constraint'] += (' Select only the supplied nominal choices, with exact spellings. '
                'Only measured combinations are executable. No temperature/time or other unlisted variable exists. '
                'All categories are unordered identifiers. Do not recall benchmark outcomes.')
            write(folder/'compass.json', config)
            complete = int(np.prod([len(v) for v in levels.values()]))
            profiles.append({'task': task, 'rows': count, 'axes': {a: len(v) for a,v in levels.items()},
                'missing_cartesian_combinations': complete-count, 'duplicate_combinations': 0,
                'missing_required_values': 0, 'target_range_percent': [min(float(r['yield']) for r in rows),
                    max(float(r['yield']) for r in rows)], 'config_changes': config_changes,
                'excluded_input_columns': ['entry','yield'],
                'pool_sha256': filehash(folder/'pool.json'), 'oracle_sha256': filehash(folder/'oracle.npz'),
                'compass_sha256': filehash(folder/'compass.json')})
    result = {'status': 'PASS', 'source_archive_sha256': filehash(archive),
        'official_commit': '4df3cd64abd1be1aab533627578523610f89788c', 'tasks': profiles,
        'original_workbooks': 'Retained separately under reaction_transfer/raw; this experiment uses the paper author-bundled subset.',
        'missing_data_policy': 'No invented or imputed experiments; never concatenate reordered Doyle workbook sheets.'}
    write(OUT/'data_quality.json', result)
    return result


def load_pool(task):
    return read(ROOT/task/'pool.json')


def reveal(task, ids):
    """Evaluation-layer only; call AFTER saving the complete simultaneous batch."""
    pool = load_pool(task)
    index = {key: i for i,key in enumerate(pool['ids'])}
    values = np.load(ROOT/task/'oracle.npz', allow_pickle=False)['values']
    assert len(set(ids)) == len(ids)
    return [{'candidate_id': key, 'y': float(values[index[key]])} for key in ids]
