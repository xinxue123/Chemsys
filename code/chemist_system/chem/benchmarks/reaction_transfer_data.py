"""Frozen empirical Perera/Doyle pools; no surrogate or missing-row imputation."""
from collections import Counter, OrderedDict
from hashlib import sha256
from itertools import product
from pathlib import Path
import csv
import json
import math

import numpy as np

from .hte_workbook import table, sheets
from .suzuki_domain import digest

ROOT = Path('data/benchmarks/reaction_transfer')
OUT = Path('research/reaction_transfer')
SPECS = {
    'Perera_Suzuki': {'file': 'Perera_Suzuki.xlsx', 'sheet': 'Sheet1',
        'fixed': ['Reactant_1_Name', 'Reactant_2_Name'],
        'axes': {'ligand': 'Ligand_Short_Hand', 'base': 'Reagent_1_Short_Hand', 'solvent': 'Solvent_1_Short_Hand'},
        'target': 'Product_Yield_PCT_Area_UV', 'expected_rows': 5760,
        'unit': 'UV chromatographic product area percent', 'reaction': 'Suzuki-Miyaura'},
    'Doyle_Buchwald_Hartwig': {'file': 'Doyle_Buchwald_Hartwig.xlsx', 'sheet': 'Plates1-3',
        'fixed': ['Aryl halide'], 'axes': {'ligand': 'Ligand', 'base': 'Base', 'additive': 'Additive'},
        'target': 'Output', 'expected_rows': 3955, 'unit': 'reported experimental yield percent',
        'reaction': 'Buchwald-Hartwig'},
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def canonical(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('missing required nominal label; do not silently impute')
    return value.strip()


def filehash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def stress_knowledge(pool, values, confidence=.7):
    """Outcome-aware stress construction, isolated from policy/LLM inputs.

    Correct includes a fixed reference optimum, with quarter-sized level sets.
    Wrong-category replaces only ligand by low-marginal levels excluding the
    reference ligand; mixed also replaces base. These are artificial stress
    priors, not chemistry discoveries or estimates of real LLM error rates.
    """
    ids, axes = pool['ids'], pool['axes']
    peak = min(range(len(ids)), key=lambda i: (-float(values[i]), ids[i]))
    good, bad = {}, {}
    for axis in axes:
        levels = pool['levels'][axis]
        k = max(1, math.ceil(len(levels)/4))
        means = {label: float(np.mean([values[i] for i, p in enumerate(pool['points']) if p[axis] == label])) for label in levels}
        reference = pool['points'][peak][axis]
        good[axis] = sorted([reference] + [v for v in sorted(levels, key=lambda v: (-means[v], v)) if v != reference][:k-1])
        bad[axis] = sorted([v for v in sorted(levels, key=lambda v: (means[v], v)) if v != reference][:k])
    def knowledge(wrong_axes):
        return {'confidence': confidence, 'clauses': [
            {'id': f'H{i+1}', 'axis': axis, 'allowed': bad[axis] if axis in wrong_axes else good[axis]}
            for i, axis in enumerate(axes)]}
    return {'correct': knowledge(set()), 'wrong_category': knowledge({axes[0]}),
            'mixed': knowledge(set(axes[:2]))}, {
        'reference_candidate_id': ids[peak], 'sets_per_axis': {a: len(good[a]) for a in axes},
        'definition': 'Reference-optimum-containing preferences; wrong ligand/base use low marginal sets excluding that reference level.',
        'limit': 'Wrong sets may still contain other high-yield points; no claim that chemical effects are globally monotone.'}


def prepare():
    source = read(ROOT / 'raw/source_manifest.json')
    for row in source['files']:
        assert filehash(ROOT / 'raw' / row['name']) == row['sha256']
    all_profiles, catalog = {}, []
    for dataset, spec in SPECS.items():
        source_rows = table(ROOT / 'raw' / spec['file'], spec['sheet'])
        assert len(source_rows) == spec['expected_rows']
        axes = list(spec['axes'])
        normalized, groups, changed = [], OrderedDict(), Counter()
        for row in source_rows:
            point = {axis: canonical(row[field]) for axis, field in spec['axes'].items()}
            fixed = {key: canonical(row[key]) for key in spec['fixed']}
            y = float(row[spec['target']])
            assert np.isfinite(y) and 0 <= y <= 100
            for field in (*spec['fixed'], *spec['axes'].values()):
                changed[field] += row[field] != canonical(row[field])
            key = tuple(fixed[k] for k in spec['fixed'])
            record = {'point': point, 'fixed': fixed, 'yield_fraction': y/100,
                'source_excel_row': row['source_excel_row'],
                'source_reaction_no': row.get('Reaction_No')}
            groups.setdefault(key, []).append(record)
            normalized.append(record)
        condition_keys = [tuple(r['fixed'].values()) + tuple(r['point'][a] for a in axes) for r in normalized]
        assert len(set(condition_keys)) == len(condition_keys)
        profiles = []
        for i, records in enumerate(groups.values(), 1):
            task = f'{dataset}_task{i:02d}'
            levels = {axis: sorted({r['point'][axis] for r in records}) for axis in axes}
            # Canonical feature order; the selected TASK is source-first, not yield-ranked.
            records = sorted(records, key=lambda r: tuple(r['point'][a] for a in axes))
            ids = ['p' + digest({'point': r['point']})[:16] for r in records]
            assert len(set(ids)) == len(ids)
            initial_indices = np.random.default_rng(0).choice(len(records), 4, replace=False).tolist()
            pool = {'task_id': task, 'dataset': dataset, 'reaction_class': spec['reaction'],
                'axes': axes, 'levels': levels, 'fixed_substrates': records[0]['fixed'],
                'ids': ids, 'points': [r['point'] for r in records],
                'initial_ids': [ids[j] for j in initial_indices], 'initial_seed': 0,
                'objective_unit': spec['unit'], 'candidate_count': len(records),
                'source_sheet': spec['sheet'], 'source_file_sha256': filehash(ROOT / 'raw' / spec['file'])}
            folder = ROOT / 'tasks' / task
            write(folder / 'pool.json', pool)
            values = np.asarray([r['yield_fraction'] for r in records], dtype=float)
            provenance = [{'candidate_id': k, 'source_excel_row': r['source_excel_row'],
                           'source_reaction_no': r['source_reaction_no']} for k, r in zip(ids, records)]
            if (folder / 'oracle.npz').exists():
                with np.load(folder / 'oracle.npz') as saved:
                    assert np.array_equal(saved['values'], values)
            else:
                np.savez_compressed(folder / 'oracle.npz', values=values)
            write(folder / 'row_provenance.json', provenance)
            full = set(product(*(levels[a] for a in axes)))
            available = {tuple(r['point'][a] for a in axes) for r in records}
            profile = {'task_id': task, 'source_group_order': i, 'fixed_substrates': pool['fixed_substrates'],
                'candidate_count': len(records), 'category_counts': {a: len(levels[a]) for a in axes},
                'cartesian_size': len(full), 'missing_combinations': len(full-available),
                'missing_labels': [dict(zip(axes, v)) for v in sorted(full-available)],
                'metadata_only_selection': i == 1}
            profiles.append(profile)
            catalog.append({'task_id': task, 'dataset': dataset, 'source_group_order': i,
                'candidate_count': len(records), 'fixed_substrates': pool['fixed_substrates'],
                'pool_sha256': filehash(folder / 'pool.json'), 'oracle_sha256': filehash(folder / 'oracle.npz')})
        all_profiles[dataset] = {'rows': len(normalized), 'source_sheet': spec['sheet'],
            'objective': spec['target'], 'unit': spec['unit'], 'duplicate_condition_rows': 0,
            'missing_required_labels': 0, 'yield_min_percent': min(r['yield_fraction'] for r in normalized)*100,
            'yield_max_percent': max(r['yield_fraction'] for r in normalized)*100,
            'whitespace_normalized_cells': dict(changed), 'task_count': len(groups), 'tasks': profiles}
        # Retain a human-readable canonical table; it is evaluation data, never policy input.
        path = ROOT / f'{dataset}_observed_table.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=[*spec['fixed'], *axes, 'observed_yield_percent', 'source_excel_row'])
            writer.writeheader()
            writer.writerows({**r['fixed'], **r['point'], 'observed_yield_percent': r['yield_fraction']*100,
                             'source_excel_row': r['source_excel_row']} for r in normalized)
    # Verify that the other sheets are permutations of the same Doyle experiments.
    path = ROOT / 'raw' / SPECS['Doyle_Buchwald_Hartwig']['file']
    canonical_rows = table(path, 'Plates1-3')
    fields = ['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output']
    expected = Counter(tuple(row[k] for k in fields) for row in canonical_rows)
    variants = []
    for name, rows in sheets(path):
        current = Counter(tuple(row['values'][:5]) for row in rows[1:])
        assert current == expected, name
        variants.append(name)
    quality = {'status': 'PASS', 'datasets': all_profiles,
        'Doyle_equivalent_sheets': variants, 'sheets_concatenated': False,
        'post_outcome_features_excluded': ['Product_Yield_PCT_Area_UV', 'Product_Yield_Mass_Ion_Count', 'Output'],
        'no_ligand_and_no_base_controls': 'Literal None labels are retained as categories, not missing records.',
        'Perera_missing_ligand_eq': '480 rows are no-ligand controls; the unused amount is not a GP feature.',
        'Perera_solvent_versions': 'Preserve the six raw global labels; each fixed-substrate task has four levels. No collapse of THF/THF_V2 or MeOH/water blends.',
        'source_manifest_sha256': filehash(ROOT / 'raw/source_manifest.json'),
        'task_selection_rule': 'First fixed-substrate group in each canonical source sheet, before inspecting group yields; pilot only.',
        'scope': 'All30 fixed-substrate pools are exported; two source-first tasks are selected for the initial small experiment.'}
    write(ROOT / 'task_catalog.json', catalog)
    selected = [v for v in catalog if v['source_group_order'] == 1]
    write(ROOT / 'selected_tasks.json', {'selection_rule': quality['task_selection_rule'], 'tasks': selected})
    write(OUT / 'data_quality.json', quality)
    return {'status': 'PASS', 'datasets': {k: {'rows': v['rows'], 'tasks': v['task_count']} for k, v in all_profiles.items()},
            'selected': [{k: v[k] for k in ('task_id', 'candidate_count', 'fixed_substrates')} for v in selected]}


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False))
