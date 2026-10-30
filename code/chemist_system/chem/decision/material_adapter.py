"""Ce-Al-Ge task-interface demonstration with unresolved provenance held."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

from .hypothesis import ParameterSpec
from .replay import SynthesisTask, audit_cealge_dataset


def material_interface_demo(project_directory):
    root = Path(project_directory).resolve()
    source = root / 'data/ce_al_ge/datasets/arcmelt_dataset_rietveld.json'
    audit = audit_cealge_dataset(source)
    eligible = [row for row in audit['records'] if row['provisionally_eligible']]
    candidates = tuple({f'atom_fraction_{element}': row['recomputed_atom_fraction_assuming_mass_ratio'][element]
                        for element in ('Ce', 'Al', 'Ge')} for row in eligible)
    if not all(abs(sum(row.values()) - 1.) <= 1e-12 for row in candidates):
        raise ValueError('nominal composition conversion violated the ternary simplex')
    structure_path = root / 'data/ce_al_ge/arc_melt/cifs/CeAl2Ge2.json'
    old_plan_path = root / 'docs/ceal2ge2_arc_melting_2026-05-21.md'
    structure = json.loads(structure_path.read_text(encoding='utf-8'))
    old_plan = old_plan_path.read_text(encoding='utf-8')
    task = SynthesisTask(
        'CeAlGe_interface_only', 'UNVERIFIED:CeAl2Ge2_structure_identity',
        'UNVERIFIED:historical_arc_melt_processing',
        tuple(ParameterSpec(f'atom_fraction_{el}', 'atomic_fraction', (0., 1.)) for el in ('Ce', 'Al', 'Ge')),
        tuple(row['candidate_id'] for row in eligible), candidates,
        ({'id': 'local_dataset_audit', 'text': 'Numeric screening only; source mass basis and common processing unverified.',
          'source_sha256': audit['source_sha256']},),
        description='Interface demonstration; HOLD before any optimization or experimental recommendation.',
        structure_reference=None)
    evidence_paths = [source, structure_path, old_plan_path, root / 'chem/xrd/arcmelt.py']
    return {'schema_version': 1, 'execution_status': 'OK', 'review_status': 'HOLD',
            'recommendation_status': 'needs_validation', 'recommendations': [],
            'scientific_material_optimization_performed': False,
            'raw_population': audit['raw_record_count'], 'numeric_non_DSC_candidates': len(eligible),
            'unknown_results': audit['unknown_outcome_count'], 'raw_record_audit': audit['records'],
            'reserved_replay_budget': {'initial': 2, 'additional_maximum': 6},
            'actual_oracle_queries': 0, 'actual_llm_attempts': 0,
            'required_provenance': audit['metadata_requiring_source_verification'],
            'phase_identity_evidence': {
                'local_structure_metadata_space_group': structure.get('space_group'),
                'local_structure_source': structure.get('source'), 'source_id': structure.get('source_id'),
                'provisional_plan_states_I4_mmm': 'I4/mmm' in old_plan,
                'interpretation': 'Conflicting local descriptions require phase identity verification; neither is selected as experimental truth.'},
            'source_sha256': {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in evidence_paths},
            'task': task.to_dict(), 'task_sha256': task.fingerprint,
            'boundary': 'No 2+6 performance curve or synthesis-success claim is reported from unqualified material records.'}
