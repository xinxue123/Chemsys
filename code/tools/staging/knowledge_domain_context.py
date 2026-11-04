"""Staged public-domain context for a future knowledge-loop protocol.

Pure metadata construction: no optimizer, GP, oracle, historical results,
knowledge-quality labels, or filesystem reads. Pass the existing input-only
pool for its exact categorical labels and executable input ranges. Returned
source paths are provenance, not an instruction to retrieve benchmark yields.
This module is deliberately outside the currently frozen chem implementation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re


TASKS = tuple(f'Reizman_Suzuki_Case_{i}' for i in range(1, 5)) + (
    'Perera_Suzuki', 'Doyle_Buchwald_Hartwig', 'SnAr', 'HPLC')
P0_CATALYSTS = ('P1-L1', 'P2-L1', 'P1-L2', 'P1-L3', 'P1-L4', 'P1-L5', 'P1-L6', 'P1-L7')


def _continuous(name, meaning, unit, bounds):
    return dict(name=name, kind='continuous', meaning=meaning, unit=unit,
                declared_input_bounds=list(bounds))


def _categorical(name, meaning, representation='SMILES'):
    return dict(name=name, kind='categorical', meaning=meaning, unit=None,
                representation=representation, ordered=False,
                identity_status='supplied molecular structure string' if representation == 'SMILES'
                else 'opaque experimental label; molecular identity unavailable')


def _base(task):
    if task.startswith('Reizman_Suzuki_Case_'):
        case = int(task[-1])
        catalyst = _categorical('catalyst', 'Experimental catalyst choice; source describes different ligands.', 'opaque label')
        catalyst['levels'] = list(P0_CATALYSTS)
        catalyst['interpretation_limit'] = (
            'P1/P2 and L1...L7 have no molecular identity mapping in the supplied domain. '
            'Do not infer structures, ligand names, electronic/steric properties or ordering from these tokens.')
        return dict(task=task, group='P0', reaction_or_process='Suzuki-Miyaura cross-coupling',
            description='Optimize residence time, reactor temperature, catalyst loading and catalyst identity for one fixed substrate case.',
            objective=dict(name='yield', unit='percent', direction='maximize',
                           meaning='Reaction yield reported by the fixed experimental emulator.'),
            substrate_scope=dict(substrates_are_optimization_variables=False,
                identity_status='not specified by the supplied public pool',
                statement='Cases 1–4 use different substrate sets; the catalyst choices are shared. Do not invent substrate identities.'),
            inputs=[_continuous('t_res', 'Reactor residence time', 's', (60., 600.)),
                    _continuous('temperature', 'Reactor temperature', 'degC', (30., 110.)),
                    _continuous('catalyst_loading', 'Catalyst loading', 'mol%', (.5, 2.5)), catalyst],
            source_references=[
                'code/chemist_system/chem/benchmarks/suzuki_domain.py',
                'runtime/environments/.venv-summit/Lib/site-packages/summit/benchmarks/experimental_emulator.py::ReizmanSuzukiEmulator.setup_domain',
                f'experiments/runs/suzuki_guarded/inputs/case{case}/pool.json'])
    if task == 'Perera_Suzuki':
        inputs = [
            _categorical('Electrophile_SMILES', 'Electrophilic substrate represented by the exact supplied molecular structure.'),
            _categorical('Nucleophile_SMILES', 'Organoboron coupling partner represented by the exact supplied molecular structure.'),
            _categorical('Ligand_SMILES', 'Ligand for the palladium-catalyzed coupling.'),
            _categorical('Base_SMILES', 'Base identity.'),
            _categorical('Solvent_SMILES', 'Solvent identity.'),
        ]
        return dict(task=task, group='P1', reaction_or_process='Suzuki-Miyaura cross-coupling',
            description='Palladium-catalyzed cross-coupling with categorical electrophile, organoboron partner, ligand, base and solvent choices.',
            objective=dict(name='product_UV_area', unit='percent', direction='maximize',
                meaning='UV chromatographic product area percentage; it is not asserted to be an isolated yield.'),
            substrate_scope=dict(substrates_are_optimization_variables=True,
                identity_status='variable substrate molecular structures are supplied in the input categories',
                statement='The full supplied measured table is optimized. Electrophile and nucleophile are decision variables; this is not a fixed-substrate subtask.'),
            inputs=inputs, source_references=[
                'code/chemist_system/third_party/reasoning_bo_official/src/config/suzuki_config.json',
                'code/chemist_system/chem/benchmarks/reasoning_bo_data.py',
                'experiments/datasets/reasoning_bo/Perera_Suzuki/pool.json'])
    if task == 'Doyle_Buchwald_Hartwig':
        return dict(task=task, group='P1', reaction_or_process='Buchwald-Hartwig C-N coupling (amination)',
            description='Choose ligand, additive, base and aryl halide from the supplied measured combinations for a Buchwald-Hartwig amination.',
            objective=dict(name='reaction_yield', unit='percent', direction='maximize',
                meaning='Reported experimental reaction yield percentage; no stronger isolated-yield claim is added.'),
            substrate_scope=dict(substrates_are_optimization_variables=True,
                identity_status='variable aryl-halide structures are supplied; the fixed amine identity is not specified in the public pool',
                statement='Arylhalide is an optimization variable across the full measured table. Do not invent another varying substrate or an unspecified fixed amine identity.'),
            inputs=[_categorical('Ligand', 'Phosphine ligand identity.'),
                    _categorical('Additive', 'Additive identity; source describes nitrogen/oxygen heterocycles.'),
                    _categorical('Base', 'Base identity.'),
                    _categorical('Arylhalide', 'Aryl-halide coupling substrate identity.')],
            source_references=[
                'code/chemist_system/third_party/reasoning_bo_official/src/config/buchwald_config.json',
                'code/chemist_system/chem/benchmarks/reasoning_bo_data.py',
                'experiments/datasets/reasoning_bo/Doyle_Buchwald_Hartwig/pool.json'])
    if task == 'SnAr':
        return dict(task=task, group='P2', reaction_or_process='Nucleophilic aromatic substitution in a plug-flow reactor',
            description='2,4-dinitrofluorobenzene reacts with pyrrolidine. Optimize productivity by varying residence time, pyrrolidine equivalents, inlet substrate concentration and temperature.',
            objective=dict(name='space_time_yield', unit='kg m^-3 h^-1', direction='maximize',
                meaning='Space-time yield (STY), not conversion percentage. E-factor is an auxiliary recorded quantity, not the optimization objective.'),
            substrate_scope=dict(substrates_are_optimization_variables=False,
                identity_status='2,4-dinitrofluorobenzene and pyrrolidine are specified',
                statement='The reactant identities are fixed; concentration and pyrrolidine equivalents vary.'),
            inputs=[_continuous('tau', 'Residence time', 'min', (.5, 2.)),
                    _continuous('equiv_pldn', 'Molar equivalents of pyrrolidine', 'equiv', (1., 5.)),
                    _continuous('conc_dfnb', 'Inlet 2,4-dinitrofluorobenzene concentration after mixing', 'mol/L', (.1, .5)),
                    _continuous('temperature', 'Reactor temperature', 'degC', (30., 120.))],
            source_references=[
                'code/chemist_system/chem/benchmarks/continuous_benchmark_data.py',
                'runtime/environments/.venv-summit/Lib/site-packages/summit/benchmarks/snar.py::SnarBenchmark._setup_domain',
                'experiments/datasets/continuous_20261029/SnAr/pool.json'])
    if task == 'HPLC':
        return dict(task=task, group='P2', reaction_or_process='Automated HPLC sample-injection process',
            description='Optimize sample-loop filling and transfer to maximize measured chromatographic peak area through six physical injection-process settings.',
            objective=dict(name='peak_area', unit='source arbitrary peak-area units', direction='maximize',
                meaning='Detector peak area, not chemical reaction yield, conversion percentage or photodegradation.'),
            substrate_scope=dict(substrates_are_optimization_variables=False,
                identity_status='analyte and chromatographic chemistry are unspecified in the supplied public domain',
                statement='Only the six supplied injection-process settings may vary; no reaction, reagent or analyte identity is invented.'),
            inputs=[_continuous('sample_loop', 'Sample-loop volume', 'mL', (0., .08)),
                    _continuous('additional_volume', 'Additional volume required to draw sample', 'mL', (0., .06)),
                    _continuous('tubing_volume', 'Volume required to drive sample through tubing', 'mL', (.1, .9)),
                    _continuous('sample_flow', 'Sample-pump draw rate', 'mL/min', (.5, 2.5)),
                    _continuous('push_speed', 'Push-pump draw-rate setting, expressed as pump frequency', 'Hz', (80., 150.)),
                    _continuous('wait_time', 'Wait time', 's', (.5, 10.))],
            source_references=[
                'experiments/datasets/continuous_20261029/sources/hplc_description.txt',
                'experiments/datasets/continuous_20261029/sources/hplc_config.json',
                'code/chemist_system/chem/benchmarks/continuous_benchmark_data.py',
                'experiments/datasets/continuous_20261029/HPLC/pool.json'])
    raise ValueError('Unknown task')


def task_context(task, public_pool=None):
    """Build safe prompt metadata from a fixed registry plus public input fields.

    ``public_pool`` may be a raw JSON pool or an adapter Pool. Only ids (count),
    axis names, per-axis levels and input bounds can enter the returned value.
    Arbitrary source description/target/prior fields are never forwarded.
    """
    match = re.fullmatch(r'(?:Reizman_Suzuki_Case_|Suzuki_Case|case)([1-4])', task)
    if match:
        task = f'Reizman_Suzuki_Case_{match.group(1)}'
    if task not in TASKS:
        raise ValueError(f'Unsupported task: {task}')
    result = _base(task)
    result['candidate_policy'] = (
        'Only the explicitly presented unqueried measured/simulated candidate conditions are executable. '
        'Do not propose an unlisted combination or add a variable. Category strings are identities, '
        'not numeric ranks or distances. Candidate IDs convey no chemistry or objective ranking.')
    result['knowledge_policy'] = (
        'Mechanistic expectations are uncertain, locally testable hypotheses. Use only supplied '
        'own observations as measurements. Do not retrieve or recall benchmark target values, '
        'invent molecular identities behind opaque labels, or use other tasks as observations.')
    if public_pool is not None:
        p = public_pool if isinstance(public_pool, dict) else public_pool.raw_pool
        claimed_task = p.get('task', p.get('task_id'))
        if claimed_task is not None and claimed_task != task:
            raise ValueError('Public pool task mismatch')
        expected_axes = [row['name'] for row in result['inputs']]
        if 'axes' in p and list(p['axes']) != expected_axes:
            raise ValueError('Public pool axes mismatch')
        result['candidate_count'] = len(p['ids'])
        for row in result['inputs']:
            axis = row['name']
            if row['kind'] == 'categorical' and 'levels' in p:
                levels = p['levels'][axis]
                if not levels or any(not isinstance(v, str) or not v for v in levels) or len(set(levels)) != len(levels):
                    raise ValueError('Expected distinct named categories')
                row['levels'] = list(levels)
            elif row['kind'] == 'continuous' and 'bounds' in p:
                bounds = p['bounds'][axis]
                if len(bounds) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                    raise ValueError('Invalid public input bounds')
                lo, hi = row['declared_input_bounds']
                if bounds[0] < lo - 1e-10 or bounds[1] > hi + 1e-10:
                    raise ValueError('Executable input range exceeds public domain')
                row['executable_pool_bounds'] = list(bounds)
    return deepcopy(result)


def audit_notes():
    """Provenance discrepancies for the report; not required in the LLM prompt."""
    return [
        dict(task='HPLC', issue='Original description lists wait_time [1,10], while config uses [0.5,10].',
             resolution='Use the configuration domain and the supplied pool input range; units remain seconds.',
             sources=['experiments/datasets/continuous_20261029/sources/hplc_description.txt',
                      'experiments/datasets/continuous_20261029/sources/hplc_config.json']),
        dict(task='HPLC', issue='Original description target line says photo degradation, while its introduction and config identify peak response/peak_area.',
             resolution='Use configured peak_area and the existing objective contract; do not relabel it as reaction yield or photodegradation.',
             sources=['experiments/datasets/continuous_20261029/sources/hplc_description.txt',
                      'experiments/datasets/continuous_20261029/sources/hplc_config.json']),
        dict(task='P0', issue='Supplied catalyst tokens and case number do not resolve ligand or substrate structures.',
             resolution='Keep catalyst identities opaque and substrate identities unspecified; do not invent a lookup table.',
             sources=['code/chemist_system/chem/benchmarks/suzuki_domain.py']),
    ]
