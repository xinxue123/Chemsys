"""Read-only material-data audit and an exact finite-pool response oracle."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import time
from pathlib import Path
from typing import Sequence
from dataclasses import dataclass
from copy import deepcopy
from uuid import uuid4
from types import MappingProxyType

import numpy as np

from .hypothesis import (FeatureEncoder, LLMResponsePending, ParameterSpec,
                         canonical_hash)


QUANTIFIABLE = frozenset({"pass", "pass_caution", "near"})
ATOMIC_MASSES = {"Ce": 140.12, "Al": 26.98, "Ge": 72.63}


class ReplayOracle:
    """Return one saved outcome per ID; never interpolate unseen conditions.

    The policy receives features and revealed observations separately. The
    evaluator can retain the complete table for regret calculation.
    """
    def __init__(self, candidate_ids: Sequence[str], values: Sequence[float], *, sigmas=None):
        ids = tuple(candidate_ids)
        y = tuple(float(v) for v in values)
        if (len(ids) != len(y) or len(set(ids)) != len(ids) or not np.isfinite(y).all()
                or any(not isinstance(key, str) or not key for key in ids)):
            raise ValueError("oracle IDs and finite outcomes must align one to one")
        self._values = dict(zip(ids, y))
        uncertainty = None if sigmas is None else tuple(float(v) for v in sigmas)
        if uncertainty is not None and (len(uncertainty) != len(y) or not np.isfinite(uncertainty).all()
                                        or any(v < 0 for v in uncertainty)):
            raise ValueError("oracle uncertainties must be finite, nonnegative, and aligned")
        self._sigmas = None if uncertainty is None else dict(zip(ids, uncertainty))
        self._ledger: list[dict] = []
        self._seen: set[str] = set()

    def query(self, candidate_id: str) -> float:
        if candidate_id not in self._values:
            raise ValueError("candidate is outside the exact replay table")
        if candidate_id in self._seen:
            raise ValueError("repeat queries are prohibited in exact-pool replay")
        value = self._values[candidate_id]
        self._seen.add(candidate_id)
        self._ledger.append({"query": len(self._ledger) + 1, "candidate_id": candidate_id, "value": value})
        if self._sigmas is not None:
            self._ledger[-1]['sigma'] = self._sigmas[candidate_id]
        return value

    @property
    def fingerprint(self):
        return canonical_hash({'values': self._values, 'sigmas': self._sigmas})

    @property
    def ledger(self) -> tuple[dict, ...]:
        return tuple(dict(row) for row in self._ledger)

    def restore(self, ledger: Sequence[dict]):
        """Restore already revealed outcomes without querying them again."""
        rows = deepcopy(list(ledger))
        ids = [row["candidate_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("saved oracle ledger repeats a candidate")
        for j, row in enumerate(rows):
            key = row["candidate_id"]
            if (row["query"] != j + 1 or key not in self._values
                    or row["value"] != self._values[key]):
                raise ValueError("saved oracle outcomes do not match the frozen response table")
            if (self._sigmas is None and 'sigma' in row) or (self._sigmas is not None and row.get('sigma') != self._sigmas[key]):
                raise ValueError("saved oracle uncertainty protocol changed")
        self._ledger, self._seen = rows, set(ids)


def replace_file_with_retry(source, target):
    """Keep atomic replacement while tolerating short Windows file locks."""
    for attempt in range(8):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.02 * 2 ** attempt, .25))


def write_json_atomic(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    replace_file_with_retry(temporary, target)


@dataclass(frozen=True)
class SynthesisTask:
    """Public task context; intentionally contains no unrevealed outcomes."""
    task_id: str
    target_phase_id: str
    route_template_id: str
    parameters: tuple[ParameterSpec, ...]
    candidate_ids: tuple[str, ...]
    candidates: tuple[dict, ...]
    evidence: tuple[dict, ...]
    description: str = ""
    structure_reference: str | None = None
    evaluation_only: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'parameters', tuple(self.parameters))
        object.__setattr__(self, 'candidate_ids', tuple(self.candidate_ids))
        object.__setattr__(self, 'candidates', tuple(MappingProxyType(dict(row)) for row in self.candidates))
        object.__setattr__(self, 'evidence', tuple(MappingProxyType(deepcopy(dict(row))) for row in self.evidence))
        if not self.task_id or not self.target_phase_id or not self.route_template_id:
            raise ValueError("task, target-phase, and route identities are required")
        if (not self.candidate_ids or len(self.candidate_ids) != len(self.candidates)
                or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or any(not isinstance(key, str) or not key for key in self.candidate_ids)):
            raise ValueError("task needs aligned, distinct, nonempty candidate IDs")
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("parameter names must be unique")
        encoder = FeatureEncoder(self.schema)
        encoder.encode(self.candidates)
        if len({e['id'] for e in self.evidence}) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")

    @property
    def schema(self):
        return {p.name: p for p in self.parameters}

    def to_dict(self):
        return {"task_id": self.task_id, "target_phase_id": self.target_phase_id,
                "route_template_id": self.route_template_id, "description": self.description,
                "structure_reference": self.structure_reference, "evaluation_only": self.evaluation_only,
                "parameters": [{"name": p.name, "unit": p.unit, "bounds": p.bounds,
                                "levels": p.levels} for p in self.parameters],
                "candidate_ids": list(self.candidate_ids), "candidates": [dict(row) for row in self.candidates],
                "evidence": [deepcopy(dict(row)) for row in self.evidence]}

    @property
    def fingerprint(self):
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        parameters = tuple(ParameterSpec(row['name'], row['unit'],
                                         None if row['bounds'] is None else tuple(row['bounds']),
                                         tuple(row['levels'])) for row in value['parameters'])
        return cls(task_id=value['task_id'], target_phase_id=value['target_phase_id'],
                   route_template_id=value['route_template_id'], parameters=parameters,
                   candidate_ids=tuple(value['candidate_ids']), candidates=tuple(value['candidates']),
                   evidence=tuple(value['evidence']), description=value.get('description', ''),
                   structure_reference=value.get('structure_reference'),
                   evaluation_only=value.get('evaluation_only', True))


class InteractiveAssistantProvider:
    """Queue an exact request for this chat's assistant and replay its response.

    No heuristic response is generated here. The response must be supplied by
    the assistant, linked to the request hash, and identify the frozen model.
    Token usage is unknown unless the actual inference surface reports it.
    """
    def __init__(self, directory, model_metadata, *, prompt_text=None):
        self.directory = Path(directory)
        self.model_metadata = deepcopy(model_metadata)
        self.prompt_text = prompt_text
        if prompt_text is not None:
            self.model_metadata['prompt_sha256'] = sha256(prompt_text.encode('utf-8')).hexdigest()
        if self.model_metadata.get('provider') != 'interactive_codex_assistant':
            raise ValueError("interactive assistant provider identity is required")

    def __call__(self, context):
        fingerprint = canonical_hash(context)
        folder = self.directory / fingerprint
        request_path, response_path = folder / 'request.json', folder / 'response.json'
        if not request_path.exists():
            write_json_atomic(request_path, {"context_sha256": fingerprint, "context": context,
                                            "model_metadata": self.model_metadata})
        else:
            saved = json.loads(request_path.read_text(encoding='utf-8'))
            if canonical_hash(saved['context']) != fingerprint or saved['model_metadata'] != self.model_metadata:
                raise ValueError("queued request or model identity changed")
        if not response_path.exists():
            raise LLMResponsePending(str(request_path.resolve()))
        payload = json.loads(response_path.read_text(encoding='utf-8'))
        if payload.get('context_sha256') != fingerprint:
            raise ValueError("assistant response is linked to another context")
        response = payload['response']
        provider = response.get('provider', {})
        if (provider.get('kind') != 'interactive_codex_assistant'
                or provider.get('model_id') != self.model_metadata['model_id']
                or provider.get('conversation_id') != self.model_metadata['conversation_id']):
            raise ValueError("assistant response provider differs from the frozen identity")
        return response


def load_cealge_observations(path):
    """One common numerical filter for CLI, legacy replay, and research adapters.

    This returns provisionally comparable numerical records, not certification
    that laboratory conditions or mass-ratio provenance have been verified.
    """
    audit = audit_cealge_dataset(path)
    rows = []
    for row in audit['records']:
        if not row['provisionally_eligible']:
            continue
        fractions = row['recomputed_atom_fraction_assuming_mass_ratio']
        rows.append({"candidate_id": row['candidate_id'],
                     "params": {"atom_fraction_Ce": fractions['Ce'], "atom_fraction_Al": fractions['Al']},
                     "yield": row['physical_yield_pct'], "yield_sigma": row['physical_sigma_pct'],
                     "source_sha256": audit['source_sha256'], "qualification": "needs_validation"})
    return rows


MATH_REGIMES = ('relevant', 'single_harmful', 'joint_only_harmful', 'mixed')


def mathematical_task(family: int, instance: int, *, pool_seed: int, pool_size=512):
    """Five explicit function families; no target scaling by pool extrema."""
    from scipy.special import expit
    from scipy.stats import qmc
    if family not in range(1, 6) or instance not in range(5) or pool_size != 512:
        raise ValueError('math design has five families, instances 0..4, and a fixed 512-point pool')
    dimensions = 2 + instance
    x = qmc.Sobol(dimensions, scramble=True, seed=pool_seed).random_base2(9)
    center = .2 + .6 * ((np.arange(dimensions) + 2 * instance) % 7) / 6
    weights = 1 + np.arange(dimensions)
    distance = np.sum((x - center) ** 2, axis=1)
    if family == 1:
        values = np.exp(-np.sum(weights * (x - center) ** 2, axis=1))
    elif family == 2:
        values = np.exp(-8 * (x[:, 0] + x[:, 1] - 1) ** 2
                        - 2 * (x[:, 0] - x[:, 1] - .2) ** 2
                        - np.sum(weights[2:] * (x[:, 2:] - center[2:]) ** 2, axis=1))
    elif family == 3:
        values = np.maximum(.8 * np.exp(-12 * np.sum((x - (1 - center)) ** 2, axis=1)),
                            np.exp(-20 * distance))
    elif family == 4:
        values = np.exp(-distance) * np.prod(.5 + .5 * np.cos(4 * np.pi * (x - center)), axis=1)
    else:
        values = expit(2 - 4 * distance + 2 * np.cos(6 * np.pi * (x[:, 0] * x[:, 1] - center[0] * center[1])))
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('mathematical objective violated its predefined [0,1] range')
    names = tuple(f'x{j}' for j in range(dimensions))
    task = SynthesisTask(
        task_id=f'F{family}_r{instance}', target_phase_id='not_applicable_mathematical_benchmark',
        route_template_id='finite_pool_numerical_evaluation',
        parameters=tuple(ParameterSpec(name, 'fraction', (0, 1)) for name in names),
        candidate_ids=tuple(f'c{i:04d}' for i in range(pool_size)),
        candidates=tuple(dict(zip(names, map(float, row))) for row in x),
        evidence=({'id': 'synthetic_prior_constructor',
                   'text': 'Predefined oracle-aware artificial prior stress test; not a natural LLM output.'},),
        description=f'Fixed synthetic function family F{family}, instance {instance}; normalized mathematical objective.')
    return task, values


def make_stress_priors(task, values):
    """Freeze oracle-aware positive/negative controls before any BO runs.

    The constructor alone sees full outcomes. The policy receives only the
    resulting data predicates. High quality is the top ceil(0.1*N) candidates.
    Two overlapping windows can individually retain this reference set while
    their nonempty conjunction excludes it. GP near-optimal conflicts are
    measured during execution and are not guaranteed by this construction.
    """
    from .hypothesis import compile_membership, parse_hypotheses
    ids = task.candidate_ids
    x = np.array([[row[p.name] for p in task.parameters] for row in task.candidates])
    y = np.asarray(values, dtype=float)
    top_count = int(np.ceil(.1 * len(ids)))
    high = sorted(range(len(ids)), key=lambda i: (-float(y[i]), ids[i]))[:top_count]
    high_set = set(high)

    def gap_interval(dimension):
        coordinates = sorted(x[high, dimension])
        options = []
        for i in range(len(ids)):
            z = x[i, dimension]
            if i in high_set or not coordinates[0] < z < coordinates[-1]:
                continue
            below = max(v for v in coordinates if v < z)
            above = min(v for v in coordinates if v > z)
            options.append((min(z - below, above - z), ids[i], i, (z + below) / 2, (z + above) / 2))
        if not options:
            # Retain the instance rather than resampling until it is favorable.
            return (0.0, 1.0, False)
        _, _, _, lo, hi = min(options, key=lambda row: (-row[0], row[1]))
        return float(lo), float(hi), True

    def interval(key, dim, lo, hi):
        return {'id': key, 'kind': 'interval', 'parameter': task.parameters[dim].name,
                'unit': 'fraction', 'lower': float(lo), 'upper': float(hi),
                'evidence_ids': ['synthetic_prior_constructor'], 'observation_ids': [],
                'rationale': 'Frozen artificial prior control, generated before sequential optimization.'}

    def good_window(key, dim):
        return interval(key, dim, max(0, float(x[high, dim].min()) - .05),
                         min(1, float(x[high, dim].max()) + .05))

    def linear(key, sign):
        lhs = x[high, 0] + sign * x[high, 1]
        return {'id': key, 'kind': 'linear_le', 'unit': 'normalized',
                'coefficients': {task.parameters[0].name: 1.0, task.parameters[1].name: float(sign)},
                'rhs': float(lhs.max() + .05), 'evidence_ids': ['synthetic_prior_constructor'],
                'observation_ids': [], 'rationale': 'Frozen artificial compatible interaction prior.'}
    lo0, hi0, realized0 = gap_interval(0)
    lo1, hi1, realized1 = gap_interval(1)
    correct = [good_window('H1', 0), good_window('H2', 1), linear('H3', 1), linear('H4', -1)]
    sets = {
        'relevant': correct,
        'single_harmful': [interval('H1', 0, lo0, hi0), *correct[1:]],
        'joint_only_harmful': [interval('H1', 0, 0, hi0), interval('H2', 0, lo0, 1), *correct[2:]],
        'mixed': [interval('H1', 0, lo0, hi0), correct[1], interval('H3', 1, lo1, hi1), correct[3]],
    }
    result = {}
    for regime in MATH_REGIMES:
        unique, signatures, duplicate_ids = [], set(), []
        for row in sets[regime]:
            signature = canonical_hash({k: v for k, v in row.items()
                                        if k not in {'id', 'rationale', 'evidence_ids', 'observation_ids'}})
            if signature in signatures:
                duplicate_ids.append(row['id'])
            else:
                signatures.add(signature)
                unique.append(row)
        batch = {'version': 1, 'hypotheses': unique}
        hypotheses = parse_hypotheses(batch, task.schema, evidence_ids={'synthetic_prior_constructor'}, observed_ids=set())
        h = compile_membership(hypotheses, task.candidates, task.schema)
        priority = sorted(range(len(ids)), key=lambda i: (-int(h[:, i].sum()), ids[i]))
        result[regime] = {'batch': batch, 'priority_ids': [ids[i] for i in priority],
                          'construction': {'oracle_aware': True, 'reference_top_count': top_count,
                                           'gap0_realized': realized0, 'gap1_realized': realized1,
                                           'duplicate_predicate_ids_merged': duplicate_ids,
                                           'reference_high_counts_per_constraint': [int(row[high].sum()) for row in h],
                                           'reference_high_count_joint': int(h[:, high].all(axis=0).sum()),
                                           'full_pool_joint_count': int(h.all(axis=0).sum())}}
    return result


def _summit_inference_module():
    """Load the unmodified emulator module without unused optimizer imports.

    Summit's package initializer eagerly imports every optimization strategy.
    Namespace package bootstrap avoids those optional, unrelated dependencies;
    the actual domain, data, experiment, and emulator source files are unchanged.
    """
    import importlib
    import importlib.metadata
    import importlib.util
    import sys
    import types
    if importlib.metadata.version('summit') != '0.8.9':
        raise ValueError('the frozen emulator adapter requires Summit 0.8.9')
    location = Path(importlib.metadata.distribution('summit').locate_file('summit')).resolve()
    if 'summit' not in sys.modules:
        spec = importlib.util.spec_from_file_location('summit', location / '__init__.py',
                                                    submodule_search_locations=[str(location)])
        package = importlib.util.module_from_spec(spec)
        package.get_summit_config_path = lambda config_dir_name='.summit': Path.home() / config_dir_name
        sys.modules['summit'] = package
    if 'summit.benchmarks' not in sys.modules:
        package = types.ModuleType('summit.benchmarks')
        package.__file__ = str(location / 'benchmarks' / '__init__.py')
        package.__path__ = [str(location / 'benchmarks')]
        sys.modules['summit.benchmarks'] = package
    return importlib.import_module('summit.benchmarks.experimental_emulator')


def load_frozen_summit(case):
    """Official checkpoint load with zero optimizer updates.

    The upstream deserializer runs a one-epoch dummy fit to initialize its
    sklearn/skorch objects, then restores saved weights/scalers. Here that
    initialization uses zero epochs. Every optimizer step and backward call is
    forbidden, and final tensors/scalers are checked against the saved assets.
    """
    from contextlib import ExitStack
    from unittest.mock import patch
    import torch
    module = _summit_inference_module()
    original_train = module.ExperimentalEmulator.train
    steps = {'optimizer_steps': 0, 'backward_calls': 0, 'zero_epoch_initializations': 0}

    def initialize_only(self, **kwargs):
        if kwargs.get('initializing') is not True:
            raise RuntimeError('simulator training is prohibited')
        kwargs['max_epochs'] = 0
        kwargs['random_state'] = 20261026
        steps['zero_epoch_initializations'] += 1
        return original_train(self, **kwargs)

    def forbidden_step(*args, **kwargs):
        steps['optimizer_steps'] += 1
        raise RuntimeError('simulator optimizer update is prohibited')

    def forbidden_backward(*args, **kwargs):
        steps['backward_calls'] += 1
        raise RuntimeError('simulator gradient evaluation is prohibited')

    torch.set_num_threads(1)
    with ExitStack() as stack:
        stack.enter_context(patch.object(module.ExperimentalEmulator, 'train', initialize_only))
        stack.enter_context(patch.object(torch.Tensor, 'backward', forbidden_backward))
        for cls in vars(torch.optim).values():
            if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer) and 'step' in cls.__dict__:
                stack.enter_context(patch.object(cls, 'step', forbidden_step))
        emulator = module.get_pretrained_reizman_suzuki_emulator(case=case)
    folder = module.get_model_path() / f'reizman_suzuki_case_{case}'
    parameters = json.loads((folder / f'reizman_suzuki_case_{case}.json').read_text(encoding='utf-8'))
    if len(emulator.predictors) != len(parameters['experiment_params']['predictors']):
        raise ValueError('pretrained ensemble member count changed')
    for j, predictor in enumerate(emulator.predictors):
        checkpoint = folder / f'reizman_suzuki_case_{case}_predictor_{j}.pt'
        saved = torch.load(checkpoint, map_location='cpu')
        actual = predictor.regressor_.named_steps.net.module_.state_dict()
        if set(saved) != set(actual) or any(not torch.equal(actual[key].cpu(), value) for key, value in saved.items()):
            raise ValueError('loaded simulator parameters differ from the official checkpoint')
        params = parameters['experiment_params']['predictors'][j]
        num = predictor.regressor_.named_steps.preprocessor.named_transformers_.num
        target = predictor.transformer_
        for key in ('mean_', 'var_', 'scale_', 'n_samples_seen_'):
            if not np.array_equal(getattr(num, key), params['input_preprocessor']['num'][key]):
                raise ValueError('saved input scaling changed')
            if not np.array_equal(getattr(target, key), params['output_preprocessor'][key]):
                raise ValueError('saved objective scaling changed')
    return emulator, parameters, module, steps


def _manual_summit_forward(frame, parameters, folder):
    """Independent checkpoint forward pass for preprocessing/parity checks."""
    import torch
    model_name = parameters['experiment_params']['model_name']
    inputs = [row for row in parameters['domain'] if not row['is_objective']]
    numerical = [row['name'] for row in inputs if row['type'] == 'ContinuousVariable']
    categorical = [row for row in inputs if row['type'] == 'CategoricalVariable']
    outputs = [row for row in parameters['domain'] if row['is_objective']]
    predictions = []
    with torch.no_grad():
        for j, settings in enumerate(parameters['experiment_params']['predictors']):
            numeric = settings['input_preprocessor']['num']
            continuous = (frame[numerical].to_numpy(dtype=float) - numeric['mean_']) / numeric['scale_']
            encoded = [continuous]
            for variable in categorical:
                encoded.append(np.array([[float(v == level) for level in variable['levels']]
                                         for v in frame[variable['name']]], dtype=float))
            x = torch.tensor(np.concatenate(encoded, axis=1).astype(np.float32))
            weights = torch.load(folder / f'{model_name}_predictor_{j}.pt', map_location='cpu')
            hidden = torch.relu(x @ weights['input_layer.weight'].T + weights['input_layer.bias'])
            y = (hidden @ weights['output_layer.weight'].T + weights['output_layer.bias']).numpy()
            y *= np.asarray(settings['output_preprocessor']['scale_'])
            y += np.asarray(settings['output_preprocessor']['mean_'])
            for k, output in enumerate(outputs):
                y[:, k] = np.clip(y[:, k], *output['bounds'])
            predictions.append(y)
    array = np.asarray(predictions)
    return array.mean(axis=0), array.std(axis=0)


def summit_task(case, *, pool_seed, evidence):
    from scipy.stats import qmc
    catalysts = ('P1-L1', 'P2-L1', 'P1-L2', 'P1-L3', 'P1-L4', 'P1-L5', 'P1-L6', 'P1-L7')
    numerical = (ParameterSpec('t_res', 's', (60, 600)),
                 ParameterSpec('temperature', 'degC', (30, 110)),
                 ParameterSpec('catalyst_loading', 'mol%', (.5, 2.5)))
    rows = []
    for j, catalyst in enumerate(catalysts):
        points = qmc.Sobol(3, scramble=True, seed=pool_seed + j).random_base2(6)
        for point in points:
            row = {'catalyst': catalyst}
            row.update({p.name: float(p.bounds[0] + value * (p.bounds[1] - p.bounds[0]))
                        for p, value in zip(numerical, point)})
            rows.append(row)
    return SynthesisTask(task_id=f'ReizmanSuzuki_case_{case}',
                         target_phase_id='not_applicable_reaction_benchmark',
                         route_template_id=f'fixed_Reizman_Suzuki_case_{case}',
                         parameters=(ParameterSpec('catalyst', 'category', levels=catalysts), *numerical),
                         candidate_ids=tuple(f'c{i:04d}' for i in range(512)), candidates=tuple(rows),
                         evidence=tuple(evidence),
                         description=f'Pretrained Suzuki–Miyaura reaction Case {case}; maximize yld/100 only.')


def cache_summit_case(directory, case, *, pool_seed, evidence, parity_tolerance=1e-4):
    """Batch cache a frozen emulator, checking independent and public API parity."""
    import importlib.metadata
    import pandas as pd
    import time
    import torch
    if case not in (1, 2, 3, 4):
        raise ValueError('Reizman benchmark has Cases 1..4')
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    task = summit_task(case, pool_seed=pool_seed, evidence=evidence)
    if (root / 'cache_receipt.json').exists():
        saved = json.loads((root / 'cache_receipt.json').read_text(encoding='utf-8'))
        if (saved['case'] != case or saved['task_sha256'] != task.fingerprint
                or saved['pool_seed'] != pool_seed
                or sha256((root / 'oracle.npz').read_bytes()).hexdigest() != saved['oracle_sha256']):
            raise ValueError('existing emulator cache changed; refusing replacement')
        return saved
    started = time.perf_counter()
    emulator, parameters, module, steps = load_frozen_summit(case)
    loaded = time.perf_counter()
    frame = pd.DataFrame([dict(row) for row in task.candidates])
    with torch.no_grad():
        emulator._predict(frame.iloc[:8])
        warmed = time.perf_counter()
        mean, std = emulator._predict(frame)
        predicted = time.perf_counter()
    folder = module.get_model_path() / f'reizman_suzuki_case_{case}'
    manual_mean, manual_std = _manual_summit_forward(frame, parameters, folder)
    independent_error = float(max(np.max(np.abs(mean - manual_mean)), np.max(np.abs(std - manual_std))))
    from summit.utils.dataset import DataSet
    official = emulator.run_experiments(DataSet.from_df(frame.iloc[:3].copy()), return_std=True)
    public = official[parameters['experiment_params']['output_variable_names']].to_numpy(dtype=float)
    public_error = float(np.max(np.abs(public - mean[:3])))
    if independent_error > parity_tolerance or public_error > parity_tolerance:
        raise ValueError(f'pretrained inference parity failed: independent={independent_error}, public={public_error}')
    names = parameters['experiment_params']['output_variable_names']
    values = mean[:, names.index('yld')].astype(np.float64) / 100
    if np.any((values < 0) | (values > 1)) or not np.isfinite(values).all():
        raise ValueError('emulator returned invalid normalized yield')
    np.savez_compressed(root / 'oracle.npz', values=values)
    write_json_atomic(root / 'task.json', task.to_dict())
    report = {'schema_version': 1, 'case': case, 'task_sha256': task.fingerprint,
              'pool_seed': pool_seed, 'pool_size': len(values), 'scope': 'pretrained emulator cache; not observed optimization gains',
              'summit': importlib.metadata.version('summit'), 'torch': torch.__version__,
              'numpy': np.__version__, 'device': 'cpu', 'torch_threads': torch.get_num_threads(),
              'optimizer_steps': steps['optimizer_steps'], 'backward_calls': steps['backward_calls'],
              'zero_epoch_initializations': steps['zero_epoch_initializations'],
              'checkpoint_and_scaler_equality': True, 'independent_forward_max_abs_error': independent_error,
              'public_api_max_abs_error': public_error, 'parity_tolerance_percentage_units': parity_tolerance,
              'load_seconds': loaded - started, 'warmup_seconds': warmed - loaded,
              'batch_512_seconds': predicted - warmed, 'operator_compiles_in_loop': 0,
              'module_source_sha256': sha256(Path(module.__file__).read_bytes()).hexdigest(),
              'assets_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()},
              'oracle_sha256': sha256((root / 'oracle.npz').read_bytes()).hexdigest(),
              'bootstrap': 'namespace import omits unrelated optimizers; official emulator source unchanged',
              'deserialization': 'official loader with initialization max_epochs=0 and updates forbidden'}
    write_json_atomic(root / 'cache_receipt.json', report)
    return report


def make_reaction_stress_priors(task, values):
    """Map the already fixed stress constructor to physical process units.

    Catalyst identities remain unconstrained in these artificial controls.
    Natural assistant proposals are evaluated in a separate track.
    """
    from .hypothesis import compile_membership, parse_hypotheses
    numerical = tuple(p for p in task.parameters if p.bounds is not None)
    names = tuple(f'x{j}' for j in range(len(numerical)))
    normalized = tuple({name: (float(row[p.name]) - p.bounds[0]) / (p.bounds[1] - p.bounds[0])
                        for name, p in zip(names, numerical)} for row in task.candidates)
    proxy = SynthesisTask(task.task_id + '_prior_constructor', 'not_applicable_stress_constructor',
                          'fixed_numeric_stress_constructor',
                          tuple(ParameterSpec(name, 'fraction', (0, 1)) for name in names),
                          task.candidate_ids, normalized,
                          ({'id': 'synthetic_prior_constructor', 'text': 'Artificial stress constructor.'},))
    priors = {regime: deepcopy(definition) for regime, definition in make_stress_priors(proxy, values).items()}
    mapping = dict(zip(names, numerical))
    for definition in priors.values():
        for hypothesis in definition['batch']['hypotheses']:
            if hypothesis['kind'] == 'interval':
                p = mapping[hypothesis['parameter']]
                lo, hi = p.bounds
                hypothesis.update(parameter=p.name, unit=p.unit,
                                  lower=lo + hypothesis['lower'] * (hi - lo),
                                  upper=lo + hypothesis['upper'] * (hi - lo))
            else:
                hypothesis['coefficients'] = {mapping[key].name: value
                                               for key, value in hypothesis['coefficients'].items()}
        parsed = parse_hypotheses(definition['batch'], task.schema,
                                  evidence_ids={e['id'] for e in task.evidence}, observed_ids=set())
        h = compile_membership(parsed, task.candidates, task.schema)
        priority = sorted(range(len(task.candidate_ids)), key=lambda i: (-int(h[:, i].sum()), task.candidate_ids[i]))
        definition['priority_ids'] = [task.candidate_ids[i] for i in priority]
        definition['construction'].update(catalysts_constrained=False,
                                           full_pool_joint_count=int(h.all(axis=0).sum()))
    return priors


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def audit_cealge_dataset(path: str | Path) -> dict:
    """Preserve all records; separate unknown outcomes from measured zeros.

    Composition conversion follows arcmelt.py's nominal-mass interpretation.
    That interpretation still requires original laboratory provenance before
    a scientific material replay pool can be frozen.
    """
    source = Path(path)
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    samples = data.get("samples", [])
    canonical = []
    for index, sample in enumerate(samples):
        file_id = sample.get("file", f"missing_id_{index}")
        status = sample.get("status", sample.get("confidence", "unverified"))
        reasons = []
        y, sigma = sample.get("yield"), sample.get("yield_sigma")
        quantifiable = status in QUANTIFIABLE and _finite(y) and 0 <= y <= 100
        if not quantifiable:
            reasons.append("unknown_or_unquantifiable_result")
        if not _finite(sigma) or not 0 <= sigma <= 50:
            reasons.append("missing_or_invalid_uncertainty")
        if sample.get("use_for_bo") is not True:
            reasons.append("not_explicitly_enabled_for_bo")
        if "DSC" in file_id.upper():
            reasons.append("different_DSC_processing_state")
        charge = sample.get("charge_ratio", {})
        fractions = None
        if set(charge) != set(ATOMIC_MASSES) or any(not _finite(v) or v <= 0 for v in charge.values()):
            reasons.append("invalid_or_non_CeAlGe_composition")
        else:
            moles = {el: charge[el] / mass for el, mass in ATOMIC_MASSES.items()}
            total = sum(moles.values())
            fractions = {el: value / total for el, value in moles.items()}
        canonical.append({
            "candidate_id": file_id, "source_row": index,
            "confidence": status, "original_use_for_bo": sample.get("use_for_bo"),
            "original_numeric_label": y,
            "physical_yield_pct": float(y) if quantifiable else None,
            "physical_sigma_pct": float(sigma) if quantifiable and _finite(sigma) and 0 <= sigma <= 50 else None,
            "nominal_mass_ratio": charge,
            "recomputed_atom_fraction_assuming_mass_ratio": fractions,
            "provisionally_eligible": not reasons, "exclusion_reasons": reasons,
        })
    duplicate_ids = [key for key, count in Counter(row["candidate_id"] for row in canonical).items() if count > 1]
    groups = Counter(tuple(sorted(row["nominal_mass_ratio"].items())) for row in canonical if row["provisionally_eligible"])
    missing_conditions = ["verified_ratio_basis", "route_template_id", "thermal_history",
                          "sample_processing_state", "measurement_protocol_id", "target_phase_structure_id"]
    return {
        "schema_version": 1, "source": str(source.resolve()), "source_sha256": sha256(raw).hexdigest(),
        "target_formula": data.get("target"), "raw_record_count": len(samples),
        "confidence_counts": dict(Counter(row["confidence"] for row in canonical)),
        "original_bo_enabled_count": sum(row["original_use_for_bo"] is True for row in canonical),
        "provisionally_eligible_count": sum(row["provisionally_eligible"] for row in canonical),
        "provisional_unique_composition_count": len(groups),
        "unknown_outcome_count": sum(row["physical_yield_pct"] is None for row in canonical),
        "duplicate_candidate_ids": duplicate_ids,
        "exclusion_reason_counts": dict(Counter(reason for row in canonical for reason in row["exclusion_reasons"])),
        "ready_for_scientific_replay": False,
        "metadata_requiring_source_verification": missing_conditions,
        "qualification": "numeric screening only; common processing conditions and mass-ratio semantics are unverified",
        "records": canonical,
    }
