"""Register one input-only seed2 confirmation after the complete V5 audit.

Never calls an oracle, GP fitter or API. The sampling rule is durably recorded
before sampling input identities. This is a new paired initialization of the
same benchmark domains, not an unseen-dataset test.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_study as study
from chem.benchmarks import knowledge_gp as gp
from chem.benchmarks.knowledge_llm import digest


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, record):
    """Preserve original timestamps and reject any changed registered content."""
    path = Path(path)
    if path.exists():
        old = study.read(path)
        expected = {k:v for k,v in record.items() if k not in ('registered_utc', 'registration_sha256')}
        actual = {k:v for k,v in old.items() if k not in ('registered_utc', 'registration_sha256')}
        if actual != expected or old['registration_sha256'] != digest({k:v for k,v in old.items() if k != 'registration_sha256'}):
            raise ValueError('Existing confirmation registration changed: '+path.name)
        return old
    record = dict(record, registered_utc=now())
    record['registration_sha256'] = digest(record)
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return record


def main():
    reports = ROOT/'experiments/reports/knowledge_loop_20261101'
    protocol = study.read(study.paths(ROOT, 'v5')/'protocol.json')
    if (protocol['protocol_sha256'] != digest({k:v for k,v in protocol.items() if k != 'protocol_sha256'}) or
            protocol['source_sha256'] != study.source_hashes(ROOT) or
            study.sha(ROOT/protocol['goal_config_path']) != protocol['goal_config_sha256'] or
            study.sha(ROOT/protocol['design_document_path']) != protocol['design_document_sha256']):
        raise ValueError('V5 source/protocol/design changed before selecting the policy')
    results_path, audit_path = reports/'v5_results.json', reports/'v5_joint_independent_audit.json'
    results, audit = study.read(results_path), study.read(audit_path)
    if (results['preregistered_global_selection'] != 'full' or
            not results['arms']['full']['candidate_gate'] or
            audit['technical_result'] != 'PASS_LISTED_CHECKS' or audit['failures'] or audit['pending'] or
            audit['source_drift'] or audit['totals']['runs'] != 24 or audit['totals']['rounds'] != 384 or
            audit['protocol_sha256'] != protocol['protocol_sha256']):
        raise ValueError('Require complete audited V5 global candidate before registration')
    protected = ('knowledge_transfer_engine.py', 'knowledge_transfer_policy.py', 'knowledge_transfer_llm.py',
                 'knowledge_rule_math.py', 'knowledge_policy.py', 'knowledge_engine.py',
                 'knowledge_domain_context.py', 'knowledge_openai_client.py', 'knowledge_llm.py')
    core = {name: expected for name,expected in protocol['source_sha256'].items()
            if Path(name).name in protected}
    if len(core) != len(protected):
        raise ValueError('Incomplete frozen candidate core registry')
    rule_path = reports/'v5_seed2_rule_registration.json'
    rule = write_once(rule_path, dict(status='RULE_REGISTERED_BEFORE_INPUT_SAMPLING', trial_seed=2,
        stage='One additional preregistered paired initialization; no tuning on confirmation results',
        tasks=list(gp.TASKS), arms=['bo','full'], condition='agent_initial_terra', initial=4, queries=16,
        initialization=dict(sort='lexicographic pool.features; normalized continuous and true categorical names; no candidate ID key',
            rng='numpy.random.default_rng(numpy.random.SeedSequence([2,0]))',
            draw='rng.choice(N,4,replace=False)', reinitialize_rng_per_task=True, retain_draw_order=True,
            no_target_selection=True),
        tie_seed=2, tie_rule='existing semantic_priorities(pool,t,seed=2)', menu_seed=0, numerical_seed=0,
        native_gp_numerical_seed_policy='P0 native cache seed0; P1/P2 existing fixed formulas',
        numpy_version=np.__version__, core_source_sha256=core,
        base_GP_adapter_sha256=protocol['source_sha256']['code/chemist_system/chem/benchmarks/knowledge_gp.py'],
        allowed_adapter_changes='Input registration, accepted trial seed and tie stream only; no GP recipe or acquisition changes',
        candidate_protocol_sha256=protocol['protocol_sha256'], prompt_sha256=protocol['prompt_sha256'],
        model=protocol['model'], reasoning_effort=protocol['reasoning_effort'],
        primary_acceptance='At least5of8strictSearchAUCwins versus matched BO; equality is not a win; report every final-best loss.',
        mechanism_reporting='Report all actual weighted choices and forward feedback; do not impose posthoc per-case gates.',
        interpretation='New paired initialization/tie stream on the same8domains, not an independent-dataset or multi-seed statistical guarantee.',
        initial_y_read=False, oracle_access=False, LLM_calls=0, GP_fits=0))
    # Only after the rule is on disk do we draw identities from public inputs.
    rows = []
    for task in gp.TASKS:
        pool = gp.load_task(ROOT, task, seed=0)
        manifest = next(item for item in protocol['inputs'] if item['task'] == task)
        pool_relative = pool.source_path.relative_to(ROOT).as_posix()
        if study.sha(pool.source_path) != manifest['files_sha256'][pool_relative]:
            raise ValueError('Registered candidate-only pool changed')
        if len(set(pool.features)) != len(pool.ids):
            raise ValueError('Ambiguous identical physical inputs; do not fall back to ID rank')
        order = sorted(range(len(pool.ids)), key=lambda j: pool.features[j])
        positions = np.random.default_rng(np.random.SeedSequence([2, 0])).choice(len(order), 4, replace=False)
        indices = [order[int(j)] for j in positions]
        features = [list(pool.features[j]) for j in indices]
        rows.append(dict(task=task, group=pool.group, candidate_count=len(pool.ids),
            input_pool=pool.source_path.relative_to(ROOT).as_posix(), input_pool_sha256=study.sha(pool.source_path),
            axes=list(pool.axes), kinds=list(pool.kinds), initial_ids=[pool.ids[j] for j in indices],
            initial_points=[pool.points[j] for j in indices], initial_features=features,
            initial_features_sha256=digest(features), sorted_pool_features_sha256=digest([list(x) for x in sorted(pool.features)]),
            drawn_sorted_positions=[int(j) for j in positions], initial_y_read=False))
    inputs_path = reports/'v5_seed2_input_registration.json'
    inputs = write_once(inputs_path, dict(status='INITIAL_INPUTS_REGISTERED_BEFORE_TARGET_ACCESS',
        rule_registration_sha256=rule['registration_sha256'], rule_file=rule_path.relative_to(ROOT).as_posix(),
        trial_seed=2, tie_seed=2, menu_seed=0, numerical_seed=0, numerical_recipe_unchanged=True, task_count=8,
        initial_y_read=False, oracle_access=False, LLM_calls=0, GP_fits=0, tasks=rows))
    selected = write_once(reports/'v5_selected_policy_registration.json', dict(selected_global_policy='v5_full',
        development_seed=0, candidate_protocol_sha256=protocol['protocol_sha256'],
        development_summary=results['arms']['full'],
        result_file_sha256=study.sha(results_path), audit_file_sha256=study.sha(audit_path),
        core_source_sha256=core, parameters=protocol['parameters'], prompt_sha256=protocol['prompt_sha256'],
        model=protocol['model'], reasoning_effort=protocol['reasoning_effort'],
        confirmation_trial_seed=2, confirmation_native_GP_seed=0, confirmation_menu_seed=0,
        confirmation_scope=list(gp.TASKS), confirmation_arms=['bo','full'], initial=4, queries=16,
        rule_registration_sha256=rule['registration_sha256'], input_registration_sha256=inputs['registration_sha256'],
        status='REGISTERED_NOT_STARTED_NO_NEW_INITIAL_OUTCOME_READ',
        independent_dataset_test=False, initial_y_read=False, oracle_access=False,
        limitation='Development weighted policy changes Case1/4; whole-trajectory benefits mixed. No posthoc retuning is permitted in confirmation.'))
    print(json.dumps(dict(rule_sha256=rule['registration_sha256'], input_sha256=inputs['registration_sha256'],
        selected_policy_sha256=selected['registration_sha256'], oracle_reads=0, fits=0, LLM_calls=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
