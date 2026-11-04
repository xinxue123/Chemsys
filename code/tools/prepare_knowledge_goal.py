"""Freeze the confirmed task inventory without making experimental queries."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/reports/knowledge_loop_20261101'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    config_path = ROOT / 'code/resources/knowledge_loop_goal_20261101.json'
    config = read(config_path)
    specs = []
    for case in range(1, 5):
        specs.append(('P0', f'Reizman_Suzuki_Case_{case}',
            f'experiments/runs/suzuki_guarded/inputs/case{case}',
            f'experiments/runs/suzuki_mixed_kalman/base/c{case}_s0/state.json', 'advice.json'))
    for task in config['groups']['P1']:
        specs.append(('P1', task, f'experiments/datasets/reasoning_bo/{task}',
            f'experiments/runs/hte_pointwise_seed0/runs/{task}_s0_Pure_BO/state.json', 'compass.json'))
    for task in config['groups']['P2']:
        specs.append(('P2', task, f'experiments/datasets/continuous_20261029/{task}',
            f'experiments/runs/continuous_20261029/runs/{task}_Pure_BO/state.json', None))
    rows = []
    for group, task, folder, state_name, prior in specs:
        pool_path, oracle_path = ROOT / folder / 'pool.json', ROOT / folder / 'oracle.npz'
        state_path = ROOT / state_name
        pool, state = read(pool_path), read(state_path)
        history = state['history']
        assert state['status'] == 'COMPLETE' and len(history) == 20
        ids = [h['candidate_id'] for h in history]
        assert len(set(ids)) == 20 and set(ids) <= set(pool['ids'])
        values = np.asarray([h['raw_value'] if group == 'P2' else h['y'] * 100 for h in history])
        curve = np.maximum.accumulate(values)
        files = [pool_path, oracle_path, state_path]
        if prior:
            files.append(ROOT / folder / prior)
        rows.append(dict(group=group, task=task, candidate_count=len(pool['ids']), initial_ids=ids[:4],
            knowledge_conditions=config['knowledge_conditions'][group],
            historical_BO=dict(SearchAUC=float(curve[4:].mean()), final_best=float(curve[-1]),
                first_final_best_total=int(np.flatnonzero(values == curve[-1])[0]) + 1,
                status='historical_reference_until_adapter_equivalence_verified',
                state=state_name),
            unit=pool.get('objective_unit', 'yield percent'),
            baseline_alignment=('New matched BO needed if updated tie rule is retained' if group == 'P0' else
                'Shared nominal/structural GP must be fixed; old Pure_BO and Stage11 differ' if group == 'P1' else
                'R6 A0 reproduced old BO exactly; reuse only if same GP/qLogEI/tie settings'),
            files_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in files}))
    result = dict(status='INPUT_INVENTORY_VERIFIED', goal_achieved=False, task_count=len(rows),
        agent_scenarios=sum(len(r['knowledge_conditions']) for r in rows),
        configuration_sha256=digest(config_path), new_experimental_queries=0,
        source_of_metrics='recomputed from frozen full 20-observation histories', tasks=rows)
    assert result['task_count'] == 8 and result['agent_scenarios'] == 20
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'baseline_inventory.json'
    if target.exists():
        assert read(target) == result, 'Inventory changed; create a new reviewed revision instead of overwriting'
    else:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
