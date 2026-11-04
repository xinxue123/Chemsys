"""Independent seed2 confirmation audit of the frozen V5 full policy.

Only initial4/tie stream may change. No production imports, optimizer, API call,
or oracle value/file access. Reuses independent V5/V4 numerical and policy
checks; the seed/registration/initial-commit boundary is independently defined.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from math import log
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import audit_knowledge_transfer_loop as v5
common = v5.common
ROOT = v5.ROOT
read, digest, filehash = v5.read, v5.digest, v5.filehash
require, close, exact = v5.require, v5.close, v5.exact
ATOL = v5.ATOL
select, refresh, should_call = v5.select, v5.refresh, v5.should_call
candidate_menu, prompt_audit = v5.candidate_menu, v5.prompt_audit
validate_hypotheses, declarations_audit = v5.validate_hypotheses, v5.declarations_audit
probes, selection_audit, hkey = v5.probes, v5.selection_audit, v5.hkey
shadow_feedback, saved_gp_audit, transfer_role = v5.shadow_feedback, v5.saved_gp_audit, v5.transfer_role
REGISTRY_PATH = 'experiments/reports/knowledge_loop_20261101/v5_seed2_input_registration.json'
REGISTRY_SHA = '265b57efbea49dd6ef2915a6fe92763fc9b6034f6dac772c854b2068a690a2b8'
RULE_PATH = 'experiments/reports/knowledge_loop_20261101/v5_seed2_rule_registration.json'
RULE_SHA = '2717c2aedafeff8b90e3b9301802079a7ca4feca20d32ed740b67fc0d642950d'
SELECTED_PATH = 'experiments/reports/knowledge_loop_20261101/v5_selected_policy_registration.json'
SELECTED_SHA = 'ac811d40f82e72c07823dbbdf792f4bd0a6a568fd3bcbaebb608115d73611943'
CANDIDATE_PROTOCOL_SHA = '6534b7526210c5402d61af84dd527d0a7a8596f37c5870cc582f104ec4e15232'
VARIANT = 'v5_seed2_confirm'
CORE_NAMES = {'knowledge_transfer_engine.py','knowledge_transfer_policy.py','knowledge_transfer_llm.py',
              'knowledge_rule_math.py','knowledge_policy.py','knowledge_engine.py',
              'knowledge_domain_context.py','knowledge_openai_client.py','knowledge_llm.py'}
ADAPTERS = {'code/chemist_system/chem/benchmarks/knowledge_gp.py',
            'code/chemist_system/chem/benchmarks/knowledge_study.py','code/chemist_system/knowledge_loop.py'}
GP_PATH = 'code/chemist_system/chem/benchmarks/knowledge_gp.py'


def trial_seeds(protocol):
    require(protocol['variant'] == VARIANT and protocol['mechanism'] == 'transfer_rule_v5', 'confirmation variant/mechanism')
    for name, expected in (('seed',2),('trial_seed',2),('tie_seed',2),('numerical_seed',0),('menu_seed',0)):
        require(type(protocol[name]) is int and protocol[name] == expected, 'unregistered seed stream: '+name)
    require(protocol['initial'] == 4 and protocol['queries'] == 16
            and protocol['supported_modes'] == ['bo','full']
            and protocol['supported_conditions'] == ['agent_initial_terra'], 'confirmation scope/arms/budget')
    return 2, 2


def initial_commit_audit(path, state, protocol, pool, manifest):
    trial_seeds(protocol)
    ids = manifest['initial_ids']
    require(len(manifest['files_sha256']) == 1, 'confirmation may register only the input pool')
    source = next(iter(manifest['files_sha256']))
    require(source.endswith('/pool.json'), 'non-input initialization source')
    expected = dict(task=state['task'],mode=state['mode'],condition=state['condition'],seed=2,
        protocol_sha256=protocol['protocol_sha256'],initial_registry_sha256=REGISTRY_SHA,
        initial_ids=ids,initial_points=[pool['points'][pool['ix'][i]] for i in ids],
        initial_features=[list(pool['features'][pool['ix'][i]]) for i in ids],
        input_pool_sha256=manifest['files_sha256'][source])
    expected['commit_sha256']=digest(expected)
    actual=read(path.parent/'initial_commit.json')
    exact(actual,expected,'seed2 input-only initial commitment')
    require(state['initial_commit_sha256'] == expected['commit_sha256'], 'state initial commitment marker')
    require([r['candidate_id'] for r in state['history'][:4]] == ids,'initial observations differ from pre-target draw')


def registered_record(value, expected_sha, require_no_activity=True):
    require(value['registration_sha256'] == expected_sha
            and digest({k:v for k,v in value.items() if k != 'registration_sha256'}) == expected_sha,
            'immutable registration content SHA')
    require(value['initial_y_read'] is False and value['oracle_access'] is False, 'registration reports target access')
    if require_no_activity:
        require(value['LLM_calls'] == value['GP_fits'] == 0, 'registration reports model activity')


def validate_input_draw(pool, row):
    require(len(set(pool['features'])) == len(pool['ids']), 'ambiguous physical-feature ordering')
    order=sorted(range(len(pool['ids'])),key=lambda j:pool['features'][j])
    drawn=np.random.default_rng(np.random.SeedSequence([2,0])).choice(len(order),4,replace=False)
    js=[order[int(k)] for k in drawn]
    require(row['drawn_sorted_positions'] == list(map(int,drawn)), 'registered RNG positions differ')
    require(row['initial_ids'] == [pool['ids'][j] for j in js], 'registered physical-feature draw/order differs')
    features=[list(pool['features'][j]) for j in js]
    exact(row['initial_features'],features,'registered initial physical features')
    exact(row['initial_points'],[pool['points'][j] for j in js],'registered initial physical points')
    require(row['initial_features_sha256'] == digest(features), 'registered feature SHA')
    require(row['sorted_pool_features_sha256'] == digest([list(pool['features'][j]) for j in order]), 'sorted pool feature SHA')
    require(row['candidate_count'] == len(pool['ids']) and row['axes'] == pool['axes']
            and row['kinds'] == pool['kinds'] and row['initial_y_read'] is False, 'registered input schema')


def gp_numerical_projection(text):
    """Discard only reviewed registration gates, retain all numerical AST."""
    tree=ast.parse(text)
    kept=[]
    allowed_constants={'SEED2_REGISTRY_PATH','SEED2_REGISTRY_SHA256','SEED2_RULE_PATH','SEED2_RULE_SHA256'}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in allowed_constants:
            require(isinstance(node.value,ast.Constant) and isinstance(node.value.value,str), 'seed2 registration constant must be literal')
            continue
        if isinstance(node,ast.FunctionDef) and node.name in ('validate_trial_seed','registered_initial_ids','seed2_registration'):
            for call in ast.walk(node):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute):
                    require(call.func.attr not in ('read_observation','fit','fit_gpytorch_mll'), 'registration gate performs target/model work')
                if isinstance(call,ast.Constant) and isinstance(call.value,str):
                    require('oracle.npz' not in call.value, 'registration gate names full oracle')
            continue
        if isinstance(node,ast.FunctionDef) and node.name=='load_task':
            require(isinstance(node.body[-1],ast.Return), 'unrecognized registered-ID return')
            last=node.body[-1]
            require(isinstance(last.value,ast.IfExp), 'unrecognized registered-ID branch')
            calls=[n for n in ast.walk(last) if isinstance(n,ast.Call)]
            require({getattr(c.func,'id',None) for c in calls} <= {'replace','registered_initial_ids'}, 'new computation in initial-ID return')
            node=deepcopy(node);node.body[-1]=ast.Pass()
        if isinstance(node,ast.ClassDef) and node.name=='GPAdapter':
            node=deepcopy(node)
            init=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
            require(isinstance(init.body[0],ast.Expr) and isinstance(init.body[0].value,ast.Call)
                    and isinstance(init.body[0].value.func,ast.Name) and init.body[0].value.func.id=='validate_trial_seed', 'GP seed validation removed')
            guard=init.body[1]
            require(isinstance(guard,ast.If) and not guard.orelse and len(guard.body)==1
                    and isinstance(guard.body[0],ast.Raise), 'unrecognized GP initial-input guard')
            init.body[1]=ast.Pass()
        kept.append(node)
    return ast.dump(ast.Module(body=kept,type_ignores=[]),include_attributes=False)


def registration_audit(root=ROOT, protocol=None):
    root=Path(root)
    registry,rule,selected=[read(root/p) for p in (REGISTRY_PATH,RULE_PATH,SELECTED_PATH)]
    for value,sha in ((registry,REGISTRY_SHA),(rule,RULE_SHA)):
        registered_record(value,sha)
        require(tuple(value[k] for k in ('trial_seed','tie_seed','numerical_seed','menu_seed')) == (2,2,0,0), 'registered seed streams')
    registered_record(selected,SELECTED_SHA,False)
    require(registry['rule_registration_sha256'] == RULE_SHA and registry['rule_file'] == RULE_PATH,
            'input registration not bound to the frozen prior rule')
    require(rule['initial'] == selected['initial'] == 4 and rule['queries'] == selected['queries'] == 16
            and rule['arms'] == selected['confirmation_arms'] == ['bo','full']
            and rule['condition'] == 'agent_initial_terra', 'registered task budget/arms')
    require(selected['selected_global_policy']=='v5_full' and selected['development_seed']==0
            and selected['confirmation_trial_seed']==2 and selected['confirmation_native_GP_seed']==0
            and selected['confirmation_menu_seed']==0 and selected['independent_dataset_test'] is False,
            'global policy selected after/beside registered confirmation')
    require(selected['rule_registration_sha256']==RULE_SHA and selected['input_registration_sha256']==REGISTRY_SHA,
            'selected policy registration links')
    require(datetime.fromisoformat(rule['registered_utc']) <= datetime.fromisoformat(registry['registered_utc'])
            <= datetime.fromisoformat(selected['registered_utc']), 'registration ordering metadata')
    require(rule['numpy_version']==np.__version__, 'registered NumPy RNG version changed')
    recipe=rule['initialization']
    require(recipe['rng']=='numpy.random.default_rng(numpy.random.SeedSequence([2,0]))'
            and recipe['draw']=='rng.choice(N,4,replace=False)' and recipe['reinitialize_rng_per_task'] is True
            and recipe['retain_draw_order'] is True and recipe['no_target_selection'] is True,
            'registered input-only RNG recipe')
    candidate=read(root/'experiments/runs/knowledge_loop_20261101/v5/protocol.json')
    require(candidate['protocol_sha256']==CANDIDATE_PROTOCOL_SHA
            and digest({k:v for k,v in candidate.items() if k!='protocol_sha256'})==CANDIDATE_PROTOCOL_SHA
            and rule['candidate_protocol_sha256']==selected['candidate_protocol_sha256']==CANDIDATE_PROTOCOL_SHA,
            'selected frozen V5 candidate protocol')
    expected_core={'code/chemist_system/chem/benchmarks/'+name for name in CORE_NAMES}
    require(set(rule['core_source_sha256'])==expected_core, 'protected nine-core registry incomplete')
    exact(rule['core_source_sha256'],selected['core_source_sha256'],'selected core registry')
    for path,sha in rule['core_source_sha256'].items():
        require(candidate['source_sha256'][path]==sha and filehash(root/path)==sha,'protected core changed: '+path)
    for path,sha in candidate['source_sha256'].items():
        if path not in ADAPTERS:
            require(filehash(root/path)==sha,'non-adapter candidate source changed: '+path)
    actual_sources={'code/chemist_system/knowledge_loop.py'} | {p.relative_to(root).as_posix() for p in (root/'code/chemist_system/chem').rglob('*.py')}
    require(actual_sources==set(candidate['source_sha256']),'unregistered source added/removed')
    require(rule['base_GP_adapter_sha256']==candidate['source_sha256'][GP_PATH], 'base GP adapter registration')
    snapshot=root/'runtime/knowledge_loop/snapshots/v5'/GP_PATH
    require(filehash(snapshot)==rule['base_GP_adapter_sha256'],'V5 GP snapshot custody')
    require(gp_numerical_projection(snapshot.read_text(encoding='utf-8'))==
            gp_numerical_projection((root/GP_PATH).read_text(encoding='utf-8')),'GP numerical AST changed outside registration gates')
    for key in ('prompt_sha256','model','reasoning_effort'):
        exact(rule[key],candidate[key],'registered candidate '+key)
        exact(selected[key],candidate[key],'selected candidate '+key)
    exact(selected['parameters'],candidate['parameters'],'selected policy parameters')
    require(filehash(root/candidate['goal_config_path'])==candidate['goal_config_sha256'],'goal config custody')
    require(filehash(root/candidate['design_document_path'])==candidate['design_document_sha256'],'V5 design custody')
    reports=root/'experiments/reports/knowledge_loop_20261101'
    require(filehash(reports/'v5_results.json')==selected['result_file_sha256']
            and filehash(reports/'v5_joint_independent_audit.json')==selected['audit_file_sha256'],
            'development result/audit selection receipts changed')
    require(len(rule['tasks'])==registry['task_count']==8 and rule['tasks']==candidate['tasks']==selected['confirmation_scope']
            ==[r['task'] for r in registry['tasks']], 'registered eight-task scope')
    pools,manifests={},[]
    for row in registry['tasks']:
        require(row['input_pool'].endswith('/pool.json'),'registration contains a target-bearing input path')
        require(filehash(root/row['input_pool'])==row['input_pool_sha256'],'registered candidate-only pool SHA')
        item=dict(task=row['task'],group=row['group'],candidate_count=row['candidate_count'],
            files_sha256={row['input_pool']:row['input_pool_sha256']})
        pool=dict(common.public_pool(root,item),task=row['task'])
        validate_input_draw(pool,row)
        pools[row['task']]=pool
        manifests.append(dict(item,initial_ids=row['initial_ids'],initial_features_sha256=row['initial_features_sha256'],
                              unit=pool['unit'],knowledge_conditions=['agent_initial_terra']))
    if protocol is not None:
        trial_seeds(protocol)
        require(protocol['initial_registry_path']==REGISTRY_PATH and protocol['initial_registry_sha256']==REGISTRY_SHA
                and protocol['initial_rule_path']==RULE_PATH and protocol['initial_rule_sha256']==RULE_SHA,
                'protocol registration paths/SHAs')
        require(protocol['initial_registry_file_sha256']==filehash(root/REGISTRY_PATH)
                and protocol['initial_rule_file_sha256']==filehash(root/RULE_PATH),'protocol registration file custody')
        require(protocol['registered_candidate_protocol_sha256']==CANDIDATE_PROTOCOL_SHA
                and protocol['registered_base_GP_adapter_sha256']==rule['base_GP_adapter_sha256'], 'registered candidate protocol/GP lineage')
        exact(protocol['registered_core_source_sha256'],rule['core_source_sha256'],'protocol protected core')
        exact(protocol['inputs'],manifests,'protocol inputs must equal registered input-only initialization')
        require(protocol['tasks']==rule['tasks'],'protocol task scope')
        for key in ('parameters','prompt_sha256','model','reasoning_effort','numerical_seed_policy',
                    'goal_config_path','goal_config_sha256','design_document_path','design_document_sha256'):
            exact(protocol[key],candidate[key],'confirmation candidate setting changed: '+key)
        require(set(protocol['source_sha256'])==set(candidate['source_sha256']),'confirmation source inventory changed')
        for path,sha in candidate['source_sha256'].items():
            if path not in ADAPTERS: require(protocol['source_sha256'][path]==sha,'confirmation non-adapter hash changed')
    receipt=dict(result='PASS_INPUT_ONLY_REGISTRATION',rule_sha256=RULE_SHA,input_sha256=REGISTRY_SHA,
        selected_policy_sha256=SELECTED_SHA,candidate_protocol_sha256=CANDIDATE_PROTOCOL_SHA,
        protected_core_files=len(expected_core),registered_tasks=len(pools),initial_points=32,
        trial_seed=2,tie_seed=2,numerical_seed=0,menu_seed=0,GP_numerical_AST='UNCHANGED',
        current_GP_adapter_sha256=filehash(root/GP_PATH),oracle_files_opened=0,oracle_values_read=0,LLM_calls=0,GP_fits=0,
        limitation='Registration hashes/order establish internal custody, not an external wall-clock attestation; no experiment outcome is audited in registry-only mode.')
    return receipt,pools,{r['task']:r for r in manifests}


def audit_run(folder, path, protocol, pool, manifest, include_incomplete=False, joint=False):
    state = read(path)
    params = protocol['parameters']
    trial, tie_seed = trial_seeds(protocol)
    require(state['protocol_sha256'] == protocol['protocol_sha256'], 'state protocol')
    require(state['condition'] == 'agent_initial_terra' and state['mode'] in protocol['supported_modes'], 'unsupported V5 arm/condition')
    history, decisions = state['history'], state['decisions']
    require(state['seed'] == trial and len(history) == 4+len(decisions), 'trial/budget/history')
    require(len(decisions) == 16 or include_incomplete and 0 < len(decisions) < 16 and state['status'] != 'COMPLETE', 'query budget')
    require([r['candidate_id'] for r in history[:4]] == manifest['initial_ids'], 'initial design changed')
    initial_commit_audit(path, state, protocol, pool, manifest)
    require(len({r['candidate_id'] for r in history}) == len(history), 'duplicate query')
    for row in history:
        require(row['candidate_id'] in pool['ix'] and np.isfinite(row['y']), 'invalid observed point')
        close(row['raw_value'], row['y']*pool['scale'], 'observed engineering unit')
    previous, calls, modes, forecasts = [], [], Counter(), 0
    max_quad_error, roles = 0., []
    seen_predictions = set()
    for t, decision in enumerate(decisions, 1):
        prefix, observation = history[:t+3], history[t+3]
        cp = path.parent/'rounds'/f'{t:02d}_commit.json'
        c = read(cp)
        require(c['query'] == decision['query'] == t and c['mechanism'] == 'transfer_rule_v5', 'commit sequence/mechanism')
        require(c['history_sha256'] == digest(prefix), 'commit observed prefix hash')
        require(c['protocol_sha256'] == protocol['protocol_sha256'] and digest(c) == decision['commit_sha256'], 'commit content hash')
        require(c['selected_id'] == decision['selected_id'] == observation['candidate_id'], 'selected outcome identity')
        require(c['mode'] == decision['mode'] and c['BO_reference_id'] == decision['BO_reference_id'], 'selected mode/reference')
        close(decision['y'], observation['y'], 'decision y')
        close(decision['raw_value'], observation['raw_value'], 'decision objective')
        close(decision['incumbent_before_y'], max(r['y'] for r in prefix), 'incumbent before')
        close(decision['incumbent_after_y'], max(r['y'] for r in history[:t+4]), 'incumbent after')
        with np.load(cp.with_name(f'{t:02d}_GP.npz'), allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        require(set(arrays) == {'mu_y', 'sd_y', 'sd_predictive_y', 'base_logei', 'available', 'eligible', 'tie_priority'}, 'GP artifact fields')
        n = len(pool['ids'])
        require(all(x.shape == (n,) and np.isfinite(x).all() for x in arrays.values()), 'invalid stored GP fields')
        require(np.all(arrays['sd_y'] >= 0) and np.all(arrays['sd_predictive_y'] >= arrays['sd_y']), 'stored variance contract')
        available = ~np.isin(pool['ids'], [r['candidate_id'] for r in prefix])
        order = sorted(range(n), key=lambda j: pool['features'][j])
        ties = np.empty(n, np.int64)
        ties[order] = np.random.default_rng(np.random.SeedSequence([tie_seed, t])).permutation(n)
        require(np.array_equal(ties, arrays['tie_priority']), 'physical tie RNG/order')
        require(np.array_equal(available, arrays['available']), 'available mask')
        base = arrays['base_logei']
        bo = select(base, available, ties)
        epsilon = log(4)*(16-t)/15
        eligible = available & (base[bo]-base <= epsilon)
        require(np.array_equal(eligible, arrays['eligible']), 'qLogEI opportunity set')
        selected = pool['ix'][c['selected_id']]
        require(c['BO_reference_id'] == pool['ids'][bo] and eligible[selected], 'BO argmax or selected budget')
        require(c['eligible_count'] == int(eligible.sum()), 'eligible count')
        for field, value in [('epsilon', epsilon), ('base_logei', base[selected]), ('reference_logei', base[bo]), ('opportunity_gap', base[bo]-base[selected])]:
            close(c[field], value, 'commit '+field)
        common.native_seed_audit(pool, c)
        require(c['GP_fit_key'] == c['GP_acquisition_meta']['fit_key'], 'GP acquisition fit key')
        fit = read(cp.with_name(f'{t:02d}_fit.json'))
        require(fit and all(np.isfinite(np.asarray(x)).all() for x in fit.values()), 'nonfinite fit receipt')
        if joint: saved_gp_audit(folder, pool, prefix, c, arrays, fit)
        prepared, events, pre_rows = refresh(pool, previous, prefix, arrays, t)
        exact(c['pre_call_coverage_events'], events, 'pre-call coverage transition')
        want_call = state['mode'] != 'bo' and should_call(calls, decisions[:t-1], pre_rows, t, params)
        expected_menu = candidate_menu(pool, prefix, arrays, t) if want_call else []
        exact(c['menu_ids'], expected_menu, 'semantic candidate menu')
        require(c['calls_at_commit'][:len(calls)] == calls and len(c['calls_at_commit']) == len(calls)+int(want_call and bool(expected_menu)), 'LLM event schedule/call prefix')
        if len(c['calls_at_commit']) > len(calls):
            call = c['calls_at_commit'][-1]
            require(call['query'] == t, 'LLM call current round')
            prompt_audit(folder, protocol, pool, call, prefix, c, prepared, pre_rows)
        else:
            require(c['proposal'] is None, 'proposal without a scheduled call')
        calls = c['calls_at_commit']
        hyps = deepcopy(c['hypotheses_at_commit'])
        validate_hypotheses(pool, hyps, prepared, prefix, t, c['menu_ids'])
        declarations_audit(c)
        for h in hyps:
            if h['status'] == 'retracted': continue
            ps = probes(pool, h, prefix, eligible)
            require((h['status'] == 'active') == bool(ps), 'rule dormancy/executable probes')
        calc, err = selection_audit(pool, prefix, c, arrays, state['mode'], modes['optional_verification'], params)
        max_quad_error = max(max_quad_error, err)
        expected_shadows = [hkey(h) for h in hyps if hkey(h) in calc and selected in calc[hkey(h)]['probes']]
        require([hkey(s) for s in c['shadows']] == expected_shadows, 'missing/extraneous new-plus forecasts')
        require(len(c['shadows']) == len(decision['evidence']), 'forecast/receipt count')
        weights = {hkey(x): x['effective_mixture_weight'] for x in c['deployed_versions']}
        expected_mixture = [dict(id=s['id'], version=s['version'], weight=weights[hkey(s)], prediction_sha256=s['prediction']['prediction_sha256'])
                            for s in c['shadows'] if weights.get(hkey(s), 0) > 0]
        exact(c['selected_mixture'], expected_mixture, 'selected actual deployed mixture')
        scores = {}
        for sh, receipt in zip(c['shadows'], decision['evidence']):
            pred_sha = sh['prediction']['prediction_sha256']
            require(pred_sha not in seen_predictions, 'prediction reused')
            seen_predictions.add(pred_sha)
            h = next(h for h in hyps if hkey(h) == hkey(sh))
            scores[pred_sha] = shadow_feedback(pool, h, sh, receipt, observation, prefix, arrays, calc[hkey(sh)], t)
            forecasts += 1
        mixture = c['selected_mixture']
        total = sum(r['weight'] for r in mixture)
        actual_gain = float(logsumexp([log(1-total)]+[log(r['weight'])+scores[r['prediction_sha256']] for r in mixture])) if mixture else 0.
        close(decision['deployed_mixture_log_score_gain'], actual_gain, 'actual deployed mixture logscore')
        require(actual_gain >= -log(2)-ATOL, 'half-GP mixture predictive logloss bound')
        modes[c['mode']] += 1
        roles.append(dict(query=t, mode=c['mode'], selected_id=c['selected_id'], BO_reference_id=c['BO_reference_id'],
            deployed_experts=len(c['selected_mixture']), forecasts=len(c['shadows']), mixture_gain=actual_gain,
            positive_prediction_failed_fact=sum(e['delta_log_score'] > 0 and not e['fact_success'] for e in decision['evidence']),
            **transfer_role(pool, c)))
        previous = hyps
    exact(state['hypotheses'], previous, 'state final rule ledgers differ from independent replay')
    exact(state['calls'], calls, 'state final call history')
    require(state['verification_used'] == modes['optional_verification'] <= params['maximum_optional_verifications'], 'optional query budget')
    require(state.get('cache_counters', {}).get('operator_compiles_in_loop', 0) == 0, 'operators compiled during experiment loop')
    raw = np.array([h['raw_value'] for h in history])
    curve = np.maximum.accumulate(raw)
    return dict(trial_seed=trial, tie_seed=tie_seed, numerical_seed=0, menu_seed=0, task=state['task'], group=pool['group'], mode=state['mode'], condition=state['condition'], status=state['status'],
        technical_checks='PASS_LISTED_CHECKS', queries=len(decisions), forecasts=forecasts, selection_modes=dict(modes),
        initial_history_sha256=digest(history[:4]), SearchAUC=float(curve[4:].mean()), final_best=float(curve[-1]),
        first_best_total=int(np.flatnonzero(raw == curve[-1])[0])+1,
        argmax_scope='FULL_CANDIDATE_RULE_AND_OPTIONAL_SELECTION', saved_GP_and_MC512_recomputed=bool(joint),
        max_selected_BO_quad_logratio_error=max_quad_error, roles=roles,
        transfer_usage={k:sum(row[k] for row in roles) for k in ('transfer_forecasts','weighted_transfer_forecasts',
                       'optional_target_transfers','guided_point_has_transfer_expert')},
        run=path.relative_to(ROOT).as_posix())


def write_report(report, destination):
    import json
    destination=Path(destination).resolve()
    allowed=(ROOT/'experiments/reports/knowledge_loop_20261101').resolve()
    require(destination.is_relative_to(allowed),'confirmation audit output outside reports')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return destination


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry-only',action='store_true')
    parser.add_argument('--joint',action='store_true')
    parser.add_argument('--include-incomplete',action='store_true')
    parser.add_argument('--run',action='append')
    parser.add_argument('--task',action='append')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    folder=ROOT/'experiments/runs/knowledge_loop_20261101'/VARIANT
    protocol=read(folder/'protocol.json') if (folder/'protocol.json').exists() else None
    if protocol is not None:
        require(protocol['protocol_sha256']==digest({k:v for k,v in protocol.items() if k!='protocol_sha256'}),'confirmation protocol self SHA')
        for path,sha in protocol['source_sha256'].items():
            require(filehash(ROOT/path)==sha,'frozen confirmation source changed: '+path)
    registry,pools,manifests=registration_audit(ROOT,protocol)
    if args.registry_only:
        report=dict(created_at=datetime.now(timezone.utc).isoformat(),variant=VARIANT,
            technical_result='PASS_INPUT_ONLY_REGISTRATION',auditor_sha256=filehash(Path(__file__)),
            protocol_sha256=protocol['protocol_sha256'] if protocol else None,
            registry=registry,performance_goal_achieved=None,experiments_audited=0)
        destination=args.output or ROOT/'experiments/reports/knowledge_loop_20261101/v5_seed2_registration_independent_audit.json'
        written=write_report(report,destination)
        print(json.dumps(dict(result=report['technical_result'],report=str(written)),ensure_ascii=False))
        return 0
    require(protocol is not None,'confirmation protocol is not yet frozen')
    source=common.source_order_audit(ROOT)
    require(source['initial_commit_before_initial_oracle'],'confirmation initial commitment order')
    rows,failures,pending=[],[],[]
    for path in sorted((folder/'runs').glob('*/state.json')):
        state=read(path)
        if args.run and path.parent.name not in args.run:continue
        if args.task and state['task'] not in args.task:continue
        if state['status']!='COMPLETE':
            pending.append(dict(run=path.parent.name,status=state['status'],queries=len(state['decisions'])))
            if not args.include_incomplete or not state['decisions']:continue
        try:
            rows.append(audit_run(folder,path,protocol,pools[state['task']],manifests[state['task']],args.include_incomplete,args.joint))
            print(json.dumps(dict(audited=path.parent.name,queries=rows[-1]['queries'],result='PASS_LISTED_CHECKS'),ensure_ascii=False),flush=True)
        except Exception as error:
            failures.append(dict(run=path.parent.name,error_type=type(error).__name__,error=str(error)))
            print(json.dumps(dict(audited=path.parent.name,result='FAIL',error=str(error)),ensure_ascii=False),flush=True)
    baselines={r['task']:r for r in rows if r['mode']=='bo' and r['status']=='COMPLETE'}
    for row in rows:
        if row['mode']=='bo' or row['status']!='COMPLETE' or row['task'] not in baselines:continue
        bo=baselines[row['task']]
        require(row['initial_history_sha256']==bo['initial_history_sha256'],'seed2 matched initial observations differ')
        row['AUC_delta_vs_matched_BO']=row['SearchAUC']-bo['SearchAUC']
        row['final_delta_vs_matched_BO']=row['final_best']-bo['final_best']
    compared=[r for r in rows if r['mode']=='full' and 'AUC_delta_vs_matched_BO' in r]
    helper_paths=[Path(v5.__file__),Path(v5.v4.__file__),Path(common.__file__)]
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),variant=VARIANT,
        auditor_sha256=filehash(Path(__file__)),audit_helper_sha256={p.relative_to(ROOT).as_posix():filehash(p) for p in helper_paths},
        protocol_sha256=protocol['protocol_sha256'],registration=registry,source_order=source,source_drift=[],
        technical_result='PASS_LISTED_CHECKS' if rows and not failures else 'FAIL_OR_NO_AUDITED_RUNS',
        performance_goal_achieved=None,performance_interpretation='One preregistered paired initialization on the same domains; report performance separately from technical PASS.',
        read_only=True,fits=0,LLM_calls=0,oracle_files_opened=0,unqueried_targets_read=0,joint=bool(args.joint),
        runs=rows,failures=failures,pending=pending,
        totals=dict(runs=len(rows),rounds=sum(r['queries'] for r in rows),forecasts=sum(r['forecasts'] for r in rows)),
        comparisons=[dict(mode='full',matched_tasks=len(compared),
            strict_SearchAUC_wins=sum(r['AUC_delta_vs_matched_BO']>1e-9 for r in compared),
            SearchAUC_ties=sum(abs(r['AUC_delta_vs_matched_BO'])<=1e-9 for r in compared),
            SearchAUC_losses=sum(r['AUC_delta_vs_matched_BO']<-1e-9 for r in compared),
            final_best_losses=sum(r['final_delta_vs_matched_BO']<-1e-9 for r in compared))],
        limitations=['Registration and commit hashes prove internal custody/order, not external timestamp attestation.',
            'Input/schema/source checks are not an OS-level I/O trace; no oracle file was opened.',
            'Saved GP parameters are independently conditioned and MC/decisions replayed, not refitted.',
            'GH64 is approximate; selected and executable BO forecasts receive independent adaptive F-integral checks.',
            'No policy selection or threshold tuning is allowed using this confirmation; this is not an unseen-dataset or multi-seed statistical guarantee.',
            *([] if args.joint else ['Stored GP/MC values were not rebuilt; complete rule argmax uses saved arrays.'])])
    destination=args.output or ROOT/'experiments/reports/knowledge_loop_20261101/v5_seed2_confirm_joint_audit.json'
    written=write_report(report,destination)
    print(json.dumps(dict(result=report['technical_result'],totals=report['totals'],report=str(written)),ensure_ascii=False),flush=True)
    return int(bool(failures) or not rows)


if __name__=='__main__':
    raise SystemExit(main())

