"""Single-process debugger for the frozen regional Suzuki ablation.

Run from an IDE, or: .venv-official/Scripts/python.exe scripts/debug_suzuki_case.py --case 2
Default: all 16 trajectories, exact recorded LLM replay, no network calls.
All writes go to a NEW outputs/suzuki_debug directory. Scientific code is reused.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from contextlib import ExitStack
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from chem.benchmarks import regional_ablation_suzuki_engine as engine
from chem.benchmarks import regional_ablation_suzuki_study as study
from chem.benchmarks.suzuki_domain import digest
from scripts.suzuki_debug_view import DebugBook, annotate, filter_preview, REASONS, TRUTH

# IDE users can edit these defaults; command-line arguments take precedence.
DEFAULT_CASE = 4
DEFAULT_REGIMES = ['assistant_initial']
DEFAULT_METHODS = ['G3_gate_KF_B4_event']
DEFAULT_ROUNDS = 16
DEFAULT_LLM_MODE = 'replay'
DEFAULT_PDB_ROUND = None
DEFAULT_PAUSE = ['kf']  # e.g. ['kf', 'round']; use ['stages', 'kf', 'round'] to pause at every display stage


class ReplayMismatch(ValueError):
    """Fail closed: an archived LLM response is valid only for its exact context."""


class DebugCache(engine.RegionalAblationCache):
    def _evaluate(self, history, tilt, value, fit, key, amplitude, spec):
        # Cache transport only. Same frozen score evaluator and same cache key.
        self.ensure(key + '.npz')
        return super()._evaluate(history, tilt, value, fit, key, amplitude, spec)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def row_dict(row):
    return {name: scalar(row[name]) for name in row.dtype.names}


def ranked(scores, mask, ids):
    result = np.zeros(len(ids), dtype=np.int32)
    indices = np.flatnonzero(mask)
    order = indices[np.lexsort((ids[indices], -scores[indices]))]
    result[order] = np.arange(1, len(order) + 1)
    return result


def candidate_view(cache, state, pool, decision):
    """Inspectable prequery table. No unobserved oracle values enter this table."""
    ids = np.asarray(pool['ids'])
    available, eligible = decision['available'], decision['eligible']
    base, guided = decision['base'], decision['score']
    raw_scores, _, _ = cache.field(state['history'], state['knowledge'], decision['retained'])
    _, fit = cache.model(state['history'])
    ys = float(fit['std'])
    mu = decision['stats']['mu_y'] * 100
    raw = decision['raw_shift_std'] * ys * 100
    deployed = decision['shift_std'] * ys * 100
    masks = engine.membership(pool, state['knowledge'])
    keep = [i for i, h in enumerate(state['knowledge']['clauses']) if h['id'] in decision['retained']]
    reason = np.full(len(ids), 'eligible_lower_guided_score', dtype='U40')
    reason[~available] = 'already_queried'
    reason[available & ~eligible] = 'outside_base_qLogEI_budget'
    tie = eligible & (guided == guided[decision['selected']])
    reason[tie] = 'eligible_tie_larger_candidate_id'
    reason[decision['selected']] = 'selected'
    columns = {
        'candidate_id': ids,
        'available': available, 'eligible': eligible,
        'matches_all_knowledge': masks.all(axis=0),
        'matches_retained_region': masks[keep].all(axis=0),
        'GP_mean_percent': mu,
        'GP_latent_SD_pp': decision['stats']['sd_y'] * 100,
        'GP_observation_SD_pp': decision['stats']['sd_predictive_y'] * 100,
        'acquisition_SD_with_jitter_pp': np.sqrt(decision['stats']['sd_y']**2 + ys**2 * 1e-6) * 100,
        'raw_region_shift_pp': raw,
        'mean_after_region_percent': mu + raw,
        'KF_amplitude': np.full(len(ids), decision['amplitude']),
        'deployed_shift_pp': deployed,
        'mean_after_KF_percent': mu + deployed,
        'base_qLogEI': base,
        'region_unscaled_qLogEI': raw_scores,
        'guided_qLogEI': guided,
        'base_qLogEI_loss': base[decision['reference']] - base,
        'budget_logEI': np.full(len(ids), decision['budget']),
        'base_rank_available': ranked(base, available, ids),
        'guided_rank_available': ranked(guided, available, ids),
        'guided_rank_eligible': ranked(guided, eligible, ids),
        'is_reference': np.arange(len(ids)) == decision['reference'],
        'is_selected': np.arange(len(ids)) == decision['selected'],
        'reason': reason,
    }
    table = np.rec.fromarrays(list(columns.values()), names=list(columns))
    selected = table[decision['selected']]
    assert selected['eligible'] and selected['guided_rank_eligible'] == 1
    assert engine.common.regional.argmax(guided, eligible, pool['ids']) == decision['selected']
    return table, fit


def debug_before_query(state, decision, candidate_table, gp_model, normalization, options):
    """IDE BREAKPOINT 1: table, GP, B4, mean shifts and rankings are all in scope."""
    selected_row = candidate_table[decision['selected']]  # <-- set an IDE breakpoint here
    if not options.pause and options.pdb_stage in ('before', 'both') and options.pdb_round == len(state['decisions']) + 1:
        breakpoint()  # candidate_table[candidate_table.candidate_id == 'p03631']
    return selected_row


def debug_after_query(state, candidate_table, options):
    """IDE BREAKPOINT 2: actual outcome and KF update are now available."""
    event = state['decisions'][-1]
    measurement, bank_after = event['measurement'], event['bank_after']  # <-- IDE breakpoint
    if not options.pause and options.pdb_stage in ('after', 'both') and options.pdb_round == event['iteration']:
        breakpoint()
    return measurement, bank_after


def pause_requested(options, kind, iteration):
    return kind in options.pause and (options.pdb_round is None or options.pdb_round==iteration)


def debug_stage(stage_name, candidate_table, stage_info, options):
    """IDE breakpoint: each displayed stage, with the full annotated table."""
    stage_path = stage_info['directory']  # <-- optional per-stage IDE breakpoint
    if pause_requested(options,'stages',stage_info['round']):
        breakpoint()
    return stage_path


def debug_kf_update(kf_info, bank, prior, options):
    """IDE BREAKPOINT: actual experiment observed; native KF update NOT run yet."""
    H, innovation, K = kf_info['H'], kf_info['innovation'], kf_info['K']  # <-- KF breakpoint
    if pause_requested(options,'kf',kf_info['iteration']):
        breakpoint()
    return H, innovation, K


def debug_llm_request(llm_messages, request_context, options):
    """IDE breakpoint: exact first request messages, before transport/replay."""
    observed_count = len(request_context['history'])
    if pause_requested(options,'stages',request_context['public']['iteration']):
        breakpoint()
    return observed_count


def debug_round_complete(state, candidate_table, round_info, output_directory, options):
    """IDE BREAKPOINT: every actual choice, all files and native KF output ready."""
    event, llm, kf = state['decisions'][-1], round_info['llm'], round_info['kf']
    stage_tables, selected_id = round_info['stages'], event['selected_id']  # <-- end-of-round breakpoint
    if pause_requested(options,'round',event['iteration']):
        breakpoint()
    return stage_tables, selected_id


def replay_refresh(original_get_refresh, archive, client, context, state, pool, folder):
    key = context['context_sha256']
    source = archive / 'requests' / key
    if not (source / 'validated.json').exists():
        raise ReplayMismatch('No exact archived LLM context. Replay stops; use --llm live for a changed policy/prefix.')
    saved_context = study.read(source / 'context.json')
    raw = deepcopy(context)
    raw.pop('context_sha256')
    # JSON roundtrip preserves content but stringifies numeric catalyst-map keys
    # and turns diagnosis tuples into lists, as the original archive writer does.
    canonical_context = json.loads(json.dumps(context, ensure_ascii=False, allow_nan=False))
    if saved_context != canonical_context or digest(raw) != key:
        raise ReplayMismatch('Archived LLM context does not exactly match the current observed prefix.')
    valid = study.read(source / 'validated.json')
    response_path = source / f"attempt{valid['accepted_attempt']:02d}" / 'response.json'
    if valid['context_sha256'] != key or study.filehash(response_path) != valid['response_sha256']:
        raise ReplayMismatch('Archived LLM response hash mismatch.')
    response = study.read(response_path)
    plan = engine.validate_refresh(json.loads(response['choices'][0]['message']['content']),
                                   state['knowledge'], state['slots'], len(state['history']), pool)
    if plan != valid['plan']:
        raise ReplayMismatch('Revalidated archived knowledge plan differs.')
    shutil.copytree(source, folder, dirs_exist_ok=True)
    # The original transport returns the exact validated record before any HTTP call.
    return original_get_refresh(None, context, state, pool, folder)


class Trace:
    def __init__(self, options, root, book):
        self.options, self.root = options, root
        self.book=book
        self.native_setup = engine.setup
        self.native_update_bank = engine.update_bank
        self.phases = []
        self.last_table = None
        self.book.on_stage=lambda name,table:debug_stage(name,table,
            {'directory':str(self.book.directory),'round':self.book.current['round'],
             'rule':next(s['rule'] for s in self.book.current['stages'] if s['id']==name)},self.options)

    def setup(self, cache, state, pool, iteration, knowledge=None):
        # F11 here enters the actual frozen decision mechanism, without a copy.
        decision = self.native_setup(cache, state, pool, iteration, knowledge=knowledge)
        candidate_table, fit = candidate_view(cache, state, pool, decision)
        # This separate COPY is for a human only. No annotated values are passed
        # to setup(), GP fitting, acquisition functions, public_request(), or LLM.
        candidate_table = annotate(candidate_table,pool,self.book.oracle_values)
        phase = len(self.phases) + 1
        stem = f'round{iteration:02d}_phase{phase}'
        directory = self.root / 'runs' / state['run_id'] / 'candidates'
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / (stem + '.npz'),
                            **{name: candidate_table[name] for name in candidate_table.dtype.names})
        if self.options.csv_all_candidates:
            write_csv(directory / (stem + '.csv'), [row_dict(row) for row in candidate_table])
        snapshot = {'iteration': iteration, 'phase': phase, 'final_for_round': False,
                    'history_before_query': deepcopy(state['history']),
                    'knowledge': deepcopy(state['knowledge']), 'B4': decision['diagnosis'].to_dict(),
                    'retained_ids': decision['retained'], 'bank_key': decision['bank_key'],
                    'bank_before': decision['bank'], 'bank_prior': decision['prior'],
                    'normalization': {'y_mean': float(fit['mean']), 'y_std': float(fit['std']), 'best_std': fit['best_std']},
                    'GP_parameters_raw': {k: v.detach().cpu().tolist() for k, v in fit['model'].named_parameters()},
                    'base_acquisition': decision['base_meta'], 'field_acquisition': decision['field_meta'],
                    'guided_acquisition': decision['guided_meta'],
                    'selected': row_dict(candidate_table[decision['selected']]),
                    'reference_id': pool['ids'][decision['reference']]}
        path = directory / (stem + '.json')
        write_json(path, snapshot)
        self.phases.append(path)
        self.last_table = candidate_table
        if phase==1:
            self.book.base_stages(candidate_table,decision,fit,state)
        will_refresh=(state['method']==engine.METHODS[3] and not any(c['iteration']==iteration for c in state['calls'])
                      and bool(engine.event_reasons(state,decision,iteration)))
        self.book.knowledge(state,decision,phase,will_refresh)
        self.book.publish()
        debug_stage('knowledge',candidate_table,{'directory':str(self.book.directory),'round':iteration,
            'rule':'当前知识及B4诊断；触发刷新时先查看请求，响应后再重新诊断。'},self.options)
        if not will_refresh:
            self.book.final_ranking(candidate_table,decision)
            for candidate_id in self.options.watch:
                found=candidate_table[candidate_table.candidate_id==candidate_id]
                if len(found):
                    row=found[0]
                    self.book.line(f"  WATCH {candidate_id}：μ={row.GP_mean_percent:.6f}%，真={row[TRUTH]:.6f}%，基础q={row.base_qLogEI:.9f}，最终q={row.guided_qLogEI:.9f}，预算={bool(row.eligible)}，预算内名次={row.guided_rank_eligible}；{REASONS[str(row.reason)]}。")
            debug_before_query(state, decision, candidate_table, fit['model'], snapshot['normalization'], self.options)
        return decision

    def update_bank(self, bank, prior, iteration, raw_shift_y, mu_y, predictive_sd_y, y):
        kf_info=filter_preview(bank,prior,iteration,raw_shift_y,mu_y,predictive_sd_y,y)
        self.book.kf_before(kf_info)
        debug_kf_update(kf_info,bank,prior,self.options)
        after,measurement=self.native_update_bank(bank,prior,iteration,raw_shift_y,mu_y,predictive_sd_y,y)
        self.book.kf_after(after,measurement)
        return after,measurement

    def finish_round(self):
        path = self.phases[-1]
        snapshot = study.read(path)
        snapshot['final_for_round'] = True
        write_json(path, snapshot)


def output_directory(requested, case):
    target = Path(requested).expanduser().resolve() if requested else REPO / 'outputs' / 'suzuki_debug' / (
        f'case{case}_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
    protected = [REPO / 'data' / 'runs', REPO / 'research', REPO / 'third_party', REPO / 'chem', REPO / 'tests']
    if any(target == p or p in target.parents for p in protected):
        raise ValueError('Debug output must be outside the scientific data/source archives; use outputs/suzuki_debug.')
    target.mkdir(parents=True, exist_ok=False)
    return target


def run(options):
    # Windows redirected stdout can otherwise be GBK and reject e.g. σ².
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    os.chdir(REPO)
    protocol = study.check_protocol()
    archive = (REPO / study.ROOT).resolve()
    manifest = [row for row in study.read(archive / 'manifest.json') if row['case'] == options.case
                and row['regime'] in options.regimes and row['method'] in options.methods]
    expected = {row['run_id']: study.read(archive / 'runs' / row['run_id'] / 'state.json') for row in manifest}
    root = output_directory(options.output, options.case)
    print('DEBUG_OUTPUT:', root, flush=True)
    write_json(root / 'debug_config.json', {**vars(options), 'output': str(root), 'protocol_sha256': protocol['protocol_sha256'],
               'source_archive': str(archive), 'seed': 0, 'init_queries': 4, 'scheduled_queries': 16,
               'note': 'Single-process instrumentation of original setup/step. replay=exact-context recorded LLM; live=new LLM feedback.'})
    study.worker_init()  # Single-thread GP/BLAS, warm kernels/Cholesky once.
    pool = study.inputs(options.case)
    with np.load(study.prior.original.ROOT / 'inputs' / f'case{options.case}' / 'oracle.npz', allow_pickle=False) as f:
        values = f['values'].copy()
    sources = [] if options.recompute else [archive / 'cache' / f'c{options.case}_s0',
        archive / 'qualification_cache' / f'c{options.case}_s0', archive / 'audit_cache' / f'c{options.case}_s0']
    if not options.recompute:
        sources += [REPO / source / folder / f'c{options.case}_s0'
                    for source in (study.prior.ROOT, study.prior.previous.ROOT)
                    for folder in ('cache', 'audit_cache', 'qualification_cache')]
    cache = DebugCache(pool, root / 'cache', options.case, sources=sources)
    write_csv(root / 'candidate_pool.csv', [{'candidate_id':key, **point,
              'GP_x0':float(cache.z[i,0]), 'GP_x1':float(cache.z[i,1]),
              'GP_x2':float(cache.z[i,2]), 'GP_categorical_label':float(cache.z[i,3])}
              for i,(key,point) in enumerate(zip(pool['ids'],pool['points']))])
    book=DebugBook(root,options,values,pool)
    trace = Trace(options, root,book)
    original_refresh = study.get_refresh
    client = None
    states, summary_rows = [], []
    prefix_checks = 0
    try:
        if options.llm == 'live' and engine.METHODS[3] in options.methods:
            client = study.api.APIClient(study.api.load_config(options.config))
        with ExitStack() as stack:
            stack.enter_context(patch.object(study, 'ROOT', root))
            stack.enter_context(patch.object(engine, 'setup', trace.setup))
            stack.enter_context(patch.object(engine,'update_bank',trace.update_bank))
            stack.enter_context(patch.object(study,'print',lambda *args,**kwargs:None,create=True))
            def observed_refresh(client,context,state,pool,folder):
                llm_messages=[{'role':'system','content':engine.SYSTEM_PROMPT},
                              {'role':'user','content':json.dumps(context['public'],ensure_ascii=False,separators=(',',':'))}]
                book.llm_start(context,llm_messages)
                debug_llm_request(llm_messages,context,options)
                try:
                    valid=(replay_refresh(original_refresh,archive,client,context,state,pool,folder)
                           if options.llm=='replay' else original_refresh(client,context,state,pool,folder))
                except Exception as error:
                    book.llm_finish(folder,error=error)
                    raise
                book.llm_finish(folder,valid=valid)
                return valid
            stack.enter_context(patch.object(study,'get_refresh',observed_refresh))
            for spec in manifest:
                state, path = study.state_for(spec, protocol, pool)
                states.append(state)
                for iteration in range(1, options.rounds + 1):
                    trace.phases = []
                    book.begin(state)
                    study.step(state, path, pool, cache, values, client)
                    trace.finish_round()
                    event = state['decisions'][-1]
                    same = (state['history'] == expected[state['run_id']]['history'][:4+iteration])
                    if options.llm == 'replay':
                        reference = expected[state['run_id']]
                        if not same or state['knowledge'] != reference['decisions'][iteration-1]['knowledge']:
                            raise ReplayMismatch(f"Selection/knowledge diverged at {state['run_id']}, added round {iteration}.")
                        prefix_checks += 1
                    row = row_dict(trace.last_table[trace.last_table.is_selected][0])
                    row = {'case':options.case, 'regime':state['regime'], 'method':state['method'],
                           'added_round':iteration, 'total_query':4+iteration, **row,
                           'eligible_count':int(trace.last_table.eligible.sum()),
                           'retained_ids':';'.join(event['retained_ids']),
                           'diagnosed_revoked_ids':';'.join(event['diagnosis']['revoked_ids']),
                           'phase_count':len(trace.phases),
                           'observed_yield_percent':event['measurement']['observed_y']*100,
                           'incumbent_percent':max(h['y'] for h in state['history'])*100,
                           'matches_archived_observation_prefix':same,
                           'final_snapshot':str(trace.phases[-1].relative_to(root))}
                    summary_rows.append(row)
                    write_csv(root / 'decision_summary.csv', summary_rows)
                    book.finish(state)
                    debug_after_query(state, trace.last_table, options)
                    debug_round_complete(state,trace.last_table,book.current,book.directory,options)
    finally:
        if client:
            client.close()
        # Even a debugger interruption restores module bindings and records progress.
        write_json(root / 'progress.json', {'trajectories_started':len(states), 'queries_finished':len(summary_rows),
                   'cache':cache.counters, 'replayed_prefix_checks':prefix_checks})
    study.check_protocol()  # Frozen scientific sources and archive inventories still match.
    receipt = {'status':'COMPLETE' if options.rounds==16 else 'PARTIAL_DEBUG_PREFIX',
               'case':options.case, 'trajectories':len(states), 'completed_trajectories':sum(s['status']=='COMPLETE' for s in states),
               'additional_queries':len(summary_rows), 'replayed_prefix_checks':prefix_checks,
               'llm_mode':options.llm, 'network_calls':0 if options.llm=='replay' else 'see requests/*/attempt*',
               'logical_refreshes':sum(len(s['calls']) for s in states), 'cache':cache.counters,
               'operator_compiles_in_loop':sum(s['operator_compiles_in_loop'] for s in states),
               'scientific_protocol_unchanged':True, 'output':str(root)}
    write_json(root / 'debug_receipt.json', receipt)
    book.directory=None
    book.line(f"\n调试完成：{len(states)}条轨迹，{len(summary_rows)}次选择，精确回放核对{prefix_checks}次。")
    book.line('打开逐轮查看器：'+str(root/'index.html'))
    return receipt


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', type=int, choices=(1,2,3,4), default=DEFAULT_CASE)
    parser.add_argument('--regimes', nargs='+', choices=('all',*study.REGIMES), default=DEFAULT_REGIMES)
    parser.add_argument('--methods', nargs='+', choices=('all',*engine.METHODS), default=DEFAULT_METHODS)
    parser.add_argument('--rounds', type=int, choices=range(1,17), default=DEFAULT_ROUNDS,
                        help='16 for complete trajectories; fewer rounds only truncate a debug prefix, not the gate schedule')
    parser.add_argument('--llm', choices=('replay','live'), default=DEFAULT_LLM_MODE)
    parser.add_argument('--config', default=str(REPO.parent / 'config.txt'))
    parser.add_argument('--output', help='New directory only; existing outputs/archives are never overwritten')
    parser.add_argument('--recompute', action='store_true', help='Fit GP and evaluate acquisitions again in a new cache')
    parser.add_argument('--csv-all-candidates', action='store_true', help='Also export 4692-row CSV per decision phase')
    parser.add_argument('--no-stage-csv', action='store_true', help='Keep the staged HTML tables; download CSV in viewer instead of prewriting four CSVs per query')
    parser.add_argument('--top',type=int,choices=range(0,11),default=3,help='Only this many leading candidates are printed per stage (all candidates stay in tables)')
    parser.add_argument('--print-llm',action='store_true',help='Also print complete LLM prompts/responses; always available in the viewer and text files')
    parser.add_argument('--pause',nargs='+',choices=('stages','kf','round'),default=DEFAULT_PAUSE,
                        help='PDB pause after displayed stages, before native KF update, and/or after every completed choice')
    parser.add_argument('--watch', nargs='*', default=[], help='Print the row of these candidate IDs at each phase')
    parser.add_argument('--pdb-round', type=int, choices=range(1,17), default=DEFAULT_PDB_ROUND)
    parser.add_argument('--pdb-stage', choices=('before','after','both'), default='before')
    options = parser.parse_args(argv)
    for name, complete in (('regimes',study.REGIMES),('methods',engine.METHODS)):
        selected = getattr(options,name)
        if 'all' in selected and len(selected)>1:
            parser.error(f'--{name}: use all alone')
        setattr(options,name,list(complete) if selected==['all'] else list(dict.fromkeys(selected)))
    return options


if __name__ == '__main__':
    run(arguments())
