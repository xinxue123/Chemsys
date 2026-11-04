"""Audit a new EI numeric component against all archived executable distributions.

Uses frozen V5/V7/seed2 own prefixes and saved GP arrays only. No fitting, API,
oracle, new decisions, or online state writes. Fixed-prefix selection comparisons
do not pretend to be a new sequential rollout: changing scores could change the
future ledger or LLM context. Original failed audits remain untouched.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from math import exp, log, pi, sqrt
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import log_ndtr, logsumexp

import audit_knowledge_transfer_loop as old

ROOT = old.ROOT
sys.path.insert(0, str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_conditional_ei as numeric

REPORTS = ROOT/'experiments/reports/knowledge_loop_20261101'
VARIANTS = ('v5', 'v7', 'v5_seed2_confirm')
ATOL = 1e-10
INDEPENDENT_ERROR_LIMIT = ATOL / 10
TASKS = tuple(f'Reizman_Suzuki_Case_{i}' for i in range(1,5)) + (
    'Perera_Suzuki','Doyle_Buchwald_Hartwig','SnAr','HPLC')


def complete_run_manifest(protocol, paths):
    """Fail closed if a directory listing omits a registered experiment."""
    if protocol['protocol_sha256']!=old.digest({k:v for k,v in protocol.items() if k!='protocol_sha256'}):
        raise AssertionError('Protocol content hash differs')
    if tuple(protocol['tasks'])!=TASKS or len(protocol['inputs'])!=8 or {x['task'] for x in protocol['inputs']}!=set(TASKS):
        raise AssertionError('Incomplete registered eight-task manifest')
    expected={f'{task}_full_agent_initial_terra' for task in TASKS}
    actual=[p.parent.name for p in paths]
    if len(actual)!=8 or set(actual)!=expected:
        raise AssertionError('Missing, extra, or duplicated full trajectories')


def require_independent_precision(difference, independent_error, component_error):
    values=(difference,independent_error,component_error)
    if not all(np.isfinite(x) and x>=0 for x in values):
        raise ArithmeticError('Nonfinite numerical comparison/error estimate')
    if independent_error>INDEPENDENT_ERROR_LIMIT or component_error>ATOL:
        raise ArithmeticError('Declared component/independent precision not resolved')
    if difference>ATOL+independent_error or difference>component_error+independent_error:
        raise ArithmeticError('Independent discrepancy exceeds stated error budget')


def independent_log_ratio(mu, vf, vy, reference, best, tau, expected):
    """Different F-coordinate integral, with analytic base EI normalization.

    The expected value is only a multiplicative integrand scale. Both the base
    normalization and conditional integral are independently evaluated. This
    check has QUADPACK estimates, not formal interval arithmetic guarantees.
    """
    if vf == 0:
        return dict(log_ratio=0., error_estimate=0., base_normalization=1., method='zero_latent')
    sd=sqrt(vf); h=(best-mu)/sd; scale=1/max(1., h)
    lb=float(old.v4.normal_log_ei(np.asarray(mu),np.asarray(vf),np.asarray(best)))
    lp=float(log_ndtr((mu-reference)/sqrt(vy+tau*tau)))
    def integrate(conditional):
        def function(u):
            if u == 0:return 0.
            dz=scale*u; z=h+dz
            value=log(sd)+log(dz)-z*z/2-log(2*pi)/2-lb+log(scale)
            if conditional:
                value+=float(log_ndtr((best+sd*dz-reference)/sqrt(vy-vf+tau*tau)))-lp-expected
            return exp(value)
        center_z=0.
        if conditional:
            a=(mu-reference)/sqrt(vy+tau*tau)
            mills=exp(-a*a/2-log(2*pi)/2-lp)
            center_z=sd/sqrt(vy+tau*tau)*mills
        center=max(0.,(center_z-h)/scale)
        lo,hi=(max(0.,center-18/scale),center+18/scale) if center>20 else (0.,np.inf)
        with warnings.catch_warnings():
            warnings.simplefilter('error',IntegrationWarning)
            return quad(function,lo,hi,epsabs=2e-12,epsrel=2e-12,limit=400)
    base,eb=integrate(False); conditional,ec=integrate(True)
    if abs(base-1)>1e-9 or conditional<=0 or not np.isfinite(conditional):
        raise ArithmeticError(f'Independent integral normalization failed: {base}, {conditional}')
    return dict(log_ratio=expected+log(conditional),error_estimate=ec/conditional+eb,
                base_normalization=base,method='independent_scaled_F')


def fixed_prefix_choice(c, arrays, rows, calc, used, parameters):
    """Reconstruct only this prefix's choices, keeping its actual old ledger."""
    base,eligible,ties=arrays['base_logei'],arrays['eligible'],arrays['tie_priority']
    bo=old.select(base,eligible,ties); combined=base.copy()
    deployed=c['deployed_versions']
    if deployed:
        weights=[d['effective_mixture_weight'] for d in deployed]
        offsets=np.stack([calc[old.hkey(d)]['offsets'] for d in deployed])
        correction=logsumexp(np.vstack([np.full_like(base,log(1-sum(weights)))]+
            [log(w)+v for w,v in zip(weights,offsets)]),axis=0)
        correction[np.all(offsets<=old.ATOL,axis=0)]=0.
        combined+=correction
    selected=old.select(combined,eligible,ties)
    weighted=selected; mode='knowledge_guided' if selected!=bo else 'base_bo'
    candidates=[]
    if selected==bo and c['query']<=parameters['last_verification_round'] and used<parameters['maximum_optional_verifications']:
        for row in rows:
            key=old.hkey(row)
            if row['weight']>0 or not row['can_change_rank_at_cap'] or key not in calc:continue
            h=next(h for h in c['hypotheses_at_commit'] if old.hkey(h)==key)
            evidence=old.evidence(h)
            if evidence['effective_prediction_score']<0 or evidence['failures']>evidence['successes']:continue
            values=calc[key]['values']; pos=calc[key]['positions'].get(bo)
            base_info=float(values['sign_information'][pos]) if pos is not None else 0.
            for position,j in enumerate(calc[key]['indices']):
                gap=float(base[bo]-base[j]); info=float(values['sign_information'][position])
                if (j!=bo and gap<=min(c['epsilon'],parameters['verification_maximum_gap'])
                    and info-base_info>=parameters['minimum_incremental_information']
                    and values['offset'][position]>old.ATOL):
                    candidates.append((info-base_info,-gap,-ties[j],int(j),key))
        if candidates:
            selected=max(candidates,key=lambda x:x[:3])[3];mode='optional_verification'
    return dict(selected_index=int(selected),weighted_index=int(weighted),mode=mode,
                optional_candidates=len(candidates))


def main():
    start=time.perf_counter(); numeric.warm()
    pairs=[];rounds=[];failures=[];states=[];custody=[]
    for variant in VARIANTS:
        folder=ROOT/'experiments/runs/knowledge_loop_20261101'/variant
        protocol=old.read(folder/'protocol.json'); manifests={x['task']:x for x in protocol['inputs']}
        run_paths=sorted((folder/'runs').glob('*_full_agent_initial_terra/state.json'))
        complete_run_manifest(protocol,run_paths)
        pool_hashes={}
        for manifest in manifests.values():
            inputs=[(rel,sha) for rel,sha in manifest['files_sha256'].items() if rel.endswith('/pool.json')]
            assert len(inputs)==1,'Expected exactly one candidate-only pool per task'
            rel,sha=inputs[0];measured=old.filehash(ROOT/rel)
            assert measured==sha,'Candidate-only pool content differs from frozen manifest'
            pool_hashes[manifest['task']]=dict(path=rel,sha256=measured)
        custody.append(dict(variant=variant,protocol_sha256=protocol['protocol_sha256'],
            protocol_file_sha256=old.filehash(folder/'protocol.json'),candidate_pools=pool_hashes))
        for path in run_paths:
            state=old.read(path);assert state['status']=='COMPLETE' and len(state['decisions'])==16
            assert state['task'] in manifests and state['mode']=='full' and len(state['history'])==20
            assert path.parent.name==state['task']+'_full_agent_initial_terra','State belongs to another task directory'
            assert state['protocol_sha256']==protocol['protocol_sha256']
            pool=old.common.public_pool(ROOT,manifests[state['task']]);used=0
            states.append(dict(variant=variant,task=state['task'],sha256=old.filehash(path)))
            for t in range(1,17):
                cp=path.parent/'rounds'/f'{t:02d}_commit.json';c=old.read(cp);prefix=state['history'][:t+3]
                assert old.digest(prefix)==c['history_sha256']
                assert old.digest(c)==state['decisions'][t-1]['commit_sha256']
                with np.load(cp.with_name(f'{t:02d}_GP.npz'),allow_pickle=False) as z:
                    arrays={k:z[k].copy() for k in z.files}
                rows,calc=old.calculations(pool,c['hypotheses_at_commit'],prefix,arrays)
                old.exact(rows,c['coverage'],'archived prefix coverage')
                baseline=fixed_prefix_choice(c,arrays,rows,calc,used,protocol['parameters'])
                assert pool['ids'][baseline['selected_index']]==c['selected_id'] and baseline['mode']==c['mode']
                changed=deepcopy(calc); changed_rows=deepcopy(rows)
                base=arrays['base_logei'];best=max(x['y'] for x in prefix)
                for key,data in calc.items():
                    js=data['indices'];ref=np.asarray([data['probes'][int(j)]['reference_y'] for j in js])
                    values=numeric.conditional_ei_ratio(arrays['mu_y'][js],arrays['sd_y'][js]**2,
                        arrays['sd_predictive_y'][js]**2,ref,best,atol=ATOL)
                    for pos,j in enumerate(js):
                        args=[float(arrays['mu_y'][j]),float(arrays['sd_y'][j]**2),float(arrays['sd_predictive_y'][j]**2),float(ref[pos]),best,.06]
                        lr=float(values['log_ratio'][pos]); independent=independent_log_ratio(*args,expected=lr)
                        error=abs(lr-independent['log_ratio'])
                        before=float(data['values']['pure_log_ei_ratio'][pos])
                        pair=dict(variant=variant,task=state['task'],query=t,rule=list(key),point_id=pool['ids'][int(j)],
                            input_distribution=args,old_GH64_log_ratio=before,new_log_ratio=lr,
                            GH64_difference=lr-before,independent=independent,independent_difference=error,
                            numerical_error_estimate=float(values['log_ratio_error_estimate'][pos]),
                            method=str(values['method'][pos]),old_tempering=float(data['values']['tempering'][pos]),
                            new_tempering=float(values['tempering'][pos]))
                        pairs.append(pair)
                        try:
                            require_independent_precision(error,independent['error_estimate'],pair['numerical_error_estimate'])
                        except ArithmeticError as problem:
                            failures.append(dict(pair,reason=str(problem)))
                    for name in ('offset','tempering','log_tempering'):
                        changed[key]['values'][name]=np.asarray(values[name])
                    offsets=np.zeros_like(base);offsets[js]=values['offset'];changed[key]['offsets']=offsets
                    row=next(r for r in changed_rows if old.hkey(r)==key)
                    correction=np.logaddexp(log(.5),log(.5)+offsets);correction[offsets<=old.ATOL]=0.
                    bo=pool['ix'][c['BO_reference_id']]
                    row['can_change_rank_at_cap']=old.select(base+correction,arrays['eligible'],arrays['tie_priority'])!=bo
                    row['positive_effect_probes']=int(np.count_nonzero(values['offset']>old.ATOL))
                new=fixed_prefix_choice(c,arrays,changed_rows,changed,used,protocol['parameters'])
                score_changes=[]
                for sh in c['shadows']:
                    key=old.hkey(sh);j=pool['ix'][c['selected_id']];pos=calc[key]['positions'][j]
                    before=sh['prediction'];after=deepcopy(before)
                    after['tempering']=float(changed[key]['values']['tempering'][pos])
                    lt=float(changed[key]['values']['log_tempering'][pos]);after['log_tempering']=lt if np.isfinite(lt) else None
                    y=state['history'][t+3]['y']
                    score_changes.append(dict(rule=list(key),old_logscore=old.score(before,y),
                        fixed_prefix_new_logscore=old.score(after,y),difference=old.score(after,y)-old.score(before,y)))
                rounds.append(dict(variant=variant,task=state['task'],query=t,commit_sha256=old.digest(c),
                    GP_sha256=old.filehash(cp.with_name(f'{t:02d}_GP.npz')),old=baseline,new=new,
                    point_changed=new['selected_index']!=baseline['selected_index'],mode_changed=new['mode']!=baseline['mode'],
                    cap_gate_changes=sum(a['can_change_rank_at_cap']!=b['can_change_rank_at_cap'] for a,b in zip(rows,changed_rows)),
                    feedback_score_changes=score_changes))
                used+=int(c['mode']=='optional_verification')
            print(json.dumps(dict(variant=variant,task=state['task'],prefixes=16,stage='AUDITED'),ensure_ascii=False),flush=True)
    summary=dict(status='PASS_LISTED_NUMERICAL_CHECKS' if not failures else 'FAIL_NUMERICAL_CHECKS',
        runs=len(states),prefixes=len(rounds),rule_point_distributions=len(pairs),
        maximum_independent_difference=max(p['independent_difference'] for p in pairs),
        maximum_old_GH64_difference=max(abs(p['GH64_difference']) for p in pairs),
        old_GH64_beyond_original_2e_5=sum(abs(p['GH64_difference'])>2e-5 for p in pairs),
        fixed_prefix_point_changes=sum(r['point_changed'] for r in rounds),
        fixed_prefix_mode_changes=sum(r['mode_changed'] for r in rounds),
        cap_gate_changes=sum(r['cap_gate_changes'] for r in rounds),
        frozen_forecasts=sum(len(r['feedback_score_changes']) for r in rounds),
        maximum_fixed_prefix_logscore_change=max(abs(e['difference']) for r in rounds for e in r['feedback_score_changes']),
        methods=dict(Counter(p['method'] for p in pairs)),elapsed_seconds=time.perf_counter()-start)
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),summary=summary,
        numerical_component_sha256=old.filehash(numeric.__file__),auditor_sha256=old.filehash(__file__),
        tolerance=ATOL,independent_error_limit=INDEPENDENT_ERROR_LIMIT,
        independent_comparison_rule='difference <= component_estimate + independent_estimate, component<=atol, independent<=atol/10',
        variants=list(VARIANTS),custody=custody,
        fits=0,LLM_calls=0,oracle_access=False,new_optimization_runs=0,
        original_strict_V7_audit_unchanged_sha256=old.filehash(REPORTS/'v7_joint_audit.json'),
        states=states,pairs=pairs,rounds=rounds,failures=failures,
        limitations=['Only archived executable distributions, not every possible future GP input.',
            'Fixed-prefix numeric counterfactuals keep the original credit/LLM state and are not new online trajectories.',
            'A changed feedback score may alter later credit or public context even if each fixed-prefix selected point is unchanged.',
            'Internal adaptive quadrature estimates and floating point are not formal interval certificates.',
            'Original V7 precision FAIL is retained; this report validates a new unconnected numeric component.'])
    out=REPORTS/'conditional_ei_archived_corpus_review.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['条件EI新数值组件：冻结分布全量回归（不重跑优化）',json.dumps(summary,ensure_ascii=False,indent=2),
        '只使用已冻结的GP/规则/自身前缀；未读未查询目标，没有新增LLM、拟合或实验。',
        '局部对比保持当轮原信用不变，不得称为新在线轨迹或新增性能优势。原V7严格审计FAIL保留。',
        '全部分布、独立积分误差、门限/选点和反馈差异见JSON。']
    out.with_suffix('.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
