"""Prefix-only V5/V6 optional-probe density and remaining-action diagnosis.

No production modules, optimizer, model fitting, LLM, or oracle files. The
already committed marginal GP and acquisition arrays remain fixed throughout.
Hypothetical positive feedback changes only the target credit; it does NOT
condition the GP, update references, shrink next-round epsilon, or predict AUC.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
from math import exp, log, pi, sqrt, tanh
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/tools'))
import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, logsumexp, ndtri_exp
import audit_knowledge_transfer_loop as a

OUT=ROOT/'experiments/reports/knowledge_loop_20261101'
LN2=log(2.); LHALF2=log(.5*LN2)
GH_X,GH_W=np.polynomial.hermite.hermgauss(64)
GH_X=np.sqrt(2.)*GH_X;GH_W=GH_W/np.sqrt(pi)


def log_fraction_complement(lt):
    return -np.inf if lt==0 else float(np.log1p(-exp(lt)))


def log_ratio(y,mu,vy,ref,lt,tau=.06):
    if not np.isfinite(lt):
        return np.zeros_like(np.asarray(y,float))
    lp=float(log_ndtr((mu-ref)/sqrt(vy+tau*tau)))
    return np.logaddexp(log_fraction_complement(lt),lt+log_ndtr((np.asarray(y)-ref)/tau)-lp)


def log_js_integrand_factor(ell):
    # f(L)=.5[L logL-(1+L)log((1+L)/2)] >=0. Stable at L~1
    # and at huge rare-event density ratios; no subtraction of two KL values.
    if ell==0:return -np.inf
    if ell>40:return ell+LHALF2
    if ell<-40:return LHALF2
    delta=np.expm1(ell)
    if abs(delta)<1e-3:
        f=delta*delta*(1/8-delta/16+7*delta*delta/192-3*delta**3/128+31*delta**4/1920)
    else:
        f=.5*(exp(ell)*ell-(2+delta)*np.log1p(delta/2))
    if f<0 and f>-1e-15:f=0.
    assert f>=0
    return log(f) if f else -np.inf


def integration_windows(mu,vy,ref,tau):
    sd=sqrt(vy);vz=vy+tau*tau;alpha=(mu-ref)/sqrt(vz)
    lp=float(log_ndtr(alpha));mills=exp(-.5*alpha*alpha-.5*log(2*pi)-lp)
    center=sqrt(vy/vz)*mills
    intervals=[(-16.,16.),(center-16.,center+16.)]
    if intervals[1][0]<=16:
        intervals=[(-16.,max(16.,intervals[1][1]))]
    pieces=[]
    for lo,hi in intervals:
        points=sorted(set([lo,hi]+[x for x in (0.,center,(ref-mu)/sd) if lo<x<hi]))
        pieces.extend(zip(points[:-1],points[1:]))
    return pieces


@lru_cache(maxsize=None)
def density_js_receipt(mu,vy,ref,lt,tau=.06):
    """Actual tempered observed-density JS in nats by fixed adaptive quadrature.

    X~N(mu,vy), L=e/g=(1-t)+t Phi((X-ref)/tau)/P(Z>0).
    JS=.5 E_g[(1+L)ln2+LlnL-(1+L)ln(1+L)].
    Finite windows cover both the original normal and truncated-sign expert;
    independent normalization/algebra checks are available in self-test.
    """
    assert vy>=0 and lt<=0 and tau>0
    if vy==0 or not np.isfinite(lt):return dict(JS=0.,quadrature_error_estimate=0.)
    sd=sqrt(vy)
    def integrand(z):
        ell=float(log_ratio(mu+sd*z,mu,vy,ref,lt,tau))
        return exp(-.5*z*z-.5*log(2*pi)+log_js_integrand_factor(ell))
    pieces=[quad(integrand,lo,hi,epsabs=2e-11,epsrel=2e-9,limit=160)
            for lo,hi in integration_windows(mu,vy,ref,tau)]
    value=sum(x[0] for x in pieces)
    assert -1e-12<=value<=LN2+2e-8,(mu,vy,ref,lt,value)
    return dict(JS=float(np.clip(value,0,LN2)),quadrature_error_estimate=sum(x[1] for x in pieces))


def density_js(mu,vy,ref,lt,tau=.06):
    return density_js_receipt(mu,vy,ref,lt,tau)['JS']


def density_check(mu,vy,ref,lt,tau=.06):
    if vy==0 or not np.isfinite(lt):return dict(normalization=1.,independent_JS=0.,error=0.)
    sd=sqrt(vy);windows=integration_windows(mu,vy,ref,tau)
    def functions(z):
        ell=float(log_ratio(mu+sd*z,mu,vy,ref,lt,tau));lg=-.5*z*z-.5*log(2*pi)
        normal=exp(lg);expert=exp(lg+ell);lm=float(np.logaddexp(0.,ell))
        return expert,.5*(normal*(LN2-lm)+expert*(LN2+ell-lm))
    norm=sum(quad(lambda z:functions(z)[0],lo,hi,epsabs=2e-10,epsrel=2e-9,limit=160)[0] for lo,hi in windows)
    independent=sum(quad(lambda z:functions(z)[1],lo,hi,epsabs=2e-10,epsrel=2e-9,limit=160)[0] for lo,hi in windows)
    value=density_js(mu,vy,ref,lt,tau)
    assert abs(norm-1)<2e-7,(norm,mu,vy,ref,lt)
    assert abs(independent-value)<2e-8,(independent,value)
    return dict(normalization=norm,independent_JS=independent,error=abs(independent-value))


def js_for(arrays,data,j):
    pos=data['positions'][int(j)];ref=data['probes'][int(j)]['reference_y']
    return density_js(float(arrays['mu_y'][j]),float(arrays['sd_predictive_y'][j]**2),
                      float(ref),float(data['values']['log_tempering'][pos]))


def score_rank(base,calculated,rows,target,weight,eligible,ties,target_alone=False):
    raw=[]
    for row in rows:
        key=a.hkey(row)
        w=weight if key==target else (0. if target_alone else row['weight'])
        if w>0 and key in calculated:raw.append((w,calculated[key]['offsets']))
    if not raw:return a.select(base,eligible,ties)
    total=sum(w for w,_ in raw);factor=min(1.,.5/total)
    ws=[w*factor for w,_ in raw]; offsets=np.stack([off for _,off in raw])
    corr=logsumexp(np.vstack([np.full_like(base,log(1-sum(ws)))]+[log(w)+off for w,off in zip(ws,offsets)]),axis=0)
    corr[np.all(offsets<=a.ATOL,axis=0)]=0.
    return a.select(base+corr,eligible,ties)


def positive_weight(h,t,point,ref,gain):
    before=a.evidence(h);changed=deepcopy(h)
    changed['ledger']['events'].append(dict(query_round=t,point_id=point,reference_id=ref,
        fact_success=True,delta_log_score=float(gain)))
    formula=a.evidence(changed)['formula_weight']
    ceiling=before['trusted_weight']+max(0.,tanh(gain/2))
    return min(formula,ceiling),dict(formula_weight=formula,recovery_ceiling=ceiling)


def review(run,protocol,pool,state,t):
    c=a.read(run/'rounds'/f'{t:02d}_commit.json');prefix=state['history'][:t+3]
    assert c['mode']=='optional_verification' and a.digest(prefix)==c['history_sha256']
    with np.load(run/'rounds'/f'{t:02d}_GP.npz',allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files}
    assert set(arrays)=={'mu_y','sd_y','sd_predictive_y','base_logei','available','eligible','tie_priority'}
    rows,calc=a.calculations(pool,c['hypotheses_at_commit'],prefix,arrays)
    base,eligible,ties=arrays['base_logei'],arrays['eligible'],arrays['tie_priority']
    bo,selected=pool['ix'][c['BO_reference_id']],pool['ix'][c['selected_id']]
    target=a.hkey(c['verification']);h=next(h for h in c['hypotheses_at_commit'] if a.hkey(h)==target)
    data=calc[target];p=data['positions'][selected]
    # Only actual frozen prefix/GP quantities, never state.decisions[t-1].y.
    original=[];all_target=[]
    for row in rows:
        key=a.hkey(row)
        if key not in calc:continue
        hh=next(hh for hh in c['hypotheses_at_commit'] if a.hkey(hh)==key)
        ev=a.evidence(hh);dd=calc[key];bp=dd['positions'].get(bo)
        bmi=float(dd['values']['sign_information'][bp]) if bp is not None else 0.
        bjs=js_for(arrays,dd,bo) if bp is not None else 0.
        eligible_rule=row['weight']==0 and row['can_change_rank_at_cap'] and ev['effective_prediction_score']>=0 and ev['failures']<=ev['successes']
        proposed={r['point_id'] for r in c.get('validation_proposals',[]) if
            (r['id'],r['version'],r['semantic_key'])==(hh['id'],hh['version'],hh['semantic_key'])}
        for pos,j in enumerate(dd['indices']):
            j=int(j);gap=float(base[bo]-base[j]);mi=float(dd['values']['sign_information'][pos]);dmi=mi-bmi
            passed=bool(eligible_rule and j!=bo and gap<=min(c['epsilon'],log(1.1)) and dmi>=.01 and dd['values']['offset'][pos]>a.ATOL
                        and (protocol['variant']!='v6' or pool['ids'][j] in proposed))
            if key!=target and not passed:continue
            js=js_for(arrays,dd,j)
            detail=dict(rule=list(key),point_id=pool['ids'][j],reference_id=dd['probes'][j]['reference_id'],
                sign_MI=mi,BO_sign_MI=bmi,delta_sign_MI=dmi,JS=js,BO_JS=bjs,delta_JS=js-bjs,
                gap=gap,original_gates_pass=passed,log_tempering=float(dd['values']['log_tempering'][pos]) if np.isfinite(dd['values']['log_tempering'][pos]) else None,
                offset=float(dd['values']['offset'][pos]),tie_priority=int(ties[j]))
            if key==target:all_target.append(detail)
            if passed:original.append(detail)
    assert original
    old=max(original,key=lambda x:(x['delta_sign_MI'],-x['gap'],-x['tie_priority']))
    new=max(original,key=lambda x:(x['delta_JS'],-x['gap'],-x['tie_priority']))
    assert old['point_id']==c['selected_id'] and tuple(old['rule'])==target
    remaining=eligible.copy();remaining[selected]=False
    cap_j=score_rank(base,calc,rows,target,.5,remaining,ties,True)
    rem_js=[int(j) for j in data['indices'] if int(j)!=selected]
    lt=float(data['values']['log_tempering'][p]);mu=float(arrays['mu_y'][selected]);vy=float(arrays['sd_predictive_y'][selected]**2)
    ref=data['probes'][selected];lp=float(data['values']['event_log_probability'][p])
    lc=log_fraction_complement(lt)
    gain_max=float(np.logaddexp(lc,lt-lp)) if np.isfinite(lt) else 0.
    gain_ref=float(np.logaddexp(lc,lt+log(.5)-lp)) if np.isfinite(lt) else 0.
    gain_min=max(0.,gain_ref)
    weight_max,wm=positive_weight(h,t,c['selected_id'],ref['reference_id'],gain_max)
    rows_sim=[]
    for fraction in (.25,.5,.75):
        gain=gain_min+fraction*(gain_max-gain_min)
        if gain<=0 or gain_max<=gain_min:continue
        log_cdf=lp+float(np.logaddexp(log(np.expm1(gain)) if gain else -np.inf,lt))-lt
        y=float(ref['reference_y']+.06*ndtri_exp(min(log_cdf,0.)))
        assert y>ref['reference_y'] and np.isfinite(y)
        assert abs(float(log_ratio(y,mu,vy,ref['reference_y'],lt))-gain)<1e-8
        w,bounds=positive_weight(h,t,c['selected_id'],ref['reference_id'],gain)
        alone=score_rank(base,calc,rows,target,w,remaining,ties,True)
        together=score_rank(base,calc,rows,target,w,remaining,ties,False)
        rows_sim.append(dict(positive_gain_fraction=fraction,hypothetical_y=y,gain=gain,weight=w,**bounds,
            target_only_winner=pool['ids'][alone],current_other_weights_winner=pool['ids'][together],
            target_only_nonBO=alone!=bo,current_other_weights_nonBO=together!=bo))
    maxalone=score_rank(base,calc,rows,target,weight_max,remaining,ties,True)
    maxtogether=score_rank(base,calc,rows,target,weight_max,remaining,ties,False)
    selected_detail=next(x for x in all_target if x['point_id']==c['selected_id'])
    check=density_check(mu,vy,ref['reference_y'],lt)
    # How often ordering differs within the same target rule, including points
    # outside the optional narrow gate. This is separate from the global test.
    order_mi=[x['point_id'] for x in sorted(all_target,key=lambda x:(-x['delta_sign_MI'],x['gap'],x['tie_priority']))]
    order_js=[x['point_id'] for x in sorted(all_target,key=lambda x:(-x['delta_JS'],x['gap'],x['tie_priority']))]
    return dict(variant=protocol['variant'],task=state['task'],query=t,target_rule=list(target),
        prefix_sha256=c['history_sha256'],commit_sha256=a.digest(c),GP_fit_key=c['GP_fit_key'],selected=c['selected_id'],BO=c['BO_reference_id'],
        remaining=dict(scope_candidates_before=next(r['eligible_inside_scope'] for r in rows if a.hkey(r)==target),
            executable_before=len(data['indices']),executable_after=len(rem_js),
            cap_target_only_winner_after=pool['ids'][cap_j],cap_target_only_nonBO_after=cap_j!=bo,
            achievable_positive_gain_supremum=gain_max,positive_fact_gain_lower_bound=gain_min,
            maximum_target_trust_supremum=weight_max,maximum_trust_formula=wm,
            maximum_trust_target_only_winner=pool['ids'][maxalone],maximum_trust_target_only_nonBO=maxalone!=bo,
            maximum_trust_current_other_weights_winner=pool['ids'][maxtogether],maximum_trust_current_other_weights_nonBO=maxtogether!=bo,
            positive_feedback_sensitivity=rows_sim),selected_density=selected_detail,density_check=check,
        target_rule_order_changed=order_mi!=order_js,target_rule_MI_order=order_mi,target_rule_JS_order=order_js,
        target_rule_eligible_candidates=all_target,original_global_gated_candidates=original,
        original_global_candidate_count=len(original),original_MI_winner=old,JS_global_winner=new,
        JS_changes_selected_point=new['point_id']!=old['point_id'],JS_changes_target_rule=new['rule']!=old['rule'],
        all_gated_delta_JS_nonpositive=all(x['delta_JS']<=0 for x in original),
        negative_delta_JS_candidates=sum(x['delta_JS']<0 for x in original))


def self_test():
    assert density_js(.2,.3,.1,-np.inf)==0
    assert density_js(.2,0.,.1,0.)==0
    cases=[(.2,.1,.3,0.),(.2,.1,.3,log(.15)),(-8.,1.,0.,0.),(-30.,1.,0.,log(.3)),(.7,.001,.3,0.),(.2,.1,.3,log(1e-8))]
    for case in cases:density_check(*case)
    x=density_js(.2,.1,.3,log(.4),.06)
    y=density_js(20.,1000.,30.,log(.4),6.)
    assert abs(x-y)<1e-9
    assert density_js(.2,.1,.3,log(.05))<density_js(.2,.1,.3,log(.4))<density_js(.2,.1,.3,0.)
    # sign MI is positive even if the actual expert is a declared no-op.
    sign=float(a.math_fields(np.array([.2]),np.array([0.]),np.array([.1]),np.array([.3]),.5)['sign_information'][0])
    assert sign>0 and density_js(.2,.1,.3,-np.inf)==0
    print(json.dumps(dict(result='PASS',cases=len(cases),no_oracle=True,sign_MI_positive_with_JS_zero=sign)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    rows=[];protocols={}
    for variant in ('v5','v5_seed2_confirm','v6'):
        folder=ROOT/'experiments/runs/knowledge_loop_20261101'/variant
        p=a.read(folder/'protocol.json');protocols[variant]=p['protocol_sha256']
        assert p['protocol_sha256']==a.digest({k:v for k,v in p.items() if k!='protocol_sha256'})
        for item in p['inputs']:
            pool=a.common.public_pool(ROOT,item)
            run=folder/'runs'/f"{item['task']}_full_agent_initial_terra";s=a.read(run/'state.json')
            for d in s['decisions']:
                if d['mode']=='optional_verification':rows.append(review(run,p,pool,s,d['query']))
    assert Counter(r['variant'] for r in rows)=={'v5':7,'v5_seed2_confirm':5,'v6':5}
    summary=[]
    for variant in protocols:
        rr=[r for r in rows if r['variant']==variant]
        summary.append(dict(variant=variant,optional_rounds=len(rr),
            remaining_executable=sum(r['remaining']['executable_after']>0 for r in rr),
            remaining_cap_target_nonBO=sum(r['remaining']['cap_target_only_nonBO_after'] for r in rr),
            remaining_achievable_trust_target_nonBO=sum(r['remaining']['maximum_trust_target_only_nonBO'] for r in rr),
            remaining_achievable_trust_mixture_nonBO=sum(r['remaining']['maximum_trust_current_other_weights_nonBO'] for r in rr),
            JS_changes_actual_choice=sum(r['JS_changes_selected_point'] for r in rr),
            all_gated_delta_JS_nonpositive=sum(r['all_gated_delta_JS_nonpositive'] for r in rr),
            target_rule_order_changes=sum(r['target_rule_order_changed'] for r in rr)))
    old=a.read(OUT/'v4_fixed_routing_counterfactual.json')
    oldsummary={mode:dict(rounds=sum(r['mode']==mode for r in old['rows']),
        nonBO_routed_choices=sum(r['mode']==mode and r['differs_from_bo'] for r in old['rows']),
        original_optional_replaced_by_BO=sum(r['mode']==mode and r['original_mode']=='optional_verification' and r['routed_id']==r['BO_reference_id'] for r in old['rows'])) for mode in ('full','deployment')}
    report=dict(development_only=True,script_sha256=a.filehash(__file__),protocol_sha256=protocols,
        methods=dict(density='Observed Y; actual normalized tempered expert. JS is equal-prior GP/expert model-label information, not the sign MI and not an expected credit gain.',
            quadrature='One-dimensional adaptive Gaussian integrals with fixed tolerances and both GP/expert mass windows; normalization and separate KL-mixture algebra cross-check selected predictors.',
            global_ranking='All live rules, same original gates and candidate set (including V6 proposal restriction); replace only max delta-sign-MI ordering by max delta-JS, then original gap/tie. No JS gate.',
            hypothetical_feedback='Only synthetic positive-fact, positive-score points at fractions .25/.5/.75 of the attainable logscore interval. Also compute the asymptotic trust upper bound; no actual validation outcome is used.',
            remaining='Remove only selected validation point. Hold GP, epsilon, references, expert fields and other current weights fixed. This is credit-channel sensitivity, not the actual next round.'),
        prior_negative_evidence=dict(legacy_scope_review='V3 seed0/seed1, 12 different old pair-model optional rounds; not reused as V5 numbers.',
            hard_gate_counterexamples='Old remaining-scoreable-winner gate would exclude winning seed0 C2/C4/Doyle and seed1 C3 validations.',
            v4_routing=oldsummary),
        LLM_calls=0,GP_fits=0,new_experiments=0,oracle_files_opened=0,current_validation_y_used=False,
        summary=summary,rows=rows,
        limitations=['No AUC or new trajectory is inferred. All three variants are already inspected development evidence.',
            'Remaining-point coverage and positive-feedback sensitivity are diagnostics, never a new mandatory filter.',
            'JS may be zero or smaller at a useful GP experiment; model discrimination and experiment reward are different objectives.',
            'Hypothetical weight effects omit GP conditioning, possible reanchoring, future proposals and next-round budget changes.',
            'A greater JS does not establish a correct expert, positive fact, deployable trust or an improved search trajectory.'])
    path=OUT/'optional_density_actionability_review.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['主动验证：预测密度可区别程度与剩余作用机会（prefix-only）',
        '本报告为已查看轨迹的开发性诊断；17轮（V5 seed0 7、seed2 5、V6 5）。不读取本轮真实反馈、不拟合、不调用LLM、不运行新优化。',
        'JS比较实际tempered观测专家与GP，不是latent-probit或signMI。全局候选保留全部原门限，仅用DeltaJS替换排序；没有新增JS门限。',
        '移除验证点后的诊断保持GP、引用值、epsilon和专家场不变；模拟正反馈仅改变信用，不是下一轮预测或性能反事实。','',
        json.dumps(summary,ensure_ascii=False),'',
        'variant | task/q | executable before->after | cap remains nonBO | attainable trust cap | trust remains nonBO | selected DeltaMI/DeltaJS | global gated count | JS winner | all DeltaJS<=0']
    for r in rows:
        z=r['remaining'];d=r['selected_density'];w=r['JS_global_winner']
        lines.append(f"{r['variant']} | {r['task']}/{r['query']} | {z['executable_before']}->{z['executable_after']} | {z['cap_target_only_nonBO_after']} | {z['maximum_target_trust_supremum']:.8g} | {z['maximum_trust_target_only_nonBO']} | {d['delta_sign_MI']:.8g}/{d['delta_JS']:.8g} | {r['original_global_candidate_count']} | {w['point_id']} {w['rule']} (changed={r['JS_changes_selected_point']}) | {r['all_gated_delta_JS_nonpositive']}")
    lines.extend(['','旧反例必须保留：严格剩余可记分winner门曾排除C2/C4/Doyle seed0及C3 seed1获益验证；不能再次把这里的remaining计数变成硬门。',
        'V4固定routing旧结果（未重算）：'+json.dumps(oldsummary,ensure_ascii=False),
        '完整原gated候选、所有target eligible点的JS/MI及BO基准、正反馈信用敏感性见JSON。不能仅凭排序变化宣称JS将改善AUC。'])
    (OUT/'optional_density_actionability_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(path),summary=summary),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
