"""Read-only independent review of the V7 Doyle GH64 audit failure."""
import json
from math import exp, log, sqrt, pi
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, logsumexp, ndtri_exp, roots_hermitenorm

import audit_knowledge_transfer_loop as a

ROOT=a.ROOT
OUT=ROOT/'experiments/reports/knowledge_loop_20261101'


def strict_f_integral(mu,vf,vy,reference,best,tau=.06):
    sd=sqrt(vf); cutoff=(best-mu)/sd
    log_base=float(a.v4.normal_log_ei(np.asarray(mu),np.asarray(vf),np.asarray(best)))
    log_p=float(log_ndtr((mu-reference)/sqrt(vy+tau*tau)))
    scale=1/max(1.,cutoff)
    def integrand(u,conditional):
        if u==0:return 0.
        z=cutoff+scale*u
        value=log(sd)+log(scale*u)-z*z/2-log(2*pi)/2-log_base+log(scale)
        if conditional:
            value+=float(log_ndtr((best+sd*scale*u-reference)/sqrt(vy-vf+tau*tau)))-log_p
        return exp(value)
    baseline,eb=quad(lambda u:integrand(u,False),0,np.inf,epsabs=2e-13,epsrel=2e-13,limit=500)
    corrected,ec=quad(lambda u:integrand(u,True),0,np.inf,epsabs=2e-13,epsrel=2e-13,limit=500)
    return dict(base_EI_normalization=baseline,normalization_error_estimate=eb,
                conditional_EI_ratio=corrected,log_ratio=log(corrected),ratio_error_estimate=ec)


def high_order_z_integral(mu,vf,vy,reference,best,n,tau=.06):
    nodes,weights=roots_hermitenorm(n)
    weights=weights/sqrt(2*pi)
    log_p=float(log_ndtr((mu-reference)/sqrt(vy+tau*tau)))
    qz=-ndtri_exp(log_ndtr(-nodes)+log_p)
    means=mu+vf/sqrt(vy+tau*tau)*qz
    variance=vf*(1-vf/(vy+tau*tau))
    conditional=logsumexp(a.v4.normal_log_ei(means,np.full(n,variance),np.full(n,best))+np.log(weights))
    base=float(a.v4.normal_log_ei(np.asarray(mu),np.asarray(vf),np.asarray(best)))
    raw=float(conditional-base)
    return dict(nodes=n,raw_log_ratio=raw,bounded_log_ratio=min(max(0.,raw),-log_p))


def main():
    folder=ROOT/'experiments/runs/knowledge_loop_20261101/v7'
    protocol=a.read(folder/'protocol.json')
    item=next(x for x in protocol['inputs'] if x['task']=='Doyle_Buchwald_Hartwig')
    pool=a.common.public_pool(ROOT,item)
    run=folder/'runs/Doyle_Buchwald_Hartwig_full_agent_initial_terra'
    state=a.read(run/'state.json')
    failures=[];checked=0
    for t in range(1,17):
        cp=run/'rounds'/f'{t:02d}_commit.json';c=a.read(cp);prefix=state['history'][:t+3]
        assert a.digest(prefix)==c['history_sha256']
        with np.load(cp.with_name(f'{t:02d}_GP.npz'),allow_pickle=False) as saved:
            arrays={k:saved[k].copy() for k in saved.files}
        rows,calc=a.calculations(pool,c['hypotheses_at_commit'],prefix,arrays)
        for key,data in calc.items():
            for j in sorted({pool['ix'][c['selected_id']],pool['ix'][c['BO_reference_id']]}):
                position=data['positions'].get(j)
                if position is None:continue
                checked+=1
                mu,vf,vy=[float(arrays[k][j]**(1 if k=='mu_y' else 2)) for k in ('mu_y','sd_y','sd_predictive_y')]
                ref=data['probes'][j]['reference_y'];best=max(x['y'] for x in prefix)
                expected=float(data['values']['pure_log_ei_ratio'][position])
                try:
                    a.adaptive_ei_check(mu,vf,vy,ref,best,expected)
                    continue
                except AssertionError as error:
                    failure=str(error)
                strict=strict_f_integral(mu,vf,vy,ref,best)
                gh=[high_order_z_integral(mu,vf,vy,ref,best,n) for n in (32,64,128,256)]
                log_p=float(log_ndtr((mu-ref)/sqrt(vy+.06**2)))
                # For every contributing F>best, Phi((F-ref)/residual_sd)
                # lies in [Phi((best-ref)/residual_sd),1]. This yields an
                # analytical log-ratio bracket, independent of quadrature.
                residual_sd=sqrt(vy-vf+.06**2)
                log_min_event=float(log_ndtr((best-ref)/residual_sd))
                analytic=dict(upper_log_ratio=-log_p,lower_log_ratio=-log_p+log_min_event,
                    bracket_width=-log_min_event,event_at_improvement_boundary_logprob=log_min_event,
                    explanation='EI-weighted conditional event probability lies between boundary probability and1; divide by P(event).')
                true_log=strict['log_ratio']
                old_cap=float(np.logaddexp(log(.5),log(.5)+expected))
                new_cap=float(np.logaddexp(log(.5),log(.5)+true_log))
                old_gap=float(arrays['base_logei'][pool['ix'][c['BO_reference_id']]]-arrays['base_logei'][j])
                pred=next(s['prediction'] for s in c['shadows'] if (s['id'],s['version'])==key)
                receipt=next(e for e in state['decisions'][t-1]['evidence'] if (e['hypothesis_id'],e['version'])==key)
                failures.append(dict(query=t,point_id=pool['ids'][j],rule=list(key),mode=c['mode'],
                    commit_sha256=a.filehash(cp),GP_sha256=a.filehash(cp.with_name(f'{t:02d}_GP.npz')),
                    inputs=dict(mu=mu,vF=vf,vY=vy,reference_y=ref,best=best,tau=.06),
                    original_audit_failure=failure,original_tolerance=2e-5,GH64_log_ratio=expected,
                    strict_F_integral=strict,analytic_log_ratio_bracket=analytic,high_order_Z_GH=gh,
                    absolute_log_ratio_error=abs(true_log-expected),
                    relative_ratio_underestimate=1-exp(expected-true_log),
                    opportunity_gap=old_gap,old_cap_bonus=old_cap,new_cap_bonus=new_cap,
                    old_cap_margin=old_cap-old_gap,new_cap_margin=new_cap-old_gap,
                    original_tempering=pred['tempering'],corrected_tempering=1. if true_log<=log(4) else 3/np.expm1(true_log),
                    optional_receipt=c['verification'],coverage=next(r for r in rows if a.hkey(r)==key),
                    feedback_receipt=receipt,
                    impact='Both ratios are positive and belowln4: tempering remains exactly1, so frozen observation density/score/credit are identical. One executable probe, zero actual weight, gap0; cap/effect gates and signMI ranking are unchanged.',
                    subsequent_H002_coverage=[dict(query=q,
                        rows=[r for r in a.read(run/'rounds'/f'{q:02d}_commit.json')['coverage'] if r['id']==key[0]]) for q in range(t+1,17)]))
    assert len(failures)==1
    f=failures[0]
    report=dict(diagnostic='V7 Doyle original GH64 audit failure, no tolerance relaxation',
        source_sha256=a.filehash(__file__),protocol_sha256=protocol['protocol_sha256'],
        checks=dict(selected_or_BO_expert_points=checked,failures=len(failures)),failures=failures,
        conclusion='GH64 conditional-EI quadrature underestimates the log ratio; the independent audit integral is corroborated by higher-order GH and an analytical bracket.',
        original_audit_status='FAIL remains; this explanatory review does not overwrite or promote the original audit.',
        trajectory_impact='For the isolated diagnosed point, corrected EI precision cannot change gates, selected point, forecast density or feedback; original known V7 trajectory/performance rejection remains. This is not a new optimization trajectory.',
        no_new_oracle=True,new_GP_fits=0,new_LLM_calls=0,production_edits=0)
    (OUT/'v7_Doyle_EI_precision_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V7 Doyle GH64独立精度失败：只读定位',f"唯一超差q{f['query']} {f['point_id']} {f['rule']}，{f['mode']}。",
        f"GH64 logR={f['GH64_log_ratio']:.16g}; 严格F积分={f['strict_F_integral']['log_ratio']:.16g}; 差={f['absolute_log_ratio_error']:.12g}，原门2e-5未修改。",
        f"解析上界logR={f['analytic_log_ratio_bracket']['upper_log_ratio']:.16g}; 上下界宽度<={f['analytic_log_ratio_bracket']['bracket_width']:.12g}。",
        '高阶Z-GH：'+json.dumps(f['high_order_Z_GH'],ensure_ascii=False),
        f"EI比相对低估{f['relative_ratio_underestimate']:.12g}; cap加成旧/新={f['old_cap_bonus']:.12g}/{f['new_cap_bonus']:.12g}; gap={f['opportunity_gap']:.12g}。",
        '两种精度下tempering都精确为1；实际信用权重0，只有1个可执行probe。signMI排序不依赖这个EI近似，cap/effect门距边界很远。',
        '因此该点的冻结预测密度、前瞻分数和反馈权重不变，原选点也不变。它仍是超过预声明精度门的真实数值问题，不得将原FAIL改成PASS。',
        '此审查不改生产、不放宽容差、不读取未查询目标，不声称模拟了新的优化轨迹。V7因0次加权改点停止的既有结论不受本点影响。']
    (OUT/'v7_Doyle_EI_precision_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(query=f['query'],point=f['point_id'],GH64=f['GH64_log_ratio'],strict=true_log,
        analytic_width=f['analytic_log_ratio_bracket']['bracket_width'],cap_margin=f['new_cap_margin'],GH=f['high_order_Z_GH']),indent=2))


if __name__=='__main__':main()
