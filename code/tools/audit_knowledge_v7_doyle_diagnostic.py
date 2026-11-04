"""Finish V7 Doyle's listed audits while retaining every EI precision failure.

Only the named adaptive-EI precision assertion is captured; all other errors
remain fatal. This does not alter production or the original strict report.
The output can never claim an unqualified technical PASS.
"""
from datetime import datetime,timezone
import json
from pathlib import Path

import audit_knowledge_actionable_loop as v7


def main():
    a=v7.v5;root=v7.ROOT
    folder=root/'experiments/runs/knowledge_loop_20261101/v7'
    output=root/'experiments/reports/knowledge_loop_20261101/v7_Doyle_diagnostic_completion.json'
    original_path=output.with_name('v7_joint_audit.json')
    protocol=a.read(folder/'protocol.json')
    a.require(protocol['protocol_sha256']==a.digest({k:v for k,v in protocol.items() if k!='protocol_sha256'}),'protocol self hash')
    drift=[name for name,sha in protocol['source_sha256'].items() if a.filehash(root/name)!=sha]
    a.require(not drift,'frozen production source drift')
    manifest=next(x for x in protocol['inputs'] if x['task']=='Doyle_Buchwald_Hartwig')
    pool=dict(a.common.public_pool(root,manifest),task=manifest['task'])
    path=folder/'runs/Doyle_Buchwald_Hartwig_full_agent_initial_terra/state.json'
    recorded=[];current={}
    native_adaptive=a.adaptive_ei_check
    native_selection=v7.selection_audit

    def selection_context(pool,prefix,c,arrays,*args,**kwargs):
        current.clear()
        current.update(query=c['query'],selected_id=c['selected_id'],BO_reference_id=c['BO_reference_id'],
                       commit_sha256=a.digest(c))
        return native_selection(pool,prefix,c,arrays,*args,**kwargs)

    def retain_precision_failure(mu,vf,vy,reference,best,expected_log_ratio,tau=.06):
        try:
            return native_adaptive(mu,vf,vy,reference,best,expected_log_ratio,tau=tau)
        except AssertionError as error:
            message=str(error)
            marker='GH64/independent F-integral log-ratio error '
            if not message.startswith(marker):
                raise
            measured=float(message[len(marker):])
            a.require(measured>2e-5,'unexpected precision assertion value')
            recorded.append(dict(current,check='adaptive_EI_GH64_precision',result='FAIL',
                original_message=message,tolerance=2e-5,absolute_log_ratio_error=measured,
                inputs=dict(mu=mu,vF=vf,vY=vy,reference_y=reference,best=best,
                            GH64_log_ratio=expected_log_ratio,tau=tau)))
            return measured

    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
        status='DIAGNOSTIC_NOT_COMPLETE',source_sha256=a.filehash(__file__),
        original_strict_audit_path=str(original_path.relative_to(root)),
        original_strict_audit_sha256=a.filehash(original_path),
        protocol_sha256=protocol['protocol_sha256'],source_drift=drift,
        audit_helper_sha256={str(Path(m.__file__).relative_to(root)):a.filehash(m.__file__)
                             for m in (v7,a,a.v4,a.common)},
        captured_numerical_failures=recorded,oracle_files_opened=0,GP_fits=0,LLM_calls=0,
        saved_GP_and_MC512_recomputed=True,original_audit_overwritten=False,
        limitation='A retained numerical failure is not a PASS; this diagnostic only completes the other listed checks.')
    fatal=None
    try:
        a.adaptive_ei_check=retain_precision_failure
        v7.selection_audit=selection_context
        result=v7.audit_run(folder,path,protocol,pool,manifest,joint=True)
        a.require(bool(recorded),'expected strict numerical failure was not reproduced')
        # Never serialize the audit_run's generic success label after catching
        # an assertion. Every persisted completion label includes FAIL.
        result['technical_checks']='NUMERICAL_FAIL_OTHER_LISTED_CHECKS_COMPLETE'
        report.update(status='NUMERICAL_FAIL_OTHER_LISTED_CHECKS_COMPLETE',run=result,
            Doyle_queries_other_checks_completed=result['queries'],Doyle_forecasts_checked=result['forecasts'],
            complete_saved_GP_reconstructions=16)
        original=a.read(original_path)
        prior_passes=[r for r in original['runs'] if r['technical_checks']=='PASS_LISTED_CHECKS']
        report['combined_scope']=dict(previous_strict_pass_runs=len(prior_passes),
            previous_strict_pass_queries=sum(r['queries'] for r in prior_passes),
            diagnostic_numerical_fail_runs=1,diagnostic_complete_queries=result['queries'],
            unique_queries_with_other_listed_checks_complete=sum(r['queries'] for r in prior_passes)+result['queries'],
            all_queries_strict_pass=False,strict_report_result_unchanged=original['technical_result'])
    except Exception as error:
        fatal=error
        report.update(status='DIAGNOSTIC_ABORTED_OTHER_FAILURE',
                      fatal_error_type=type(error).__name__,fatal_error=str(error))
    finally:
        a.adaptive_ei_check=native_adaptive
        v7.selection_audit=native_selection
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V7 Doyle全16轮继续审计：数值FAIL保留',report['status'],
           '仅捕获原adaptive_EI精度超差断言，所有其他异常原样失败；未放宽2e-5门限，未改生产或原strict报告。',
           '数值失败：'+json.dumps(recorded,ensure_ascii=False)]
    if 'combined_scope' in report:
        lines.append('完成范围：'+json.dumps(report['combined_scope'],ensure_ascii=False))
        lines.append('原strict审计只有15条/240轮PASS；本诊断补足Doyle16轮GP/结构/选点/提示/反馈等其余检查，合计256轮覆盖，不能写成256轮strict PASS。')
    if fatal is not None:lines.append('另有失败：'+str(fatal))
    output.with_suffix('.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','captured_numerical_failures','combined_scope') if k in report},ensure_ascii=False,indent=2))
    if fatal is not None:raise fatal


if __name__=='__main__':main()
