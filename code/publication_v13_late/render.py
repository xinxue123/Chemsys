"""Write evidence-derived prose only after all 75 registered runs pass audit."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import json
import math
import shutil
import statistics as st
import sys

from assets import P,ROOT,R,read,put,table,NAMES,LABELS


def wtldesc(x):return f"{x['wins']} higher, {x['ties']} tied and {x['losses']} lower"
def cells(values):
    a=[float(x) for x in values];return f'{st.mean(a):.2f} $\\pm$ {st.stdev(a):.2f}'


def main():
    audit=json.loads((R/'final_audit.json').read_text(encoding='utf-8'))
    if audit['status']!='PASS_COMPLETE' or audit['complete_verified']!=75:raise ValueError('Complete audit required')
    metrics=read(P/'data/final_metrics.csv');rounds=read(P/'data/final_rounds.csv')
    conditional=json.loads((P/'data/conditional_facts.json').read_text(encoding='utf-8'))
    costs=json.loads((P/'data/new_costs.json').read_text(encoding='utf-8'))
    forecast=read(P/'data/late_forecast_calculations.csv')
    f=lambda arm,metric='delta_continuation_SearchAUC':conditional[arm][metric]
    body=f"From matched prefixes, full V13 has {wtldesc(f('bo'))} mean remaining-budget SearchAUC values than BO, and {wtldesc(f('rbo'))} than Reasoning BO across six contexts. "
    body+=f"For final best, the corresponding counts are {f('bo','delta_final_best')['wins']}/{f('bo','delta_final_best')['ties']}/{f('bo','delta_final_best')['losses']} and {f('rbo','delta_final_best')['wins']}/{f('rbo','delta_final_best')['ties']}/{f('rbo','delta_final_best')['losses']} (wins/ties/losses). "
    body+=f"Relative to fixed KF, no refresh and coupled lifecycle, full V13 has higher mean tail SearchAUC in {f('kf_fixed')['wins']}, {f('no_refresh')['wins']} and {f('coupled')['wins']} contexts, respectively (Table~\\ref{{tab:intervention}}). These conditional contrasts retain response variation and do not identify a deterministic module benefit."
    put('generated/fresh_summary.tex',body)
    hold={arm:sorted((x for x in metrics if x['kind']=='holdout' and x['arm']==arm),key=lambda x:int(x['prefix_seed'])) for arm in ('bo','rbo','full')}
    holdfacts={}
    for control in ('bo','rbo'):
        holdfacts[control]={}
        for metric in ('SearchAUC20','final_best'):
            ds=[float(a[metric])-float(b[metric]) for a,b in zip(hold['full'],hold[control])]
            holdfacts[control][metric]=dict(wins=sum(d>1e-8 for d in ds),ties=sum(abs(d)<=1e-8 for d in ds),losses=sum(d< -1e-8 for d in ds))
    put('data/holdout_facts.json',json.dumps(holdfacts,indent=2))
    h=f"V13 has {wtldesc(holdfacts['bo']['SearchAUC20'])} SearchAUC values than BO and {wtldesc(holdfacts['rbo']['SearchAUC20'])} than Reasoning BO (Table~\\ref{{tab:holdout}}). "
    h+='Mean final yields are '+', '.join(f"{st.mean(float(x['final_best']) for x in hold[a]):.2f}\\% for {LABELS[a]}" for a in ('bo','rbo','full'))+'. '
    first=[int(x['first_final_best_query']) for x in hold['full']]
    h+=f"V13 attains its own final best by query {max(first)} in all three runs; this transfer result does not demonstrate a second-half benefit. "
    h+='The small, fully reported task transfer does not establish generalization across reaction families or model backends.'
    put('generated/holdout_summary.tex',h)
    put('generated/holdout_detail.tex',r'\begin{table}[!htbp]\centering\small'+'\n'+r'\caption{Direct Arylation summary, mean $\pm$ sample standard deviation over three starts; all starts are displayed in the main text.}'+'\n'+
        table(['Method','SearchAUC','Final best','First-final-best query'],[[LABELS[a],cells(x['SearchAUC20'] for x in hold[a]),cells(x['final_best'] for x in hold[a]),cells(x['first_final_best_query'] for x in hold[a])] for a in ('bo','rbo','full')])+ '\n'+r'\end{table}\FloatBarrier')
    # Two-realization variability is reported within each prefix, never as a
    # duplicated BO observation or a population confidence interval.
    varied=Counter();detail=[]
    for task in ('c4','perera','hplc'):
        for seed in (0,3):
            row=[NAMES[task],seed]
            for arm in ('rbo','full','kf_fixed','no_refresh','coupled'):
                a=sorted((x for x in metrics if x['kind']=='prefix' and x['task']==task and int(x['prefix_seed'])==seed and x['arm']==arm),key=lambda x:int(x['repetition']))
                first=[next(x['selected_id'] for x in rounds if x['job']==j['job'] and int(x['query'])==11) for j in a]
                av=[float(x['continuation_SearchAUC']) for x in a];bv=[float(x['final_best']) for x in a]
                varied['first_point']+=first[0]!=first[1];varied['AUC']+=abs(av[0]-av[1])>1e-8;varied['final']+=abs(bv[0]-bv[1])>1e-8
                row.append(f'{min(av):.2f}--{max(av):.2f}')
            detail.append(row)
    v=f"Among thirty LLM arm--context pairs, the two realizations differ in their first post-prefix point in {varied['first_point']}, in tail SearchAUC in {varied['AUC']}, and in final best in {varied['final']}. "
    v+='These are small conditional response checks, not estimates of serving-model reproducibility. The following ranges display both realizations without treating the deterministic BO result as replicated. '
    v+=r'\begin{table}[!htbp]\centering\small'+'\n'+r'\caption{Observed tail-SearchAUC ranges across two independent realizations. Equal endpoints mean an observed tie, not absence of response randomness.}'+'\n'+table(['Task','Prefix','RBO','V13','Fixed KF','No refresh','Coupled'],detail)+ '\n'+r'\end{table}\FloatBarrier'
    put('generated/response_variation.tex',v);put('data/response_variation.json',json.dumps(dict(varied),indent=2))
    discussion='The common-prefix interventions remove differences in early measured opportunity, but retain independently sampled language responses. '
    discussion+=f"Against fixed KF, full V13 has {wtldesc(f('kf_fixed'))} mean tail SearchAUC values; against no refresh it has {wtldesc(f('no_refresh'))}. These tests do not establish an incremental SearchAUC benefit from numerical adaptation or future editing. "
    prefix_coverage=next(x for x in read(P/'data/candidate_coverage_summary.csv') if x['cohort']=='new_prefix_full')
    discussion+=f"Structured Note feedback covers only {prefix_coverage['queries_with_selected_forecast']} of {prefix_coverage['queries']} full-V13 continuation choices, limiting the observed opportunity for adaptive updates. "
    discussion+=f"Observed tail-SearchAUC variation within {varied['AUC']} of thirty replicated arm--context pairs cautions against assigning each full-minus-removal difference to a module alone. "
    discussion+='The coupled-lifecycle control also retains a registered editor-schema instruction conflict, repaired before outcome revelation and disclosed in the supplement. '
    discussion+='The transferable claim is an auditable separation of evidence types; broader optimization superiority requires more independent task families and more response realizations per fixed history.'
    put('generated/conditional_discussion.tex',discussion)
    newfull=[x for x in forecast if x['kind']!='historical' and x['arm']=='full']
    fulljobs=[x for x in metrics if x['arm']=='full'];coverage={(x['job'],x['query']) for x in newfull}
    mechanisms=[]
    for arm in ('full','kf_fixed','no_refresh','coupled'):
        fs=[x for x in forecast if x['kind']!='historical' and x['arm']==arm]
        js=[x for x in metrics if x['arm']==arm]
        ups=sum(float(x['actual_reliability'])>float(x['reliability_before'])+1e-12 for x in fs)
        downs=sum(float(x['actual_reliability'])<float(x['reliability_before'])-1e-12 for x in fs)
        suppress=sum(x['permission_after']=='suppressed' and x['permission_before']!='suppressed' for x in fs)
        allows=sum(x['permission_after']=='allowed' and x['permission_before']!='allowed' for x in fs)
        mechanisms.append([LABELS[arm],len(js),sum(int(x['new_queries']) for x in js),len(fs),len({(x['job'],x['query']) for x in fs}),ups,downs,suppress,allows,sum(int(x['editor_calls']) for x in js)])
    value=f"Full V13 contributes {len(newfull)} new scoped Note forecasts across {len(coverage)} of {sum(int(x['new_queries']) for x in fulljobs)} new choices in its fifteen continuation/transfer trajectories. "
    value+='These counts concern the additive full arm only; intervention counts below are separate. Numerical activity and permission transitions are not performance-effect estimates. All raw forecasts, uses, claims, scopes, current-incumbent references and actual versus unapplied scalar updates are exported. '
    value+=r'\begin{table}[!htbp]\centering\small'+'\n'+r'\caption{New-stage mechanism activity. Multiple forecasts can belong to one measured query. Fixed KF logs numerical proposals without applying them. Positive support and suppression count newly entered permissions.}'+'\n'+r'\resizebox{\linewidth}{!}{'+table(['Arm','Runs','Queries','Forecasts','Covered','Reliab. up','Down','Suppress','Allow','Edits'],mechanisms)+'}\n'+r'\end{table}\FloatBarrier'
    put('generated/new_mechanism_summary.tex',value)
    coverage=read(P/'data/candidate_coverage_summary.csv')
    cv=next(x for x in coverage if x['cohort']=='cold')
    value=f"In historical cold search, live Notes exist on {cv['queries_with_any_live_note']} of {cv['queries']} decisions, intersect {cv['queries_with_any_menu_coverage']} menus, and cover a strict subset of candidates on {cv['queries_with_selective_coverage']} menus. "
    value+=f"Among {cv['changed_reliability_events_with_next_query']} numerical-reliability changes with a next decision, {cv['changed_reliability_with_next_menu_coverage']} retain the same live version and intersect the next menu. "
    value+='These are opportunity counts, not effects on search. '
    cohort_label=dict(cold='Cold 0--3',warm_primary='Warm 1--3',warm_development='Warm 0',new_prefix_full='Prefix: V13',
        new_prefix_kf_fixed='Prefix: fixed KF',new_prefix_no_refresh='Prefix: no refresh',new_prefix_coupled='Prefix: coupled',new_holdout_full='Direct Arylation')
    value+=r'\begin{table}[!htbp]\centering\small'+'\n'+r'\caption{Candidate-level scope coverage. Selective means one to four of five menu points; full-menu means all five. Counts can overlap across multiple Notes. New-stage rows exclude inherited prefix queries.}'+'\n'+r'\resizebox{\linewidth}{!}{'+table(['Cohort','Queries','Live Note','Menu overlap','Selective','Full menu','Selected forecast'],
        [[cohort_label[x['cohort']]]+[x[k] for k in ('queries','queries_with_any_live_note','queries_with_any_menu_coverage','queries_with_selective_coverage','queries_with_all_five_note','queries_with_selected_forecast')] for x in coverage])+'}\n'+r'\end{table}\FloatBarrier'
    cg=json.loads((P/'data/candidate_coverage_receipt.json').read_text(encoding='utf-8'))
    value+=f"\n\nAt the first post-prefix decision, {cg['repeated_identical_first_query_packet_groups']} repeated groups have identical public packet hashes across full/fixed-KF/no-refresh realizations; {cg['identical_packet_groups_with_different_selected_point']} group{'s' if cg['identical_packet_groups_with_different_selected_point']!=1 else ''} nevertheless select{'s' if cg['identical_packet_groups_with_different_selected_point']==1 else ''} different points. "
    value+='Coupled-lifecycle packets are excluded because their instructions explicitly differ. '
    if cg['identical_packet_groups_with_different_selected_point']:
        value+='These same-packet discrepancies precede any possible numerical post-fork KF feedback and provide direct evidence of response variation.'
    else:value+='This first-decision agreement alone does not establish reproducibility over complete response trajectories.'
    put('generated/coverage_summary.tex',value)
    previous=[x for x in read(ROOT/'paper/v12_subagent/data/paired_comparisons.csv') if x['comparison'] in ('KF_removal','refresh_removal')]
    removal=[]
    for task,label in (('Reizman_Suzuki_Case_2','Suzuki 2'),('Reizman_Suzuki_Case_4','Suzuki 4'),('HPLC','HPLC')):
        for seed in (0,3):
            row=[label,seed]
            for comparison in ('KF_removal','refresh_removal'):
                x=next(x for x in previous if x['task']==task and int(x['seed'])==seed and x['comparison']==comparison)
                row.append(f"{float(x['AUC_delta']):+.2f} / {float(x['best_delta']):+.2f}")
            removal.append(row)
    put('generated/v12_removals.tex',table(['Task','Initialization','Full minus no KF','Full minus no refresh'],removal))
    with (P/'data/v12_removal_pairs.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(previous[0]));writer.writeheader();writer.writerows(previous)
    n=costs['request_counts'];value=f"All 75 trajectories and 840 newly selected outcomes complete the registered matrix. The archive retains {n['raw_requests']} raw LLM request attempts: {n.get('choose_raw_requests',0)} selection and {n.get('edit_raw_requests',0)} editor attempts. "
    value+=f"Of these, {n.get('choose_accepted_requests',0)} selection requests and {n.get('edit_accepted_requests',0)} editor requests enter paid-query commitments; {n.get('original_schema_rejections',0)} original schema rejections remain recorded. "
    value+=f"There are {n.get('accepted_deterministic_empty_sidecar_projections',0)} accepted bare-RBO responses with deterministic empty-side-field projection, including {n.get('recovered_empty_sidecar_rejections',0)} recovered original rejections. "
    value+=f"The coupled memory-editor instruction conflict produces {n.get('coupled_editor_envelope_rejections',0)} recorded envelope rejections, handled by the registered pre-outcome schema-only repair. "
    value+=f"The audit records {costs['earlier_core_choice_differs_from_native_accepted']} earlier core-valid responses whose selected point differs from a later native-accepted response; those native decisions are preserved, not retrospectively replaced. "
    value+=f"All 831 fitted new decision prefixes bind to {costs['unique_bound_checkpoints']} distinct available native GP checkpoints; the other nine first decisions have no data or GP. "
    value+='Experimental query budgets match, while LLM call budgets differ. Token and monetary accounting are unavailable through this backend. Local persistent CPU workers and locked exact-prefix caches are used; no operator compilation occurs inside the query loop. The configured cluster identity is unavailable, and no cluster or local-GPU inference is asserted.'
    put('generated/execution_costs.tex',value)
    abstract=("Language models can guide experimental optimization, but predictive correctness does not by itself justify a recommendation. "
        "We study V13, a frozen controller that separates GP-relative predictive reliability from permission to promote a scientific hypothesis. "
        "Prospective forecasts update a Kalman-filtered skill estimate; directional experimental evidence governs suppression and review, while the GP remains unchanged. "
        "Across eight historical tasks, V13 improves cold-start SearchAUC in 22 of 32 comparisons with BO, but warm-start results are mixed. "
        "Post-hoc analysis identifies substantial later gains on Perera and HPLC, without establishing their cause. "
        f"Seventy-five registered continuation and transfer trajectories add controlled tests: full V13 improves mean remaining-budget SearchAUC over BO in {f('bo')['wins']} of six common-prefix contexts. "
        "Transfer results remain mixed and do not demonstrate a second-half feedback benefit. "
        "Independent responses and module contrasts expose conditional, rather than universal, gains. "
        "The evidence distinguishes an auditable knowledge-feedback mechanism from a generally superior optimizer.")
    if len(abstract.split())>150:raise ValueError('Abstract exceeds 150 words')
    put('generated/abstract.tex',abstract)
    old=(ROOT/'paper/v13/supplement.tex').read_text(encoding='utf-8')
    start=old.index('\\section*{Evidence scope and study boundaries}')
    body=old[start:old.rindex('\\end{document}')]
    replacements={
        'No new optimization, GP refit, oracle query or LLM call was performed while preparing the manuscript.':'The original release was prepared retrospectively without new optimization. The additive registration below separately defines the new continuation and transfer experiments.',
        'No matched no-KF or no-refresh V13 trajectory has been run in these cohorts.':'These original cohorts contain no matched V13 module removals. The new common-prefix intervention study below is separate and uses its own precisely defined fixed-KF and no-refresh interventions.',
        'This publication task performs no new fitting or optimization.':'The retrospective publication export performs no new fitting; new optimization is confined to the additive registered matrix below.'}
    for source,destination in replacements.items():
        if body.count(source)!=1:raise ValueError('Historical supplement text drift: '+source)
        body=body.replace(source,destination)
    header=r'''\documentclass[10pt]{article}
\input{preamble}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}
\title{Supplementary Information\\[.5em]\large Separating predictive reliability from recommendation value in language-guided experimental optimization}
\author{}\date{}
\begin{document}\maketitle\vspace{-3em}
'''
    notice='The original eight-task V13 results are reproduced first without replacing any trajectory. Additional phase accounting, registered common-prefix interventions and the new Direct Arylation task follow. Earlier development catalogue entries retain their original evidence labels and are not pooled as independent confirmation.\n\n'
    put('supplement.tex',header+notice+body+'\n\\FloatBarrier\n\\input{late_supplement}\n'+r'\bibliographystyle{unsrtnat}\bibliography{references,late_references}'+'\n'+r'\end{document}')
    shutil.copy2(ROOT/'paper/v13/verify_results.py',P/'verify_historical.py')
    put('audit/prose_generation.json',json.dumps(dict(status='PASS',abstract_words=len(abstract.split()),
        source_audit_sha256=__import__('hashlib').sha256((R/'final_audit.json').read_bytes()).hexdigest(),
        historical_body_replacements=list(replacements),new_query_or_LLM_calls=0),indent=2))
    print(json.dumps(dict(status='PROSE_READY',abstract_words=len(abstract.split()),holdout=holdfacts,conditional=conditional)))


if __name__=='__main__':main()
