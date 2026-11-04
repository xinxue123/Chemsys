"""Source-backed prose fragments; pending experiments never become zero results."""
from pathlib import Path
from hashlib import sha256
import csv,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper/v12_subagent';OLD=ROOT/'paper/v12'
def wr(name,value):(P/'generated'/name).write_text(value+'\n',encoding='utf-8')
def counts(rows,key):
    a=np.array([r[key] for r in rows]);return int((a>1e-9).sum()),int((np.abs(a)<=1e-9).sum()),int((a<-1e-9).sum())
def counted(n,singular,plural=None):
    return f'{n} '+(singular if n==1 else (plural or singular+'s'))
def outcome_counts(w,t,l,metric=''):
    prefix=metric+' ' if metric else ''
    return counted(w,prefix+'win')+', '+counted(t,'tie')+' and '+counted(l,'loss','losses')
def first_difference(a,b):
    return next((q for q,(x,y) in enumerate(zip(a['selected_ids'],b['selected_ids']),1) if x!=y),None)
def main():
    e=json.loads((P/'data/evidence.json').read_text(encoding='utf-8'));n=e['completed_trajectories'];pairs=e['paired'];rows=e['rows'];complete=n==68
    for d in ('generated','figures','data'):(P/d).mkdir(parents=True,exist_ok=True)
    for name in ('references.bib',):shutil.copy2(OLD/name,P/name)
    for name in ('workflow.pdf','workflow.png','llm_repetitions.pdf','llm_repetitions.png','development_heatmap.pdf','development_heatmap.png'):
        shutil.copy2(OLD/'figures'/name,P/'figures'/name)
    for name in ('metrics.csv','paired_comparisons.csv','summary.csv','evidence.json'):
        shutil.copy2(OLD/'data'/name,P/'data'/('legacy_'+name))
    shutil.copy2(OLD/'data/historical_runs.csv',P/'data/historical_runs.csv')
    for name in ('seed0_all.tex','repeats.tex','seeds.tex','mechanisms.tex','kf_example.tex','historical_groups.tex'):
        shutil.copy2(OLD/'generated'/name,P/'generated'/('legacy_'+name))
    state=('All 68 registered subagent trajectories are complete.' if complete else f'{n} of 68 registered subagent trajectories are complete; uncompleted cells remain explicitly missing.')
    audit_text=(f"{e['audit_completed']} completed trajectories have passed the recorded numerical and provenance checks." if str(e['audit_status']).startswith('PASS') else 'Complete-trajectory audit is pending or requires resolution.')
    wr('evidence_status.tex',r'{\small\textbf{Evidence status.} '+state+' Original API and new subagent executions are separate cohorts. '+audit_text+' This is a research draft, not a submission-ready claim.}')
    mainpairs=[r for r in pairs if r['comparison']=='V12_vs_BO' and r['seed'] in (1,2,3)]
    rbo=[r for r in pairs if r['comparison']=='V12_vs_RBO' and r['seed'] in (1,2,3)]
    if complete:
        w,t,l=counts(mainpairs,'AUC_delta')
        abstract=(f'Language models can suggest experimental directions, but persuasive explanations do not establish added value beyond a statistical optimizer. We study prospective evidence memory for language-guided Bayesian optimization while leaving the Gaussian process unchanged. Each applicable knowledge version forecasts whether the selected experiment will beat the current incumbent. Relative Brier-score feedback updates Kalman reliability states, while adverse facts and event-triggered revision regulate subsequent reasoning. Across eight tasks and three matched initializations, the frozen policy wins {w}, ties {t} and loses {l} search-efficiency comparisons with Bayesian optimization. Separate removal trajectories test calibration control and event refresh. Original API response repetitions remain distinct from the request-isolated subagent cohort. The evidence distinguishes an executable knowledge lifecycle from its task-dependent optimization value; predictive calibration alone neither validates a productive chemical direction nor establishes a causal module benefit.')
    else:
        abstract=('Language models can suggest experimental directions, but persuasive explanations do not establish added value beyond a statistical optimizer. We study prospective evidence memory for language-guided Bayesian optimization while leaving the Gaussian process unchanged. Each applicable knowledge version forecasts whether the selected experiment will beat the current incumbent. Relative Brier-score feedback updates Kalman reliability states, while adverse facts and event-triggered revision regulate subsequent reasoning. A registered eight-task programme compares the frozen policy with Bayesian optimization and a matched Reasoning BO implementation, and separately removes calibration control and event refresh. The original API evaluations and the request-isolated subagent cohort remain distinct. This working draft retains incomplete cells rather than imputing outcomes. The methodological distinction is between forecast reliability, productive directional support and optimization value; completion of the registered experiments is required before drawing broader empirical conclusions.')
    if len(abstract.split())>150:raise ValueError('Abstract exceeds150words')
    wr('abstract.tex',abstract)
    if len(mainpairs)==24 and len(rbo)==24:
        w,t,l=counts(mainpairs,'AUC_delta');w2,t2,l2=counts(rbo,'AUC_delta')
        text=f'Across the 24 matched initialization comparisons, V12 has {outcome_counts(w,t,l,"SearchAUC")} against BO. Against matched Reasoning BO, the counts are {w2}, {t2} and {l2}, respectively. These are counts of the registered task--initialization pairs, not independent draws from an external task population. '
        summaries=[r for r in e['summary'] if r['comparison']=='V12_vs_BO' and r['complete_three_seed_panel']]
        positive_tasks=sum(r['ratio_of_means_percent']>1e-9 for r in summaries)
        text+=f'Mean SearchAUC is higher on {positive_tasks} of eight tasks. '
        text+='The relative differences in the ratio of mean SearchAUC are '+', '.join(r['label']+f" {r['ratio_of_means_percent']:+.2f}"+r'\%' for r in summaries)+'.'
        case3_rbo=[r for r in rbo if r['task']=='Reizman_Suzuki_Case_3']
        if counts(case3_rbo,'AUC_delta')!=(0,0,3):raise ValueError('Case3 incremental comparison changed')
        text+=' The incremental advantage over Reasoning BO is not uniform: V12 loses all three Suzuki 3 SearchAUC comparisons.'
        wr('search_results.tex',text)
        w,t,l=counts(mainpairs,'best_delta');w2,t2,l2=counts(rbo,'best_delta')
        wr('final_results.tex',f'Against BO, V12 has {outcome_counts(w,t,l,"final-best")} in the 24 pairs. Against Reasoning BO, the corresponding counts are {w2}, {t2} and {l2}. Full raw-unit values and first-best query positions are retained in the Supplementary Information.')
    else:
        wr('search_results.tex',f'The registered primary analysis contains 24 V12--BO and 24 V12--Reasoning BO pairs at initializations 1--3. At this cutoff, {len(mainpairs)} V12--BO and {len(rbo)} V12--Reasoning BO pairs are complete. The full paired result is pending; no denominator is silently reduced to completed favourable cells.')
        wr('final_results.tex','Final-best comparisons will be evaluated over the complete registered pairs. Pending cells are not treated as ties or losses.')
    rbo_controls=[r for r in pairs if r['comparison']=='RBO_vs_BO' and r['seed'] in (1,2,3)]
    if len(rbo_controls)==24:
        rw,rt,rl=counts(rbo_controls,'AUC_delta')
        bw,bt,bl=counts(rbo_controls,'best_delta')
        task_panels=[r for r in e['summary'] if r['comparison']=='RBO_vs_BO' and r['complete_three_seed_panel']]
        if len(task_panels)!=8:raise ValueError('Complete Reasoning BO comparison needs eight task panels')
        positive_tasks=sum(r['ratio_of_means_percent']>1e-9 for r in task_panels)
        largest=max(task_panels,key=lambda r:r['ratio_of_means_percent'])
        context=f'Across its own 24 matched comparisons with BO, Reasoning BO has {outcome_counts(rw,rt,rl,"SearchAUC")}. Final best gives {outcome_counts(bw,bt,bl)}. Its mean SearchAUC exceeds BO on {positive_tasks} of eight tasks; the largest ratio-of-means increase is '+f"{largest['ratio_of_means_percent']:.2f}"+r'\% on '+largest['label']+'. '
        largest_pairs=[r for r in rbo_controls if r['task']==largest['task']]
        if len(largest_pairs)!=3:raise ValueError('Largest Reasoning BO task needs all three paired initializations')
        if counts(largest_pairs,'best_delta')==(0,3,0):
            context+='All three final-best comparisons on this task are ties, so that advantage concerns search efficiency. '
        context+='These matched local adaptations also show task-dependent gains.'
        wr('search_results.tex',(P/'generated/search_results.tex').read_text(encoding='utf-8').strip()+'\n\n'+context)
    removals=[r for r in pairs if r['comparison'] in ('KF_removal','refresh_removal')]
    if len(removals)==12:
        statements=[]
        for kind,label in [('KF_removal','no-KF'),('refresh_removal','no-refresh')]:
            rr=[r for r in removals if r['comparison']==kind];w,t,l=counts(rr,'AUC_delta')
            statements.append(f'Full V12 has {outcome_counts(w,t,l,"SearchAUC")} against {label} across the six registered focus cells')
        wr('removal_results.tex','; '.join(statements)+'. Every individual contrast is reported, including cases in which the removal is better.')
    else:wr('removal_results.tex',f'{len(removals)} of the twelve registered removal contrasts currently have both trajectories complete. The module comparison remains incomplete.')
    full=[r for r in rows if r['method']=='V12']
    coverage_path=P/'data/mechanism_coverage.json';coverage=json.loads(coverage_path.read_text(encoding='utf-8'))
    mechanism_receipt=json.loads((ROOT/'experiments/reports/paper_v12_subagent/mechanism_generation.json').read_text(encoding='utf-8'))
    if mechanism_receipt['evidence_sha256']!=sha256((P/'data/evidence.json').read_bytes()).hexdigest():raise ValueError('Refresh round/forecast exports before generating prose')
    if coverage['completed_trajectories']!=len(full):raise ValueError('Mechanism coverage denominator')
    totals=coverage['totals']
    mechanism=f'The {len(full)} completed subagent V12 trajectories contain '+', '.join(counted(sum(r[key] for r in full),singular,plural) for key,singular,plural in [('forecasts','prospective forecast update',None),('weight_decreases','weight decrease',None),('pauses','automatic pause',None),('retractions','automatic retraction',None),('resumes','automatic recovery','automatic recoveries'),('edits','memory-edit call',None)])+'. '
    mechanism+=f"At least one applicable forecast is updated at {totals['queries_with_forecast_update']} of {totals['queries']} query positions. Among all forecast events, {totals['positive_improvement_forecasts']} combine positive relative gain with an improved incumbent; {totals['positive_gain_nonimprovement_forecasts']} have positive gain on a non-improving outcome. There are {totals['accepted_support_citations']} accepted support citations. "
    mechanism+='These counts establish which branches executed; several forecasts can share one measured outcome and are not independent experimental replicates. The supplied per-query and per-forecast tables retain the states and selected identities behind these counts.'
    supported=[r for r in coverage['trajectories'] if r['accepted_support_citations']]
    if supported:
        supported_pairs=[]
        for trajectory in supported:
            matches=[r for r in pairs if r['comparison']=='V12_vs_BO' and r['task']==trajectory['task'] and r['seed']==trajectory['initialization_seed']]
            if len(matches)!=1:raise ValueError('Support-citation trajectory needs exactly one matched BO control')
            supported_pairs.append(matches[0])
        sw,st,sl=counts(supported_pairs,'AUC_delta')
        mechanism+=f' In a post-hoc descriptive cross-check, the {len(supported)} full-V12 trajectories with at least one accepted support citation have {outcome_counts(sw,st,sl,"SearchAUC")} against matched BO. This co-occurrence does not estimate the effect of citing knowledge; a locally supported forecast is not evidence of a better complete search policy.'
    wr('mechanism_results.tex',mechanism)
    round_path=P/'data/round_decisions.csv';forecast_path=P/'data/forecast_updates.csv'
    with round_path.open(encoding='utf-8-sig',newline='') as handle:round_rows=list(csv.DictReader(handle))
    with forecast_path.open(encoding='utf-8-sig',newline='') as handle:forecast_rows=list(csv.DictReader(handle))
    by_cell={(r['task'],r['initialization_seed'],r['method']):r for r in rows}
    case2='Reizman_Suzuki_Case_2'
    case2_required=[(case2,seed,method) for seed in (0,1,2,3) for method in ('V12','BO')]
    if all(key in by_cell for key in case2_required):
        panel=next(r for r in e['summary'] if r['task']==case2 and r['comparison']=='V12_vs_BO')
        primary=[r for r in pairs if r['task']==case2 and r['comparison']=='V12_vs_BO' and r['seed'] in (1,2,3)]
        seed0=next(r for r in pairs if r['task']==case2 and r['comparison']=='V12_vs_BO' and r['seed']==0)
        if len(primary)!=3 or counts(primary,'AUC_delta')!=(3,0,0) or counts(primary,'best_delta')!=(0,3,0) or seed0['AUC_percent']>=0:
            raise ValueError('Case2 initialization illustration changed')
        paragraph='Suzuki 2 illustrates the dependence on initialization. V12 wins all three primary SearchAUC comparisons with BO, with a '+f"{panel['ratio_of_means_percent']:.2f}"+r'\% ratio-of-means increase, while all three final-best values tie. At initialization 0, used in the separate removal analysis, V12 instead loses '+f"{abs(seed0['AUC_percent']):.2f}"+r'\% of SearchAUC relative to BO. The favourable primary panel does not imply robustness to every starting condition.'
        wr('search_results.tex',(P/'generated/search_results.tex').read_text(encoding='utf-8').strip()+'\n\n'+paragraph)
    case4='Reizman_Suzuki_Case_4'
    case4_primary=[r for r in pairs if r['task']==case4 and r['comparison']=='V12_vs_BO' and r['seed'] in (1,2,3)]
    if len(case4_primary)==3:
        panel=next(r for r in e['summary'] if r['task']==case4 and r['comparison']=='V12_vs_BO')
        if counts(case4_primary,'AUC_delta')!=(2,0,1) or panel['ratio_of_means_percent']>=0:
            raise ValueError('Case4 win-count and magnitude illustration changed')
        paragraph='Win counts and effect magnitudes also differ. On Suzuki 4, V12 wins two of three SearchAUC comparisons with BO, yet the loss at initialization 1 outweighs the two gains: mean SearchAUC is '+f"{abs(panel['ratio_of_means_percent']):.2f}"+r'\% lower. A majority of favourable starting conditions therefore need not produce a positive mean improvement.'
        wr('search_results.tex',(P/'generated/search_results.tex').read_text(encoding='utf-8').strip()+'\n\n'+paragraph)
    hplc_required=[('HPLC',seed,method) for seed in (0,1,2,3) for method in ('V12','BO')]
    if all(key in by_cell for key in hplc_required):
        hp=[r for r in mainpairs if r['task']=='HPLC']
        hpanel=next(r for r in e['summary'] if r['task']=='HPLC' and r['comparison']=='V12_vs_BO')
        h1=next(r for r in hp if r['seed']==1);h2=next(r for r in hp if r['seed']==2)
        hv2=by_cell[('HPLC',2,'V12')];hb2=by_cell[('HPLC',2,'BO')]
        if counts(hp,'AUC_delta')!=(2,0,1) or counts(hp,'best_delta')!=(2,1,0) or not (h1['AUC_percent']>0 and h2['AUC_percent']<0 and h2['best_delta']>0):
            raise ValueError('HPLC initialization and metric contrast changed')
        paragraph='HPLC demonstrates a trade-off between search speed and endpoint quality. V12 wins two of three SearchAUC comparisons with BO and ties or improves all three final maxima. Its '+f"{hpanel['ratio_of_means_percent']:.2f}"+r'\% mean SearchAUC advantage is driven by initialization 1 ('+f"{h1['AUC_percent']:+.2f}"+r'\%). At initialization 2, SearchAUC is '+f"{abs(h2['AUC_percent']):.2f}"+r'\% lower despite a final detector area '+f"{h2['best_delta']:.2f} a.u. higher; the final maxima are first reached at queries {hv2['first_best_query']} and {hb2['first_best_query']} for V12 and BO, respectively. A better endpoint alone would conceal the slower early search."
        wr('search_results.tex',(P/'generated/search_results.tex').read_text(encoding='utf-8').strip()+'\n\n'+paragraph)
    removal_required=[(case2,seed,method) for seed in (0,3) for method in ('V12','No KF','No refresh')]
    if all(key in by_cell for key in removal_required):
        full0=by_cell[(case2,0,'V12')];fixed0=by_cell[(case2,0,'No KF')];once0=by_cell[(case2,0,'No refresh')]
        full3=by_cell[(case2,3,'V12')];fixed3=by_cell[(case2,3,'No KF')];once3=by_cell[(case2,3,'No refresh')]
        f0=[r for r in forecast_rows if r['source']==full0['source']]
        f3=[r for r in forecast_rows if r['source']==full3['source']]
        edits0=[int(r['query']) for r in round_rows if r['source']==full0['source'] and int(r['memory_edit_call'])]
        d0=first_difference(full0,fixed0);d3=first_difference(full3,fixed3);dr=first_difference(full0,once0)
        if not (d0 and d3 and dr and all(int(r['query'])>=d0 for r in f0) and all(int(r['query'])>=d3 for r in f3) and dr<edits0[1] and full3['selected_ids']==once3['selected_ids']):
            raise ValueError('Case2 removal timing illustration changed')
        paragraph=f'The Suzuki 2 removal traces expose this limitation. At initializations 0 and 3, full V12 first differs from no-KF at queries {d0} and {d3}, respectively, with no preceding numerical KF update in full V12. At initialization 0, the no-refresh comparison already diverges at query {dr}, before the first post-initial note-edit call at query {edits0[1]}. At initialization 3, full V12 and no-refresh select all sixteen points in exactly the same order despite '+f"{full3['forecasts']} versus {once3['forecasts']} forecast updates and {full3['edits']} versus {once3['edits']} note calls. These observations constrain module attribution; they do not show that all knowledge states or response packets were identical."
        wr('removal_examples.tex',paragraph)
    else:
        if complete:raise ValueError('Complete cohort lacks Case2 removal traces')
        wr('removal_examples.tex','Detailed paired removal traces remain pending until their matching references are complete.')
    case4_required=[(case4,seed,method) for seed in (0,3) for method in ('V12','No KF','No refresh')]
    if all(key in by_cell for key in case4_required):
        full0=by_cell[(case4,0,'V12')];full3=by_cell[(case4,3,'V12')]
        pp=[next(r for r in pairs if r['task']==case4 and r['seed']==0 and r['comparison']==kind) for kind in ('KF_removal','refresh_removal')]
        f0=[r for r in forecast_rows if r['source']==full0['source']]
        r3=[r for r in round_rows if r['source']==full3['source']]
        firsts=[first_difference(full0,by_cell[(case4,0,method)]) for method in ('No KF','No refresh')]
        add3=sum(review.get('accepted') and review.get('op')=='add' for r in r3 for review in json.loads(r['edit_reviews_json']))
        if not (firsts==[7,7] and len(f0)==5 and all(int(r['query'])<7 for r in f0) and full3['forecasts']==0 and add3==3 and all(int(r['expected_forecasts'])==0 for r in r3)):
            raise ValueError('Case4 removal coverage illustration changed')
        if not all(full3['best_curve']==by_cell[(case4,3,method)]['best_curve'] for method in ('No KF','No refresh')):
            raise ValueError('Case4 equal best-curve illustration changed')
        paragraph='Suzuki 4 provides a different pattern. At initialization 0, full V12 exceeds no-KF and no-refresh SearchAUC by '+f"{pp[0]['AUC_percent']:.2f}"+r'\% and '+f"{pp[1]['AUC_percent']:.2f}"+r'\%, respectively; both selections first diverge at query 7, after five full-policy KF updates. This timing is compatible with a feedback contribution but does not identify it. At initialization 3, all three cumulative-best curves coincide. Full V12 generates no applicable forecast despite three accepted notes during later edits, so its numerical KF branch is not exercised. Knowledge coverage, as well as forecast quality, limits what a removal contrast tests.'
        wr('removal_examples.tex',(P/'generated/removal_examples.tex').read_text(encoding='utf-8').strip()+'\n\n'+paragraph)
    hrem_required=[('HPLC',seed,method) for seed in (0,3) for method in ('V12','No KF','No refresh')]
    if all(key in by_cell for key in hrem_required):
        hh={(r['seed'],r['comparison']):r for r in removals if r['task']=='HPLC'}
        hfull3=by_cell[('HPLC',3,'V12')]
        hk=first_difference(hfull3,by_cell[('HPLC',3,'No KF')]);hr=first_difference(hfull3,by_cell[('HPLC',3,'No refresh')])
        hpauses=[int(r['query']) for r in round_rows if r['source']==hfull3['source'] and int(r['automatic_pauses'])]
        if not (hk==10 and hr==4 and hpauses==[10] and hh[(0,'KF_removal')]['best_delta']<0 and hh[(3,'refresh_removal')]['AUC_delta']<0):
            raise ValueError('HPLC removal timing changed')
        paragraph='HPLC removals give opposing signals. At initialization 0, full V12 gains '+f"{hh[(0,'KF_removal')]['AUC_percent']:.2f}"+r'\% SearchAUC over no-KF but loses '+f"{abs(hh[(0,'KF_removal')]['best_delta']):.2f} a.u. in final area; it exceeds no-refresh SearchAUC by "+f"{hh[(0,'refresh_removal')]['AUC_percent']:.2f}"+r'\%. At initialization 3, full V12 exceeds no-KF SearchAUC by '+f"{hh[(3,'KF_removal')]['AUC_percent']:.2f}"+r'\% but falls '+f"{abs(hh[(3,'refresh_removal')]['AUC_percent']):.2f}"+r'\% below no-refresh. The no-KF selections first diverge at query '+str(hk)+', when full V12 finds its final maximum and subsequently pauses a note for relative forecast loss. The no-refresh selections already diverged at query '+str(hr)+'. The later pause cannot explain the earlier divergence, and independently generated initial notes further limit attribution.'
        wr('removal_examples.tex',(P/'generated/removal_examples.tex').read_text(encoding='utf-8').strip()+'\n\n'+paragraph)
    required=[('Doyle_Buchwald_Hartwig',2,m) for m in ('V12','BO','Reasoning BO')]
    required += [('SnAr',3,m) for m in ('V12','Reasoning BO')]
    required += [('Perera_Suzuki',seed,'V12') for seed in (1,2,3)]
    if all(key in by_cell for key in required):
        doyle=by_cell[('Doyle_Buchwald_Hartwig',2,'V12')]
        controls=[by_cell[('Doyle_Buchwald_Hartwig',2,m)]['final_best'] for m in ('BO','Reasoning BO')]
        early_best=doyle['best_curve'][2]
        doyle_forecasts=[r for r in forecast_rows if r['task']=='Doyle_Buchwald_Hartwig' and r['initialization_seed']=='2' and r['method']=='V12']
        first_update=min(int(r['query']) for r in doyle_forecasts)
        if not (abs(controls[0]-controls[1])<1e-9 and early_best>max(controls) and first_update>2):raise ValueError('Doyle timing illustration no longer matches the saved evidence')
        snar=by_cell[('SnAr',3,'V12')];snar_rbo=by_cell[('SnAr',3,'Reasoning BO')]
        if snar['selected_ids']!=snar_rbo['selected_ids'] or len(snar['selected_ids'])!=16:raise ValueError('SnAr trajectory-identity illustration changed')
        sr=[r for r in round_rows if r['task']=='SnAr' and r['initialization_seed']=='3' and r['method']=='V12']
        sf=[r for r in forecast_rows if r['task']=='SnAr' and r['initialization_seed']=='3' and r['method']=='V12']
        pauses=[int(r['query']) for r in sr if int(r['automatic_pauses'])]
        retractions=[int(r['query']) for r in sr if int(r['automatic_retractions'])]
        negative=[r for r in sf if float(r['Brier_gain'])<0]
        if not (len(pauses)==len(retractions)==1 and len({(r['note_id'],r['version']) for r in sf})==1 and negative and all(int(r['success']) for r in negative)):raise ValueError('SnAr lifecycle illustration changed')
        pf=[r for r in forecast_rows if r['task']=='Perera_Suzuki' and r['initialization_seed'] in ('1','2','3') and r['method']=='V12']
        positive=[r for r in pf if float(r['Brier_gain'])>0]
        no_improvement=sum(not int(r['success']) for r in positive)
        perera_pairs=[r for r in pairs if r['task']=='Perera_Suzuki' and r['comparison']=='V12_vs_BO' and r['seed'] in (1,2,3)]
        if len(perera_pairs)!=3 or not all(r['AUC_delta']<-1e-9 for r in perera_pairs):raise ValueError('Perera outcome illustration changed')
        examples='Post-hoc trajectory inspection makes these distinctions concrete. '
        examples+=f'On Doyle at initialization 2, V12 reached a yield of {early_best:.3f}'+r'\% by query 2, above the '+f'{controls[0]:.3f}'+r'\% final best of both controls; its first data-dependent KF update occurred only at query '+str(first_update)+'. '
        examples+=f'On SnAr at initialization 3, V12 and Reasoning BO selected the same sixteen points in the same order, although the V12 note paused after query {pauses[0]} and retracted after query {retractions[0]}. All {len(negative)} relative forecast losses in that trajectory accompanied incumbent improvements: the LLM forecast was less accurate than the GP forecast of success. '
        examples+=f'Across Perera initializations 1--3, {len(positive)} of {len(pf)} forecasts outperformed the GP probability forecast, but {no_improvement} of those gains occurred on non-improving outcomes; V12 lost all three SearchAUC comparisons with BO. These are descriptive comparisons of complete policies, not isolated module effects.'
        wr('process_examples.tex',examples)
    else:
        if complete:raise ValueError('Complete cohort lacks registered process-example cells')
        wr('process_examples.tex','Registered trace-level illustrations remain pending until their complete comparison cells are available.')
    if ('HPLC',1,'V12') in by_cell:
        h1=by_cell[('HPLC',1,'V12')]
        hr=[r for r in round_rows if r['source']==h1['source']]
        hf=[r for r in forecast_rows if r['source']==h1['source']]
        resumes=[int(r['query']) for r in hr if int(r['automatic_resumes'])]
        if not (resumes==[10] and sum(r['resumes'] for r in full)==1 and all(not int(f['success']) and float(f['Brier_gain'])>0 for f in hf if f['query'] in ('9','10')) and len([f for f in hf if f['query'] in ('9','10')])==2):
            raise ValueError('HPLC recovery illustration changed')
        high=next(r for r in hr if r['query']=='7');h7f=next(f for f in hf if f['query']=='7')
        if not (h7f['status_before']=='paused' and h7f['status_after']=='paused' and int(high['improved'])):
            raise ValueError('HPLC improvement precedes recovery illustration changed')
        paragraph='The sole automatic recovery in the new full-V12 cohort occurs on HPLC at initialization 1, after query 10. It follows positive relative gains at queries 9 and 10, both non-improving experiments. A major area increase to '+f"{float(high['raw_value']):.2f} a.u. had already occurred at query 7 while the note was paused. Thus the recovery pathway executes, but neither that transition nor its two favourable calibration outcomes establish the cause of this trajectory's optimization advantage."
        wr('process_examples.tex',(P/'generated/process_examples.tex').read_text(encoding='utf-8').strip()+'\n\n'+paragraph)
    continuations=e.get('transport_continuations',[])
    if continuations:
        wr('transport_continuations.tex',f'The audit records {len(continuations)} interrupted-request continuation(s). The same assigned agent resumed the identical transport instruction before any response or sealed submission existed. Its first output was retained; no new agent, repair attempt or experimental rerun was introduced. Retained incident and continuation receipts document this exception. The original interruption-tool transcript was unavailable and is explicitly identified as such.')
    else:wr('transport_continuations.tex','No interrupted-request continuation is recorded in the audited evidence at this cutoff.')
    if complete:
        w,t,l=counts(mainpairs,'AUC_delta')
        bw,bt,bl=counts(mainpairs,'best_delta');rw,rt,rl=counts(rbo,'AUC_delta')
        text=f'The completed programme gives a conditional empirical result: {outcome_counts(w,t,l)} in matched SearchAUC comparisons with BO, but {outcome_counts(bw,bt,bl)} in final best. The {outcome_counts(rw,rt,rl)} against Reasoning BO on SearchAUC do not establish a consistent incremental advantage. The observed majority of search-efficiency wins supports further testing of the frozen policy, while the losses and mixed removals limit stronger claims. An executed lifecycle transition is not proof of useful chemistry.'
    else:text='The registered subagent programme remains incomplete at this cutoff. Its unfinished cells cannot support an eight-task performance claim or a conclusion about the independent value of either adaptive module.'
    wr('conclusion_results.tex',text)
    units={'P0':r'Emulated yield (\%)','Perera':r'UV product area (\%)','Doyle':r'Reported yield (\%)','SnAr':r'STY (kg m$^{-3}$ h$^{-1}$)','HPLC':'Peak area (a.u.)'}
    lines=[]
    for d in e['datasets']:
        axis=f"{d['kinds'].count('continuous')} cont.; {d['kinds'].count('categorical')} cat."
        lines.append(' & '.join([d['group'],d['label'],str(d['candidate_count']),axis,units.get(d['label'],units.get(d['group'],''))])+r'\\')
    wr('datasets.tex',r'\begin{table}[t]\centering\small\caption{Frozen candidate pools in this implementation. Counts do not describe every experiment in the original source publication.}\label{tab:data}'+'\n'+r'\begin{tabularx}{\linewidth}{llrlX}\toprule Group & Task & Points & Inputs & Objective\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabularx}\end{table}')
    source=(OLD/'methods.tex').read_text(encoding='utf-8')
    source=source.split('The publication supplement registers 71 additional trajectories:')[0]
    forecast_anchor=r'the LLM records \(p_t=P_{\mathrm{LLM}}(s_t=1)\).'
    if source.count(forecast_anchor)!=1:raise ValueError('Expected prospective-forecast request anchor')
    source=source.replace(forecast_anchor,r'the LLM is asked to record \(p_t=P_{\mathrm{LLM}}(s_t=1)\). Missing or invalid annotations are retained as warnings and do not update that note; a valid official point selection remains executable.')
    anchor='The last expression is the scalar Joseph covariance update.'
    if source.count(anchor)!=1:raise ValueError('Expected KF methods anchor')
    source=source.replace(anchor,anchor+' The recurrence advances only for a valid applicable prospective forecast. A dormant note accumulates neither an update nor an additional process-variance increment at intervening global query positions.')
    source+=r'''
The first publication registration specified 71 trajectories through the original API. Fifteen new BO controls completed; the remaining 56 LLM trajectories produced no new query because API credit was exhausted. A subsequent pre-outcome amendment registered a separate subagent cohort: all 56 LLM cells plus 12 full-V12 references matching the changed execution environment. The 68 new trajectories comprise 32 full V12 runs, 24 Reasoning BO runs and six runs of each removal. The full-V12 references cover all eight tasks at initializations 0--3; Reasoning BO covers all eight at initializations 1--3. The removals use Suzuki Cases 2 and 4 and HPLC at initializations 0 and 3. Matching BO controls are reused, not rerun.

The no-KF intervention retains prospective forecasts, direct Brier-gain facts, versioned memory and explicit LLM revision/retraction. The numerical state stays at its birth values and automatic KF lifecycle transitions are disabled; local observation counts still advance. The no-refresh intervention permits the initial note call only, keeping native KF transitions and the evolving observation/Insight histories. Neither changes the GP, executable menu or experimental budget. No result triggers an additional run or a change in policy.

\subsection*{Request-isolated subagent execution and provenance}
Every new request attempt uses a separately spawned model with requested settings \texttt{gpt-6.1-terra}, medium reasoning effort and \texttt{fork\_turns=none}. A deterministic transport instruction supplies only opaque paths to an input containing exactly the frozen system/public payload and an output file. It supplies no arm or initialization metadata. Decision agents are instructed to read only this packet and write their public JSON response. They do not receive the parent conversation, other-arm outcomes or unused outcomes. A scheduling agent manages execution but does not write or amend scientific responses.

The native validator and maximum three schema-only attempts are retained. Each repair uses a fresh agent with only the original payload and the recorded schema error. The first valid response is sealed regardless of the eventual objective. Raw output bytes, public packets, model-spawn request/result records and per-attempt identities are hashed and preserved. The collaboration interface does not independently attest underlying model weights, provider-side context, token usage, or enforcement of the native caller's requested output-token cap. This is a distinct execution cohort, not an asserted exact reproduction of the API's effective system environment. The file-access restriction is a recorded instruction rather than an operating-system proof of isolation.

\subsection*{Implementation checks and computational accounting}
Frozen source, registered inputs and 15,988 pre-existing files are protected by hashes. Offline audits reconstruct candidate availability, tie/display order, official selection, current-incumbent probabilities, KF transitions, knowledge lifecycle and chronology from saved arrays and commitments. They also check exact public/repair payloads, fresh agent identity and literal output custody. Each completed replay binds the exact state, numerical arrays, commitments, request files and authored output bytes; their hashes are checked before and after replay and again at the final evidence gate. These checks do not independently refit the GP, certify chemical claims or establish externally trusted timestamps.

Two persistent CPU trajectory workers retain the original numerical recipe, with one Torch thread per worker and locked shared GP-prefix caches. The unchanged official confidence extractor is compiled once per process. At most two fresh decision agents are active under one scheduler. The maximum new experimental budget is 1,088 post-initial queries; at most 1,246 logical requests precede any bounded schema repairs. V12 may use sixteen selection calls and up to four memory calls per trajectory, whereas matched Reasoning BO uses sixteen selection calls. Experimental budgets are matched; model-call and token budgets are not asserted equal. Unavailable token accounting is not reported as zero cost. Failed attempts and all measured outcomes remain in their original namespaces.
'''
    environment_path=P/'data/runtime_versions.json'
    environment=json.loads(environment_path.read_text(encoding='utf-8'))
    source+='\nThe numerical worker environment recorded during the subagent study contains '+', '.join(
        label+' '+environment[key] for key,label in [('python','Python'),('torch','CPU PyTorch'),('botorch','BoTorch'),('gpytorch','GPyTorch'),('numpy','NumPy'),('scipy','SciPy')])+'. The version receipt is supplied with the exported data; it is not used to infer the environments of older development runs.\n'
    (P/'methods.tex').write_text(source,encoding='utf-8')
    names=('abstract','evidence_status','search_results','final_results','removal_results','removal_examples','mechanism_results','process_examples','conclusion_results','datasets','transport_continuations')
    files=[P/'generated'/f'{name}.tex' for name in names]+[P/'methods.tex']
    receipt=dict(evidence_sha256=sha256((P/'data/evidence.json').read_bytes()).hexdigest(),generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={'data/runtime_versions.json':sha256(environment_path.read_bytes()).hexdigest(),'data/mechanism_coverage.json':sha256(coverage_path.read_bytes()).hexdigest(),'data/round_decisions.csv':sha256(round_path.read_bytes()).hexdigest(),'data/forecast_updates.csv':sha256(forecast_path.read_bytes()).hexdigest()},
                 files={f.relative_to(P).as_posix():sha256(f.read_bytes()).hexdigest() for f in files})
    (ROOT/'experiments/reports/paper_v12_subagent/text_generation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=n,abstract_words=len(abstract.split()),generated_fragments=len(names),legacy_cohort_separate=True)))
if __name__=='__main__':main()
