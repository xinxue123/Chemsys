"""DeepSeek API results, independent replay and prior-artifact integrity checks."""
from copy import deepcopy
from pathlib import Path
import csv
import json
import statistics as st
import xml.etree.ElementTree as ET

import numpy as np

from . import deepseek_agent_suzuki_study as s
from . import deepseek_suzuki_client as api
from . import agent_score_suzuki_report as prior
from .suzuki_domain import digest,CATALYSTS

METHODS=('DeepSeek_score_adaptive','DeepSeek_score_frozen_initial','Current_assistant_rules','qLogEI_B0','UCB_original_B0','UCB_kappa2_B0')


def rows():
    old=prior.rows();result={m:[] for m in METHODS}
    for case in (1,2,3,4):
        for seed in range(5):
            key=f'c{case}_s{seed}'
            for method,folder in zip(METHODS[:2],('runs','frozen')):result[method].append(s.read(s.ROOT/folder/key/'state.json'))
    result['Current_assistant_rules']=old['Agent_score_adaptive']
    for method in METHODS[3:]:result[method]=old[method]
    return result


def summary(group):
    values=[prior.metric(r) for r in group];observed=[h['y']*100 for r in group for h in r['history'][4:]]
    return {'n':len(group),'complete':sum(r['status']=='COMPLETE' for r in group),'failed':sum(r['status']=='FAILED' for r in group),
            'SearchAUC_mean':st.mean(v['AUC'] for v in values),'SearchAUC_sd':st.stdev(v['AUC'] for v in values),
            'final_yield_mean':st.mean(v['final'] for v in values),'initial_yield_mean':st.mean(v['initial'] for v in values),
            'added_query_yield_mean':st.mean(observed) if observed else None,'observed_added_queries':len(observed),'planned_added_queries':16*len(group),
            'improving_queries':sum(v['improving_added_queries'] for v in values),
            'mean_curve':[st.mean(v['curve'][j] for v in values) for j in range(16)]}


def api_usage():
    responses=[s.read(f) for f in sorted((s.ROOT/'requests').glob('*/attempt*/response.json'))]
    errors=list((s.ROOT/'requests').glob('*/attempt*/transport_error.json'))
    return {'confirmed_API_responses':len(responses),'transport_errors':len(errors),'total_attempts':len(responses)+len(errors),
            'schema_failures':len(list((s.ROOT/'requests').glob('*/attempt*/validation_error.json'))),
            'returned_models':sorted({r['model'] for r in responses}),
            'system_fingerprints':sorted({r['system_fingerprint'] for r in responses if r.get('system_fingerprint')}),
            'prompt_tokens':sum(r['usage'].get('prompt_tokens',0) for r in responses),
            'completion_tokens':sum(r['usage'].get('completion_tokens',0) for r in responses),
            'total_tokens':sum(r['usage'].get('total_tokens',0) for r in responses),
            'reasoning_tokens_included_in_completion':sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in responses),
            'prompt_cache_hit_tokens':sum(r['usage'].get('prompt_cache_hit_tokens',0) for r in responses),
            'response_wall_seconds_sum':sum(r.get('wall_seconds',0.) for r in responses),
            'preflight_responses_separate':1,'preflight_usage':s.read(s.OUT/'preflight.json')['usage'],
            'limitations':'Confirmed response usage only; a timed-out request may have incurred unobserved provider work. No price estimate or immutable-model guarantee.'}


def report():
    p=s.check_protocol();groups=rows();assert all(len(v)==20 and all(r['status'] in ('COMPLETE','FAILED') for r in v) for v in groups.values())
    tables={};pairs={}
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4)),('Case1_reference',(1,)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        subset={m:[r for r in group if r['case'] in cases] for m,group in groups.items()}
        tables[scope]={m:summary(group) for m,group in subset.items()}
        pairs[scope]={m:prior.compare(subset[METHODS[0]],subset[m]) for m in METHODS[1:]}
    new=[r for method in METHODS[:2] for r in groups[method]];cards=sum(len(r['decisions']) for r in groups[METHODS[0]])
    result={'protocol_sha256':p['protocol_sha256'],'requested_model':api.MODEL,'request_parameters':api.PARAMETERS,
            'planned_new_trajectories':40,'completed_new_trajectories':sum(r['status']=='COMPLETE' for r in new),'failed_new_trajectories':sum(r['status']=='FAILED' for r in new),
            'planned_logical_queries':800,'observed_logical_queries':sum(len(r['history']) for r in new),'valid_API_scorecards':cards,
            'candidate_scores':cards*4692,'GP_fits':0,'operator_compiles_in_loop':0,'usage':api_usage(),'tables':tables,'DeepSeek_minus_control':pairs,
            'raw_metrics':[{'method':m,'case':r['case'],'seed':r['seed'],'status':r['status']}|prior.metric(r) for m,group in groups.items() for r in group],
            'limits':['Restricted scorecard executor shared with prior study, not independent per-candidate LLM scoring.',
                      'API inputs isolate each trajectory; current-assistant reference used shared chat and a different provider/prompt/context, so its difference is not a controlled model-only effect.',
                      'Model alias and observed fingerprint do not establish an immutable provider snapshot or exclude benchmark pretraining exposure.',
                      'Thinking high uses provider default sampling; temperature0 would be ignored, so no temperature or seed supplied. Five seeds vary experiment initialization, not independent inference replications at each initialization.',
                      'Frozen first-card control also freezes initial observation geometry; its difference is a whole-feedback-policy contrast.',
                      'Only three primary environments, five initializations each; exploratory descriptive paired comparisons, no significance or general superiority claim.',
                      'Failures retain incumbent for all planned budget positions; observed-query yield diagnostics use their stated observed denominators.']}
    s.write(s.OUT/'results.json',result)
    with (s.OUT/'metrics.csv').open('w',encoding='utf8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['scope','method','n','complete','failed','SearchAUC_pp','final_yield_percent'])
        for scope,table in tables.items():
            for method,row in table.items():writer.writerow([scope,method,row['n'],row['complete'],row['failed'],row['SearchAUC_mean'],row['final_yield_mean']])
    figures(result)
    return {'completed':result['completed_new_trajectories'],'failed':result['failed_new_trajectories'],'cards':cards,
            'all_four':{m:{k:round(v[k],4) for k in ('SearchAUC_mean','final_yield_mean')} for m,v in tables['all_four'].items()},
            'primary':{m:{k:round(v[k],4) for k in ('SearchAUC_mean','final_yield_mean')} for m,v in tables['primary'].items()},'usage':result['usage']}


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=('DeepSeek adaptive','DeepSeek frozen first card','Prior chat agent reference','qLogEI BO','UCB original beta','UCB kappa=2')
    colors=('#0072B2','#777777','#CC79A7','#D55E00','#56B4E9','#009E73')
    fig,axes=plt.subplots(1,4,figsize=(14,3.7),layout='constrained')
    for c,ax in enumerate(axes,1):
        for method,label,color in zip(METHODS,labels,colors):
            ax.plot(range(5,21),result['tables'][f'Case{c}'][method]['mean_curve'],label=label,color=color,lw=1.8,ls='--' if method=='Current_assistant_rules' or 'UCB' in method else '-')
        ax.set(title=f'Case {c}',xlabel='Total queries',xticks=[5,10,15,20]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean incumbent yield (%)');axes[-1].legend(fontsize=6.8,loc='lower right')
    fig.suptitle('deepseek-flash API, high thinking: isolated observed prefixes, same score executor and 4+16 budget',fontsize=10.5)
    fig.savefig(s.OUT/'incumbent_curves.png',dpi=180,bbox_inches='tight');fig.savefig(s.OUT/'incumbent_curves.pdf',bbox_inches='tight');plt.close(fig)


def verify(config_path='../config.txt'):
    p=s.check_protocol();groups=rows();freeze=s.read(s.OUT/'all_responses_frozen.json');assert freeze['frozen_states_before_start']==0
    adaptive_checks=0;frozen_checks=0;score_checks=0;attempt_checks=0
    for row,control in zip(groups[METHODS[0]],groups[METHODS[1]]):
        case,seed=row['case'],row['seed'];folder=s.PARENT/'inputs'/f'case{case}';pool=s.read(folder/'pool.json');ids=pool['ids'];ix={v:i for i,v in enumerate(ids)}
        values=np.load(folder/'oracle.npz')['values'];initial=pool['initial_ids'][str(seed)]
        assert row['history'][:4]==control['history'][:4] and [h['candidate_id'] for h in row['history'][:4]]==initial
        assert s.filehash(s.ROOT/'runs'/row['run_id']/'state.json')==freeze['adaptive_states_sha256'][row['run_id']]
        for state in (row,control):
            assert state['status'] in ('COMPLETE','FAILED') and state['protocol_sha256']==p['protocol_sha256']
            assert len(state['history'])==len(state['decisions'])+4<=20
            if state['status']=='COMPLETE':assert len(state['decisions'])==16
            assert len({h['candidate_id'] for h in state['history']})==len(state['history'])
            for h in state['history']:assert h['y']==float(values[ix[h['candidate_id']]])
        for t,event in enumerate(row['decisions'],1):
            h=row['history'][:t+3];context=event['context_sha256'];directory=s.ROOT/'requests'/context;request=s.read(directory/'context.json');raw=deepcopy(request);raw.pop('context_sha256')
            assert digest(raw)==context and request['history']==h and request['protocol_sha256']==p['protocol_sha256']
            public=request['public'];assert set(public)=={'task','iteration','remaining_queries','domain','observations','previous_card'}
            assert public['task']==row['task'] and public['iteration']==t and public['remaining_queries']==17-t
            assert public['previous_card']==(row['decisions'][t-2]['card'] if t>1 else None) and len(public['observations'])==len(h)
            for index,(obs,observation) in enumerate(zip(public['observations'],h),1):
                j=ix[observation['candidate_id']];assert set(obs)=={'index','cat','z','yield_percent'}
                assert obs['index']==index and obs['cat']==CATALYSTS.index(pool['points'][j]['catalyst'])
                assert obs['z']==[round(float(v),4) for v in pool['z'][j][:3]] and obs['yield_percent']==round(observation['y']*100,3)
            validated=s.read(directory/'validated.json');assert validated['card']==event['card'] and validated['accepted_attempt']==event['accepted_attempt']
            accepted=validated['accepted_attempt'];assert 1<=accepted<=3
            for attempt in range(1,accepted+1):
                body=s.read(directory/f'attempt{attempt:02d}'/'request.json');assert {k:v for k,v in body.items() if k!='messages'}==api.PARAMETERS
                assert body['messages'][:2]==api.messages_for(public)
                if attempt>1 and len(body['messages'])>2:assert len(body['messages'])==4
                attempt_checks+=1
            response_path=directory/f'attempt{accepted:02d}'/'response.json';answer=s.read(response_path)
            assert s.filehash(response_path)==validated['response_sha256'] and api.parse_card(answer,h)==event['card']
            assert answer['id']==event['response_id'] and answer['model']==event['response_model'] and not answer['retained_raw_reasoning']
            assert all('reasoning_content' not in choice['message'] for choice in answer['choices'])
            score_key=digest({'context':context,'card':event['card']});assert event['score_key']==score_key
            with np.load(s.ROOT/'scores'/(score_key+'.npz')) as data:scores=data['scores'].copy()
            independent=prior.independent_scores(pool,h,event['card']);assert len(scores)==event['candidates_scored']==4692
            assert np.allclose(scores,independent,atol=1e-10,rtol=1e-12)
            observed={v['candidate_id'] for v in h};available=[j for j,key in enumerate(ids) if key not in observed]
            selected=min(available,key=lambda j:(-float(independent[j]),ids[j]))
            assert event['selected_id']==ids[selected]==row['history'][t+3]['candidate_id'] and event['score']==float(scores[selected]);adaptive_checks+=1;score_checks+=len(scores)
        if row['decisions']:
            fixed=prior.independent_scores(pool,row['history'][:4],row['decisions'][0]['card'])
            for t,event in enumerate(control['decisions'],1):
                h=control['history'][:t+3];available=[j for j,key in enumerate(ids) if key not in {v['candidate_id'] for v in h}]
                selected=min(available,key=lambda j:(-float(fixed[j]),ids[j]))
                assert event['selected_id']==ids[selected]==control['history'][t+3]['candidate_id'] and abs(event['score']-fixed[selected])<1e-10;frozen_checks+=1
            assert row['decisions'][0]['selected_id']==control['decisions'][0]['selected_id']
    protected={};trees={}
    for label,path in s.PROTECTED.items():
        inv=s.read(path)
        for f in inv['files']:assert s.filehash(f['path'])==f['sha256'],f['path']
        protected[label]=len(inv['files'])
        for key,expected in inv.items():
            if key.endswith('cache_tree') or key=='score_tree':
                directory='scores' if key=='score_tree' else key.removesuffix('_tree');actual=prior.tree(Path('data/runs')/f'suzuki_{label}'/directory)
                assert actual=={k:expected[k] for k in actual},(label,key);trees[label+'_'+directory]=actual
    results=s.read(s.OUT/'results.json')
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4))]:
        for method,group in groups.items():
            subset=[r for r in group if r['case'] in cases]
            auc=st.mean(sum(max(h['y'] for h in r['history'][:min(i,len(r['history']))])*100 for i in range(5,21))/16 for r in subset)
            assert abs(auc-results['tables'][scope][method]['SearchAUC_mean'])<1e-10
            assert all(r['history'][:4]==groups[METHODS[0]][i]['history'][:4] for i,r in enumerate(group))
    # Check the actual credential without serializing its value or its hash.
    config=api.load_config(config_path);scanned=0
    for root in (s.ROOT,s.OUT):
        for f in root.rglob('*'):
            if not f.is_file() or any(part.startswith('pytest_tmp') for part in f.parts) or f.suffix not in ('.json','.log','.xml','.csv'):continue
            assert config.api_key not in f.read_text(encoding='utf8'), 'credential leaked into output';scanned+=1
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'independent_adaptive_choices':adaptive_checks,
             'independent_frozen_choices':frozen_checks,'independently_recomputed_candidate_scores':score_checks,'API_attempt_bodies_checked':attempt_checks,
             'planned_runs':40,'planned_logical_queries':800,'observed_logical_queries':results['observed_logical_queries'],
             'API_usage':api_usage(),'GP_fits':0,'operator_compiles_in_loop':0,'protected_files_unchanged':protected,'protected_trees_unchanged':trees,
             'credential_scan_text_files':scanned,'credential_leaks':0,'results_sha256':s.filehash(s.OUT/'results.json'),
             'boundary':'Implementation integrity and API provenance; scientific performance and model/context limitations remain separate.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def release():
    p=s.check_protocol();receipt=s.read(s.OUT/'validation_receipt.json');assert receipt['integrity_status']=='PASS'
    suites=list(ET.parse(s.OUT/'component_tests.xml').getroot().iter('testsuite'));counts={k:sum(int(v.attrib.get(k,0)) for v in suites) for k in ('tests','failures','errors','skipped')}
    assert counts['tests']>=7 and counts['errors']==counts['failures']==0
    files=[Path(__file__),Path(s.__file__),Path(api.__file__),Path('chem/benchmarks/suzuki_deepseek_agent.py'),Path('tests/test_suzuki_deepseek_agent.py'),Path('docs/suzuki_deepseek_agent.md')]
    files += [f for f in s.ROOT.rglob('*') if f.is_file() and not any(part.startswith('pytest_tmp') or part=='scores' for part in f.relative_to(s.ROOT).parts)]
    files += [f for f in s.OUT.rglob('*') if f.is_file() and f.name not in ('release_inventory.json','release_receipt.json')]
    inventory={'protocol_sha256':p['protocol_sha256'],'file_count':len(set(files)),
               'files':[{'path':f.as_posix(),'bytes':f.stat().st_size,'sha256':s.filehash(f)} for f in sorted(set(files))],
               'score_tree':prior.tree(s.ROOT/'scores'),'digest_definition':'Sorted native Path relative POSIX path + NUL + file SHA256 + LF.'}
    s.write(s.OUT/'release_inventory.json',inventory);result=s.read(s.OUT/'results.json')
    output={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'model_requested':api.MODEL,
            'complete_new_trajectories':result['completed_new_trajectories'],'failed_new_trajectories':result['failed_new_trajectories'],
            'planned_logical_queries':800,'observed_logical_queries':result['observed_logical_queries'],'valid_API_cards':result['valid_API_scorecards'],
            'confirmed_formal_API_responses':result['usage']['confirmed_API_responses'],'preflight_API_responses':1,
            'component_tests':counts,'GP_fits':0,'operator_compiles_in_loop':0,'credential_leaks':0,
            'validation_sha256':s.filehash(s.OUT/'validation_receipt.json'),'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),
            'boundary':'Real isolated-prefix deepseek-flash API experiment on restricted scoring rules; no immutable snapshot or general model superiority claim.'}
    s.write(s.OUT/'release_receipt.json',output);return output
