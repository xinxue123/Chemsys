"""Single-seed DeepSeek results and independent provenance/scoring verification."""
from copy import deepcopy
from pathlib import Path
import csv
import json
import statistics as st
import xml.etree.ElementTree as ET

import numpy as np

from . import deepseek_small_suzuki_study as s
from . import deepseek_suzuki_client as api
from . import agent_score_suzuki_report as old
from .suzuki_domain import digest,CATALYSTS

METHODS=('DeepSeek_score_adaptive','DeepSeek_score_frozen_initial','Current_assistant_rules','qLogEI_B0','UCB_original_B0','UCB_kappa2_B0')


def rows():
    previous=old.rows();result={m:[] for m in METHODS}
    for c in (1,2,3,4):
        for method,folder in zip(METHODS[:2],('runs','frozen')):result[method].append(s.read(s.ROOT/folder/f'c{c}_s0'/'state.json'))
    result['Current_assistant_rules']=[r for r in previous['Agent_score_adaptive'] if r['seed']==0]
    for method in METHODS[3:]:result[method]=[r for r in previous[method] if r['seed']==0]
    return result


def summary(group):
    values=[old.metric(r) for r in group];observed=[h['y']*100 for r in group for h in r['history'][4:]]
    return {'n':len(group),'seed_replicates_per_case':1,'complete':sum(r['status']=='COMPLETE' for r in group),
            'SearchAUC_mean':st.mean(v['AUC'] for v in values),'final_yield_mean':st.mean(v['final'] for v in values),
            'added_query_yield_mean':st.mean(observed) if observed else None,'observed_added_queries':len(observed),'planned_added_queries':16*len(group),
            'improving_queries':sum(v['improving_added_queries'] for v in values),'mean_curve':[st.mean(v['curve'][j] for v in values) for j in range(16)]}


def usage():
    answers=[s.read(p) for p in (s.ROOT/'requests').glob('*/attempt*/response.json')];errors=[s.read(p) for p in (s.ROOT/'requests').glob('*/attempt*/transport_error.json')]
    return {'received_API_responses':len(answers),'HTTP_errors':len(errors),'schema_failures':len(list((s.ROOT/'requests').glob('*/attempt*/validation_error.json'))),
            'HTTP_error_statuses':{str(k):sum(e['error'].get('status')==k for e in errors) for k in sorted({e['error'].get('status') for e in errors if e['error'].get('status') is not None})},
            'returned_models':sorted({r['model'] for r in answers}),'fingerprints':sorted({r['system_fingerprint'] for r in answers if r.get('system_fingerprint')}),
            **{key:sum(r['usage'].get(key,0) for r in answers) for key in ('prompt_tokens','completion_tokens','total_tokens')},
            'reasoning_tokens_included_in_completion':sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in answers),
            'response_wall_seconds_sum':sum(r['wall_seconds'] for r in answers),'interrupted_initial_attempt_separate':s.read(s.initial.OUT/'interruption_receipt.json')}


def report():
    p=s.check_protocol();groups=rows();assert all(len(v)==4 and all(r['status'] in ('COMPLETE','FAILED') for r in v) for v in groups.values())
    tables={};paired={}
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        subset={m:[r for r in group if r['case'] in cases] for m,group in groups.items()}
        tables[scope]={m:summary(v) for m,v in subset.items()};paired[scope]={m:old.compare(subset[METHODS[0]],subset[m]) for m in METHODS[1:]}
    new=[r for m in METHODS[:2] for r in groups[m]];cards=sum(len(r['decisions']) for r in groups[METHODS[0]])
    result={'protocol_sha256':p['protocol_sha256'],'scope':'Cases1-4, fixed seed0, one trajectory per case and method; user-reduced pilot.',
            'model_requested':api.MODEL,'parameters':s.PARAMETERS,'planned_new_trajectories':8,'completed_new_trajectories':sum(r['status']=='COMPLETE' for r in new),
            'planned_logical_queries':160,'observed_logical_queries':sum(len(r['history']) for r in new),'valid_scorecards':cards,'candidate_scores':cards*4692,
            'tables':tables,'DeepSeek_minus_control':paired,'usage':usage(),
            'raw_metrics':[{'method':m,'case':r['case'],'seed':r['seed'],'status':r['status']}|old.metric(r) for m,group in groups.items() for r in group],
            'limits':['One fixed initialization per case, no within-case variance or significance estimate; not comparable to old five-seed means.',
                      'Same scoring rule family/pool/query budget; API fresh own-prefix context differs from old shared-chat assistant reference.',
                      'Thinking high, max_tokens16384, no temperature/seed sent; alias not an immutable provider snapshot.',
                      'Restricted agent-generated scorecards, not independent LLM assessment of each candidate.',
                      'Frozen control fixes both first card and initial geometry; contrast is complete feedback policy, not outcome-only causal effect.',
                      'Initial interrupted35 other-seed cards excluded from selected metrics, retained separately as interface/cost evidence.']}
    s.write(s.OUT/'results.json',result)
    with (s.OUT/'metrics.csv').open('w',encoding='utf8',newline='') as handle:
        w=csv.writer(handle);w.writerow(['scope','method','n','SearchAUC_pp','final_yield_percent'])
        for scope,table in tables.items():
            for method,r in table.items():w.writerow([scope,method,r['n'],r['SearchAUC_mean'],r['final_yield_mean']])
    figures(result)
    return {'complete':result['completed_new_trajectories'],'cards':cards,'all_four':{m:{k:round(r[k],4) for k in ('SearchAUC_mean','final_yield_mean')} for m,r in tables['all_four'].items()},'usage':result['usage']}


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=('DeepSeek adaptive','DeepSeek frozen first card','Prior chat agent reference','qLogEI BO','UCB original beta','UCB kappa=2');colors=('#0072B2','#777777','#CC79A7','#D55E00','#56B4E9','#009E73')
    fig,axes=plt.subplots(1,4,figsize=(14,3.7),layout='constrained')
    for c,ax in enumerate(axes,1):
        for m,label,color in zip(METHODS,labels,colors):ax.plot(range(5,21),result['tables'][f'Case{c}'][m]['mean_curve'],label=label,color=color,lw=1.8,ls='--' if m=='Current_assistant_rules' or 'UCB' in m else '-')
        ax.set(title=f'Case {c}, seed 0',xlabel='Total queries',xticks=[5,10,15,20]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Incumbent yield (%)');axes[-1].legend(fontsize=6.8,loc='lower right');fig.suptitle('User-reduced pilot: deepseek-flash API, 1 initialization per case, same 4+16 budget',fontsize=10.5)
    fig.savefig(s.OUT/'incumbent_curves.png',dpi=180,bbox_inches='tight');fig.savefig(s.OUT/'incumbent_curves.pdf',bbox_inches='tight');plt.close(fig)


def verify(config_path='../config.txt'):
    p=s.check_protocol();groups=rows();freeze=s.read(s.OUT/'all_responses_frozen.json');assert freeze['frozen_states_before_start']==0
    decisions=0;scores_checked=0;frozen_decisions=0;attempts=0
    for row,control in zip(groups[METHODS[0]],groups[METHODS[1]]):
        c=row['case'];folder=s.PARENT/'inputs'/f'case{c}';pool=s.read(folder/'pool.json');ids=pool['ids'];ix={v:i for i,v in enumerate(ids)};values=np.load(folder/'oracle.npz')['values']
        assert row['history'][:4]==control['history'][:4] and [h['candidate_id'] for h in row['history'][:4]]==pool['initial_ids']['0']
        assert s.filehash(s.ROOT/'runs'/row['run_id']/'state.json')==freeze['state_sha256'][row['run_id']]
        for state in (row,control):
            assert state['status'] in ('COMPLETE','FAILED') and len(state['history'])==len(state['decisions'])+4<=20
            assert state['protocol_sha256']==p['protocol_sha256'] and state['seed']==0
            if state['status']=='COMPLETE':assert len(state['decisions'])==16
            assert len({h['candidate_id'] for h in state['history']})==len(state['history'])
            for h in state['history']:assert h['y']==float(values[ix[h['candidate_id']]])
        for t,event in enumerate(row['decisions'],1):
            h=row['history'][:t+3];key=event['context_sha256'];directory=s.ROOT/'requests'/key;context=s.read(directory/'context.json');raw=deepcopy(context);raw.pop('context_sha256')
            assert digest(raw)==key and context['history']==h and context['protocol_sha256']==p['protocol_sha256']
            public=context['public'];assert set(public)=={'task','iteration','remaining_queries','domain','observations','previous_card'}
            assert public['task']==row['task'] and public['iteration']==t and public['remaining_queries']==17-t and len(public['observations'])==len(h)
            assert public['previous_card']==(row['decisions'][t-2]['card'] if t>1 else None)
            for i,(obs,observed) in enumerate(zip(public['observations'],h),1):
                j=ix[observed['candidate_id']];assert set(obs)=={'index','cat','z','yield_percent'}
                assert obs['index']==i and obs['cat']==CATALYSTS.index(pool['points'][j]['catalyst'])
                assert obs['z']==[round(float(v),4) for v in pool['z'][j][:3]] and obs['yield_percent']==round(observed['y']*100,3)
            valid=s.read(directory/'validated.json');accepted=valid['accepted_attempt'];assert 1<=accepted<=8 and valid['received_responses']<=3
            received=0
            for attempt in range(1,accepted+1):
                body=s.read(directory/f'attempt{attempt:02d}'/'request.json');assert {k:v for k,v in body.items() if k!='messages'}==s.PARAMETERS
                assert body['messages'][:2]==api.messages_for(public);received+=int((directory/f'attempt{attempt:02d}'/'response.json').exists());attempts+=1
            assert received==valid['received_responses']
            response_path=directory/f'attempt{accepted:02d}'/'response.json';answer=s.read(response_path)
            assert api.parse_card(answer,h)==event['card']==valid['card'] and s.filehash(response_path)==valid['response_sha256']
            assert answer['id']==event['response_id'] and answer['model']==event['response_model'] and not answer['retained_raw_reasoning']
            score_key=digest({'context':key,'card':event['card']});assert score_key==event['score_key']
            with np.load(s.ROOT/'scores'/(score_key+'.npz')) as data:scores=data['scores'].copy()
            independent=old.independent_scores(pool,h,event['card']);assert len(scores)==event['candidates_scored']==4692 and np.allclose(scores,independent,atol=1e-10,rtol=1e-12)
            available=[j for j,key in enumerate(ids) if key not in {v['candidate_id'] for v in h}];j=min(available,key=lambda j:(-float(independent[j]),ids[j]))
            assert ids[j]==event['selected_id']==row['history'][t+3]['candidate_id'] and event['score']==float(scores[j]);decisions+=1;scores_checked+=len(scores)
        if row['decisions']:
            fixed=old.independent_scores(pool,row['history'][:4],row['decisions'][0]['card'])
            for t,event in enumerate(control['decisions'],1):
                h=control['history'][:t+3];available=[j for j,key in enumerate(ids) if key not in {v['candidate_id'] for v in h}];j=min(available,key=lambda j:(-float(fixed[j]),ids[j]))
                assert event['selected_id']==ids[j]==control['history'][t+3]['candidate_id'] and abs(event['score']-fixed[j])<1e-10;frozen_decisions+=1
            assert row['decisions'][0]['selected_id']==control['decisions'][0]['selected_id']
    protected={};trees={}
    for label,path in s.PROTECTED.items():
        inv=s.read(path)
        for f in inv['files']:assert s.filehash(f['path'])==f['sha256'],f['path']
        protected[label]=len(inv['files'])
        for key,expected in inv.items():
            if key.endswith('cache_tree') or key=='score_tree':
                folder='scores' if key=='score_tree' else key.removesuffix('_tree');actual=old.tree(Path('data/runs')/f'suzuki_{label}'/folder)
                assert actual=={k:expected[k] for k in actual};trees[label+'_'+folder]=actual
    result=s.read(s.OUT/'results.json')
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4))]:
        for method,group in groups.items():
            subset=[r for r in group if r['case'] in cases]
            auc=st.mean(sum(max(h['y'] for h in r['history'][:min(i,len(r['history']))])*100 for i in range(5,21))/16 for r in subset)
            assert abs(auc-result['tables'][scope][method]['SearchAUC_mean'])<1e-10
            assert all(r['history'][:4]==groups[METHODS[0]][i]['history'][:4] for i,r in enumerate(group))
    cfg=api.load_config(config_path);scanned=0
    for root in (s.ROOT,s.OUT):
        for f in root.rglob('*'):
            if not f.is_file() or f.suffix not in ('.json','.log','.xml','.csv') or any(part.startswith('pytest_tmp') for part in f.parts):continue
            assert cfg.api_key not in f.read_text(encoding='utf8'),'credential leaked into artifact';scanned+=1
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'planned_trajectories':8,'planned_logical_queries':160,
             'observed_logical_queries':result['observed_logical_queries'],'adaptive_choices':decisions,'frozen_choices':frozen_decisions,
             'independently_recomputed_candidate_scores':scores_checked,'API_attempts_verified':attempts,'usage':usage(),
             'GP_fits':0,'operator_compiles_in_loop':0,'protected_files_unchanged':protected,'protected_trees_unchanged':trees,
             'credential_scan_text_files':scanned,'credential_leaks':0,'results_sha256':s.filehash(s.OUT/'results.json'),
             'boundary':'Single-seed implementation/API provenance qualification, not a replicated scientific superiority claim.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def release():
    p=s.check_protocol();v=s.read(s.OUT/'validation_receipt.json');assert v['integrity_status']=='PASS';result=s.read(s.OUT/'results.json')
    suites=list(ET.parse(s.OUT/'component_tests.xml').getroot().iter('testsuite'));tests={k:sum(int(x.attrib.get(k,0)) for x in suites) for k in ('tests','failures','errors','skipped')};assert tests['tests']>=15 and tests['failures']==tests['errors']==0
    files=[Path(__file__),Path(s.__file__),Path('chem/benchmarks/suzuki_deepseek_small.py'),Path('tests/test_suzuki_deepseek_small.py'),Path('docs/suzuki_deepseek_small.md')]
    files += [f for f in s.ROOT.rglob('*') if f.is_file() and not any(part.startswith('pytest_tmp') or part=='scores' for part in f.relative_to(s.ROOT).parts)]
    files += [f for f in s.OUT.rglob('*') if f.is_file() and f.name not in ('release_inventory.json','release_receipt.json')]
    inv={'protocol_sha256':p['protocol_sha256'],'file_count':len(set(files)),'files':[{'path':f.as_posix(),'bytes':f.stat().st_size,'sha256':s.filehash(f)} for f in sorted(set(files))],'score_tree':old.tree(s.ROOT/'scores'),'digest_definition':'Sorted native Path relative POSIX path + NUL + file SHA256 + LF.'}
    s.write(s.OUT/'release_inventory.json',inv)
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'model_requested':api.MODEL,'seed':0,'cases':[1,2,3,4],
             'complete_new_trajectories':result['completed_new_trajectories'],'logical_queries':result['observed_logical_queries'],'valid_scorecards':result['valid_scorecards'],
             'received_API_responses':result['usage']['received_API_responses'],'GP_fits':0,'operator_compiles_in_loop':0,'component_tests':tests,'credential_leaks':0,
             'validation_sha256':s.filehash(s.OUT/'validation_receipt.json'),'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),
             'scope':'User-reduced single-seed pilot; original interrupted attempt retained separately and excluded from performance.'}
    s.write(s.OUT/'release_receipt.json',receipt);return receipt
