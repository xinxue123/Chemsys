"""Independent outcome summaries and score-table/prefix integrity verification."""
from hashlib import sha256
from pathlib import Path
import csv
import json
import statistics as st
import time
import xml.etree.ElementTree as ET

import numpy as np

from . import agent_score_suzuki_study as s
from .suzuki_domain import digest,CATALYSTS

METHODS=('Agent_score_adaptive','Agent_score_frozen_initial','qLogEI_B0','UCB_original_B0','UCB_kappa2_B0')


def metric(row):
    y=[v['y'] for v in row['history']];curve=[max(y[:min(i,len(y))])*100 for i in range(5,21)]
    return {'AUC':st.mean(curve),'final':max(y)*100,'initial':max(y[:4])*100,'curve':curve,
            'added_query_mean_yield':st.mean(y[4:])*100 if len(y)>4 else None,
            'improving_added_queries':sum(y[i]>max(y[:i])+1e-10 for i in range(4,len(y)))}


def rows():
    result={m:[] for m in METHODS}
    for c in (1,2,3,4):
        for seed in range(5):
            key=f'c{c}_s{seed}';stage='development' if c==1 else 'test'
            result[METHODS[0]].append(s.read(s.ROOT/'runs'/key/'state.json'))
            result[METHODS[1]].append(s.read(s.ROOT/'frozen'/key/'state.json'))
            result[METHODS[2]].append(s.read(s.PARENT/stage/'runs'/(key+'_none_B0')/'state.json'))
            result[METHODS[3]].append(s.read(Path('data/runs/suzuki_acquisition/main/runs')/(key+'_none_official')/'state.json'))
            folder='development' if c==1 else 'main'
            result[METHODS[4]].append(s.read(Path('data/runs/suzuki_acquisition')/folder/'runs'/(key+'_none_fixed_k2')/'state.json'))
    return result


def summary(group):
    v=[metric(r) for r in group]
    return {'n':len(v),'completed':sum(r['status']=='COMPLETE' for r in group),
            'SearchAUC_mean':st.mean(r['AUC'] for r in v),'SearchAUC_sd':st.stdev(r['AUC'] for r in v),
            'final_yield_mean':st.mean(r['final'] for r in v),'initial_yield_mean':st.mean(r['initial'] for r in v),
            'added_query_yield_mean':st.mean(r['added_query_mean_yield'] for r in v),
            'improving_queries':sum(r['improving_added_queries'] for r in v),'added_queries':16*len(v),
            'mean_curve':[st.mean(r['curve'][j] for r in v) for j in range(16)]}


def compare(agent,control):
    delta=[metric(a)['AUC']-metric(b)['AUC'] for a,b in zip(agent,control)]
    return {'n':len(delta),'AUC_delta_mean_pp':st.mean(delta),'wins':sum(v>1e-10 for v in delta),
            'losses':sum(v<-1e-10 for v in delta),'ties':sum(abs(v)<=1e-10 for v in delta),
            'final_yield_delta_mean_pp':st.mean(metric(a)['final']-metric(b)['final'] for a,b in zip(agent,control)),
            'raw_pairs':[{'case':a['case'],'seed':a['seed'],'AUC_delta_pp':v} for a,v in zip(agent,delta)]}


def report():
    p=s.check_protocol();allrows=rows()
    assert all(len(group)==20 and all(r['status'] in ('COMPLETE','FAILED') for r in group) for group in allrows.values())
    assert s.read(s.OUT/'all_cards_frozen.json')['cards']==320
    tables={};paired={}
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4)),('Case1_reference',(1,)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        groups={m:[r for r in group if r['case'] in cases] for m,group in allrows.items()}
        tables[scope]={m:summary(group) for m,group in groups.items()}
        paired[scope]={m:compare(groups[METHODS[0]],groups[m]) for m in METHODS[1:]}
    raw=[{'method':m,'case':r['case'],'seed':r['seed'],'status':r['status']}|metric(r) for m,group in allrows.items() for r in group]
    result={'protocol_sha256':p['protocol_sha256'],'study':p['study'],'new_runs':40,'completed_new_runs':sum(r['status']=='COMPLETE' for m in METHODS[:2] for r in allrows[m]),
            'logical_queries':800,'adaptive_logical_queries':400,'frozen_control_logical_queries':400,'added_queries':640,
            'fresh_current_assistant_scorecards':320,'independent_LLM_API_calls':0,'adaptive_full_score_tables':320,
            'adaptive_candidate_scores':320*4692,'GP_fits':0,'operator_compiles_in_loop':0,
            'tables':tables,'adaptive_minus_control':paired,'raw_metrics':raw,
            'limits':['Assistant-authored restricted scorecards, not per-candidate independent LLM assessments.',
                      'Same chat has seen prior results; opaque task IDs are not blinding or isolated contexts.',
                      'No immutable model snapshot; current-assistant and scoring-grammar-specific exploratory result.',
                      'Five seeds per case; only three primary reaction environments; no statistical significance or broad superiority claim.',
                      'Frozen control fixes both first rule and initial observation geometry; difference is the complete feedback policy, not a pure outcome-only causal effect.',
                      'SearchAUC retains the incumbent and does not penalize all later low-yield trials; added-query mean and improvement count also reported.',
                      'No external regional advice, GP scores, hidden yields or prior quality labels in provider requests; shared context exposure remains a limitation.']}
    s.write(s.OUT/'results.json',result)
    with (s.OUT/'metrics.csv').open('w',encoding='utf8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['scope','method','n','SearchAUC_pp','final_yield_percent','added_query_yield_mean_percent','improving_queries'])
        for scope,table in tables.items():
            for method,r in table.items():writer.writerow([scope,method,r['n'],r['SearchAUC_mean'],r['final_yield_mean'],r['added_query_yield_mean'],r['improving_queries']])
    figures(result)
    return {'completed_new_runs':result['completed_new_runs'],'all_four':{m:{k:round(r[k],4) for k in ('SearchAUC_mean','final_yield_mean')} for m,r in tables['all_four'].items()},
            'primary':{m:{k:round(r[k],4) for k in ('SearchAUC_mean','final_yield_mean')} for m,r in tables['primary'].items()},
            'feedback_gain_primary':paired['primary'][METHODS[1]]['AUC_delta_mean_pp']}


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13,3.5),layout='constrained')
    labels=('Adaptive agent rule','Frozen initial rule','qLogEI BO','UCB original beta','UCB kappa=2')
    colors=('#0072B2','#777777','#D55E00','#CC79A7','#009E73')
    for c,ax in enumerate(axes,1):
        for m,label,color in zip(METHODS,labels,colors):
            r=result['tables'][f'Case{c}'][m];ax.plot(range(5,21),r['mean_curve'],label=label,color=color,lw=1.8,ls='--' if 'UCB' in m else '-')
        ax.set(title=f'Case {c}',xlabel='Total queries',xticks=[5,10,15,20]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean incumbent yield (%)');axes[-1].legend(fontsize=7,loc='lower right')
    fig.suptitle('Current-assistant scorecards: observed feedback, full fixed pool, 5 seeds per case',fontsize=11)
    fig.savefig(s.OUT/'incumbent_curves.png',dpi=180,bbox_inches='tight');fig.savefig(s.OUT/'incumbent_curves.pdf',bbox_inches='tight');plt.close(fig)


def tree(folder):
    files=[p for p in sorted(folder.rglob('*')) if p.is_file()];h=sha256()
    for f in files:h.update((f.relative_to(folder).as_posix()+'\0'+s.filehash(f)+'\n').encode('utf8'))
    return {'files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':h.hexdigest()}


def independent_scores(pool,history,card):
    """Separate algebra/order from the executing scorer, using coordinates only."""
    ix={key:i for i,key in enumerate(pool['ids'])};z=np.asarray(pool['z'])[:,:3]
    c=np.array([CATALYSTS.index(p['catalyst']) for p in pool['points']]);indices=[ix[r['candidate_id']] for r in history]
    observed=z[indices];hc=c[indices]
    output=np.zeros(len(z))
    for cat in range(8):
        mask=c==cat
        output[mask]=float(card['bias'][cat])
        if np.any(hc==cat):
            distances=np.linalg.norm(z[mask,None,:]-observed[None,hc==cat,:],axis=-1)/np.sqrt(3)
            output[mask]+=float(card['novelty'])*np.minimum(distances.min(axis=1),1.)
        else:output[mask]+=float(card['novelty'])
    for j in range(3):output-=float(card['curvature'][j])*np.square(z[:,j]-float(card['target'][j]))
    for index,amplitude,width in card['anchors']:
        i=index-1;distance=np.square(z-observed[i]).sum(axis=1)
        output+=float(amplitude)*np.exp(-distance/(2*float(width)*float(width)))*(c==hc[i])
    return output


def verify():
    p=s.check_protocol();allrows=rows();frozen=s.read(s.OUT/'all_cards_frozen.json');assert frozen['frozen_control_states_before_start']==0 and frozen['cards']==320
    for name,h in frozen['batches_sha256'].items():assert s.filehash(s.ROOT/'batches'/name)==h
    checks=0;score_values=0;frozen_checks=0;changed_rules=0;adapted_choices=0
    for row,control in zip(allrows[METHODS[0]],allrows[METHODS[1]]):
        case,seed=row['case'],row['seed'];folder=s.PARENT/'inputs'/f'case{case}';pool=s.read(folder/'pool.json');ids=pool['ids'];ix={key:i for i,key in enumerate(ids)}
        values=np.load(folder/'oracle.npz')['values'];assert row['status']==control['status']=='COMPLETE'
        assert s.filehash(s.ROOT/'runs'/row['run_id']/'state.json')==frozen['adaptive_states_sha256'][row['run_id']]
        assert row['history'][:4]==control['history'][:4] and [r['candidate_id'] for r in row['history'][:4]]==pool['initial_ids'][str(seed)]
        assert len(row['history'])==len(control['history'])==20 and len(row['decisions'])==len(control['decisions'])==16
        for r in (row,control):
            assert r['protocol_sha256']==p['protocol_sha256'] and len({v['candidate_id'] for v in r['history']})==20
            for v in r['history']:assert v['y']==float(values[ix[v['candidate_id']]])
        initial_card=row['decisions'][0]['card']
        fixed=independent_scores(pool,row['history'][:4],initial_card)
        for t,event in enumerate(row['decisions'],1):
            h=row['history'][:t+3];card=event['card'];context=event['context_sha256'];request=s.read(s.ROOT/'requests'/context/'request.json');response=s.read(s.ROOT/'requests'/context/'response.json')
            raw=dict(request);raw.pop('context_sha256');assert digest(raw)==context and request['history']==h
            assert request['iteration']==t and request['task']==row['task'] and request['protocol_sha256']==p['protocol_sha256']
            assert request['previous_card']==(row['decisions'][t-2]['card'] if t>1 else None)
            assert response['context_sha256']==context and response['card']==card and response['origin']=='current_assistant'
            assert response['source_batch_sha256']==s.filehash(s.ROOT/'batches'/f'round{t:02d}.json')
            assert all(1<=i<=len(h) for i in card['support']) and all(1<=v[0]<=len(h) for v in card['anchors'])
            assert len(request['observations'])==len(h)
            for observation,v in zip(request['observations'],h):assert observation['id']==v['candidate_id'] and observation['yield_percent']==v['y']*100
            score_key=digest({'context':context,'card':card});assert score_key==event['score_key']
            with np.load(s.ROOT/'scores'/(score_key+'.npz')) as d:scores=d['scores'].copy()
            recomputed=independent_scores(pool,h,card);assert len(scores)==event['candidates_scored']==4692
            assert np.allclose(scores,recomputed,atol=1e-10,rtol=1e-12);observed={v['candidate_id'] for v in h}
            eligible=[i for i,key in enumerate(ids) if key not in observed]
            j=min(eligible,key=lambda i:(-float(recomputed[i]),ids[i]))
            assert event['selected_id']==ids[j]==row['history'][t+3]['candidate_id'] and event['score']==float(scores[j])
            score_values+=len(scores);checks+=1
            # Posthoc same-prefix diagnostic controls for new observation geometry.
            old_rule=independent_scores(pool,h,initial_card);other=min(eligible,key=lambda i:(-float(old_rule[i]),ids[i]));adapted_choices+=int(other!=j)
            numeric_keys=('bias','target','curvature','novelty','anchors');changed_rules+=int(any(card[k]!=initial_card[k] for k in numeric_keys))
            fh=control['history'][:t+3];available=[i for i,key in enumerate(ids) if key not in {v['candidate_id'] for v in fh}]
            j=min(available,key=lambda i:(-float(fixed[i]),ids[i]));fe=control['decisions'][t-1]
            assert fe['selected_id']==ids[j]==control['history'][t+3]['candidate_id'] and abs(fe['score']-fixed[j])<1e-10;frozen_checks+=1
        assert row['decisions'][0]['selected_id']==control['decisions'][0]['selected_id']
    protected={};cache_trees={}
    for label,path in s.PROTECTED.items():
        inventory=s.read(path)
        for f in inventory['files']:assert s.filehash(f['path'])==f['sha256'],f['path']
        protected[label]=len(inventory['files'])
        for key,expected in inventory.items():
            if key.endswith('cache_tree'):
                directory=key.removesuffix('_tree');actual=tree(Path('data/runs')/f'suzuki_{label}'/directory)
                assert actual=={k:expected[k] for k in actual},(label,key);cache_trees[label+'_'+directory]=actual
    results=s.read(s.OUT/'results.json')
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4))]:
        for m,group in allrows.items():
            subset=[r for r in group if r['case'] in cases]
            auc=st.mean(sum(max(v['y'] for v in r['history'][:i])*100 for i in range(5,21))/16 for r in subset)
            assert abs(auc-results['tables'][scope][m]['SearchAUC_mean'])<1e-10
            assert all(r['history'][:4]==allrows[METHODS[0]][i]['history'][:4] for i,r in enumerate(group))
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'new_complete_runs':40,'logical_queries':800,
             'independent_adaptive_choices':checks,'independent_frozen_choices':frozen_checks,'independently_recomputed_candidate_scores':score_values,
             'cards_numerically_changed_from_initial':changed_rules,'choices_changed_from_initial_rule_at_same_actual_prefix':adapted_choices,
             'same_prefix_diagnostic':'Posthoc diagnostic only, not a new policy trajectory or separate performance score.',
             'GP_fits':0,'operator_compiles_in_loop':0,'protected_files_unchanged':protected,'protected_cache_trees_unchanged':cache_trees,
             'results_sha256':s.filehash(s.OUT/'results.json'),'boundary':'Integrity is distinct from scientific performance and shared-context exposure.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def release():
    p=s.check_protocol();receipt=s.read(s.OUT/'validation_receipt.json');assert receipt['integrity_status']=='PASS'
    suites=list(ET.parse(s.OUT/'component_tests.xml').getroot().iter('testsuite'));counts={k:sum(int(v.attrib.get(k,0)) for v in suites) for k in ('tests','failures','errors','skipped')}
    assert counts['tests']>=6 and counts['failures']==counts['errors']==0
    files=[Path(__file__),Path(s.__file__),Path('chem/benchmarks/agent_score_suzuki_engine.py'),Path('chem/benchmarks/suzuki_agent_score.py'),Path('tests/test_suzuki_agent_score.py'),Path('docs/suzuki_agent_score.md')]
    files += [f for f in s.ROOT.rglob('*') if f.is_file() and not any(part in ('pytest_tmp','scores') for part in f.relative_to(s.ROOT).parts)]
    files += [f for f in s.OUT.rglob('*') if f.is_file() and f.name not in ('release_inventory.json','release_receipt.json')]
    inventory={'protocol_sha256':p['protocol_sha256'],'file_count':len(set(files)),
               'files':[{'path':f.as_posix(),'bytes':f.stat().st_size,'sha256':s.filehash(f)} for f in sorted(set(files))],
               'score_tree':tree(s.ROOT/'scores'),'digest_definition':'Sorted native Path relative POSIX path + NUL + file SHA256 + LF.'}
    s.write(s.OUT/'release_inventory.json',inventory)
    output={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'new_trajectories':40,'logical_queries':800,
            'current_assistant_scorecards':320,'independent_LLM_API_calls':0,'component_tests':counts,'GP_fits':0,'operator_compiles_in_loop':0,
            'validation_receipt_sha256':s.filehash(s.OUT/'validation_receipt.json'),'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),
            'scientific_boundary':'Exploratory current-assistant restricted-scorecard policy; not independent pointwise LLM scoring or a blind general superiority claim.'}
    s.write(s.OUT/'release_receipt.json',output);return output
