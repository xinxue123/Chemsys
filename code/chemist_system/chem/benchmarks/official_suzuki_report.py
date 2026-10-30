"""Full-population reporting for the pinned upstream Suzuki experiment."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from hashlib import sha256,sha1
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from chem.decision.replay import write_json_atomic
from . import official_suzuki_runner as run
from . import official_suzuki_adapter as adapter
from .suzuki_domain import digest,FrozenSuzukiOracle,same_point,NUMERIC


def run_metrics(s):
    ys=[h['y']*100 for h in s['history']]
    best=[];inc=0.
    for i in range(20):
        if i<len(ys): inc=max(inc,ys[i])
        best.append(inc)
    repeats=sum(any(same_point(h['point'],p['point']) for p in s['history'][:i])
                for i,h in enumerate(s['history']))
    return {'run_id':s['run_id'],'case':s['case'],'seed':s['seed'],'method':s['method'],
        'status':s['status'],'queries':len(ys),'additional_queries':max(0,len(ys)-4),
        'initial_best_pct':best[3],'final_best_pct':best[-1],
        'search_auc_pct':sum(best[4:])/16,'best_so_far_pct':best,
        'attained_80':best[-1]>=80,'attained_90':best[-1]>=90,
        'repeated_physical_queries':repeats,
        'accepted_llm':sum(bool(d.get('accepted_llm')) for d in s['decisions']),
        'native_branches':dict(Counter(d.get('native_branch',d.get('plan',{}).get('mode','')) for d in s['decisions'] if s['method']=='LGBO')),
        'operator_compiles_in_loop':s.get('operator_compiles_in_loop',0),
        'upstream_numerics':s.get('upstream_numerics',{}),
        'optimization_seconds':s['optimization_seconds']}


def aggregate(rows):
    return {'n':len(rows),'completed':sum(r['status']=='COMPLETED' for r in rows),
        **{key:{'mean':statistics.mean(r[key] for r in rows),
                 'sd':statistics.stdev(r[key] for r in rows) if len(rows)>1 else 0.}
           for key in ('search_auc_pct','final_best_pct','initial_best_pct')},
        **{key:sum(r[key] for r in rows) for key in ('queries','additional_queries','repeated_physical_queries','accepted_llm','operator_compiles_in_loop')},
        'attainment_80':sum(r['attained_80'] for r in rows),
        'attainment_90':sum(r['attained_90'] for r in rows),
        'abandoned_fit_epochs':sum(r['upstream_numerics'].get('abandoned_fit_epochs',0) for r in rows),
        'random_prediction_fallbacks':sum(r['upstream_numerics'].get('random_prediction_fallbacks',0) for r in rows)}


def validate_records():
    protocol=run.check_protocol();states=[];contexts=[];oracle_checks=[];shared={}
    valid_protocols={protocol['protocol_sha256'],protocol.get('transport_amendment',{}).get('previous_protocol_sha256')}
    for spec in run.read(run.ROOT/'manifest.json'):
        s=run.read(run.ROOT/'runs'/spec['run_id']/'state.json');states.append(s)
        assert s['status'] in ('COMPLETED','FAILED'),(s['run_id'],s['status'])
        assert s['protocol_sha256']==protocol['protocol_sha256']
        assert len(s['history'])<=20 and s['oracle_queries']==len(s['history'])
        assert len(s['decisions'])==len(s['history'])-4
        assert [h['point'] for h in s['history'][:4]]==spec['initial_points']
        assert all(h['query']==i+1 and h['initial']==(i<4) for i,h in enumerate(s['history']))
        for h in s['history']:
            assert same_point(adapter.decode_z(h['z']),h['point'])
        key=(s['case'],s['seed']);initial=s['history'][:4]
        if key in shared: assert initial==shared[key]
        else: shared[key]=initial
        oracle=FrozenSuzukiOracle(run.ORACLES,s['case'])
        y=oracle.predict([h['point'] for h in s['history']])
        err=max(abs(float(a)-h['y'])*100 for a,h in zip(y,s['history']))
        assert err<1e-4
        oracle_checks.append({'run_id':s['run_id'],'queries':len(y),'max_error_percentage_points':err})
        if s['method']=='HEBO': assert not s['responses']
        else:
            assert len(s['responses'])==len(s['decisions'])
            for i,a in enumerate(s['responses']):
                q=run.read(run.ROOT/'requests'/a['context_sha256']/'request.json')
                raw=deepcopy(q);h=raw.pop('context_sha256');assert digest(raw)==h
                assert q['protocol_sha256'] in valid_protocols
                assert q['history']==s['history'][:i+4] and q['iteration']==i+1
                assert q['previous_response']==(s['responses'][i-1] if i else None)
                assert run.read(run.ROOT/'requests'/h/'response.json')==a
                run.validate_answer(a,q)
                if s['method']=='LGBO':
                    sys,user=adapter.lgbo_prompts(s['case'],q['history'])
                    assert q['system_prompt']==sys and q['user_prompt']==user
                contexts.append(h)
    assert len(states)==60 and len(shared)==20 and len(set(contexts))==len(contexts)
    # Verify the exact human-readable assistant transcript -> source transport.
    from scripts.suzuki_official_assistant_io import text_for
    consumed={a['context_sha256']:a for s in states for a in s['responses']}
    transcribed=[]
    for f in sorted(run.RESEARCH.glob('assistant_round[0-9][0-9].json')):
        raw=run.read(f);expanded=run.read(f.with_suffix('.expanded.json'))
        assert raw['origin'] in ('actual current assistant','actual current assistant; no algorithmic proposer')
        for a in expanded['responses']:
            q=run.read(run.ROOT/'requests'/a['context_sha256']/'request.json')
            manual=raw['answers'][a['run_id']]
            assert q['iteration']==raw['round']
            assert text_for(manual,q['method'],q['previous_response'])==a['upstream_reply']
            assert manual['reasoning']==a['reasoning'] and manual['supporting_queries']==a['supporting_queries']
            assert consumed[a['context_sha256']]==a
            transcribed.append(a['context_sha256'])
    assert Counter(transcribed)==Counter(contexts)
    # The original before-amendment protocol remains independently verifiable.
    original=run.read(run.RESEARCH/'amendment_01/protocol_before.json')
    unsigned=deepcopy(original);h=unsigned.pop('protocol_sha256');assert digest(unsigned)==h
    for name,h in protocol['oracles_sha256'].items(): assert sha256(Path(name).read_bytes()).hexdigest()==h
    return states,{'status':'PASS','planned_runs':60,'validated_states':len(states),
        'common_initial_designs':len(shared),'logical_assistant_responses':len(contexts),
        'unique_contexts':len(set(contexts)),'manual_transcriptions':len(transcribed),
        'oracle_checks':oracle_checks,'source_files_identical':len(adapter.check_sources()['files']),
        'protocol_sha256':protocol['protocol_sha256'],'amendment':protocol['transport_amendment']}


def make_figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'LLINBO':'#1764ab','LGBO':'#d5671c','HEBO':'#3b8559'}
    fig,axes=plt.subplots(2,2,figsize=(10,7.2),sharex=True,sharey=True)
    for case,ax in zip((1,2,3,4),axes.ravel()):
        for method in run.METHODS:
            a=np.array([r['best_so_far_pct'] for r in rows if r['case']==case and r['method']==method])
            mean=a.mean(0);sd=a.std(0,ddof=1);x=np.arange(1,21)
            ax.plot(x,mean,label=method,color=colors[method],linewidth=1.8)
            ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=colors[method],alpha=.13)
        ax.axvline(4.5,color='#777777',linestyle=':',linewidth=1)
        ax.set_title(f'Reizman Suzuki Case {case}');ax.set_xlim(1,20);ax.set_ylim(0,100)
        ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.15)
    axes[1,0].set_xlabel('Charged simulator queries');axes[1,1].set_xlabel('Charged simulator queries')
    axes[0,0].set_ylabel('Best observed yield (%)');axes[1,0].set_ylabel('Best observed yield (%)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.96),frameon=False)
    fig.suptitle('Pinned official source implementations; matched 4 + 16 query budget',y=.995,fontsize=12)
    fig.text(.5,.025,'5 seeds; bands = SD. Current assistant in a shared chat; continuous catalyst-code interface for LLINBO/LGBO.\nNative fallback branches and repeated queries are retained. Historical emulator, not laboratory experiments.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.085,1,.93))
    for ext in ('png','pdf','svg'):fig.savefig(run.RESEARCH/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)


def report(replay_official=False):
    states,checks=validate_records();rows=[run_metrics(s) for s in states]
    if replay_official:
        from scripts.replay_suzuki_official import main as replay
        replay()
    result={'study':'Suzuki_official_sources_v1','protocol_sha256':checks['protocol_sha256'],
        'primary_metric':'mean incumbent yield over the 16 added-query positions, percentage points',
        'overall':{m:aggregate([r for r in rows if r['method']==m]) for m in run.METHODS},
        'by_case':{str(c):{m:aggregate([r for r in rows if r['case']==c and r['method']==m]) for m in run.METHODS} for c in (1,2,3,4)},
        'runs':rows,'validation':checks,
        'interpretation_limits':['All 60 planned trajectories retained, including failures if any.',
            'Current assistant shared-chat evaluation; not independent or blind model calls.',
            'Continuous interval category encoding is necessary for the continuous-only upstream interfaces and is label-order dependent.',
            'Algorithms retain different native acquisitions and representations; this comparison does not isolate causal benefit of chemical knowledge.',
            'Pinned compatible environment; not a reproduction of the original papers numerical tables.',
            'No global continuous optimum or global-regret assertion.']}
    write_json_atomic(run.RESEARCH/'results.json',result)
    write_json_atomic(run.RESEARCH/'record_validation.json',checks)
    keys=['run_id','case','seed','method','status','queries','additional_queries','initial_best_pct','search_auc_pct','final_best_pct','attained_80','attained_90','repeated_physical_queries','accepted_llm','operator_compiles_in_loop','optimization_seconds']
    with (run.RESEARCH/'run_metrics.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
    make_figures(rows)
    return {'status':'PASS','overall':result['overall'],'by_case':result['by_case']}


def verify():
    states,records=validate_records()
    saved=run.read(run.RESEARCH/'results.json')
    assert saved['runs']==[run_metrics(s) for s in states]
    replay=run.read(run.RESEARCH/'native_replay.json');assert replay['status']=='PASS'
    assert replay['protocol_sha256']==records['protocol_sha256']
    assert replay['replayed_runs']==60 and replay['replayed_additional_decisions']==960
    for row in replay['rows']:
        state=next(s for s in states if s['run_id']==row['run_id'])
        assert row['state_sha256']==digest(state) and row['checked_decisions']==16
    independent=run.read(run.RESEARCH/'independent_metrics.json');assert independent['status']=='PASS'
    assert independent['results_sha256']==sha256((run.RESEARCH/'results.json').read_bytes()).hexdigest()
    blobs=[]
    for row in adapter.check_sources()['files']:
        raw=Path(row['path']).read_bytes()
        blob=sha1(b'blob '+str(len(raw)).encode('ascii')+b'\0'+raw).hexdigest()
        assert blob==row['git_blob_sha']
        blobs.append({'path':row['path'],'git_blob_sha':blob,'matches_official':True})
    from importlib import metadata
    dist=metadata.distribution('HEBO')
    for path,row in run.read(Path('research/suzuki_baselines/hebo_expected_sources.json')).items():
        raw=Path(dist.locate_file(path)).read_bytes()
        blob=sha1(b'blob '+str(len(raw)).encode('ascii')+b'\0'+raw).hexdigest()
        assert blob==row['upstream_git_blob_sha']
        blobs.append({'path':path,'git_blob_sha':blob,'matches_official':True})
    from xml.etree import ElementTree
    tests=[]
    for name in ('official_unit_tests.xml','original_env_regression.xml'):
        path=run.ROOT/name
        suite=ElementTree.parse(path).find('testsuite')
        assert suite.attrib['errors']==suite.attrib['failures']=='0'
        tests.append({'file':str(path),'tests':int(suite.attrib['tests']),
                      'sha256':sha256(path.read_bytes()).hexdigest()})
    receipt={'status':'PASS','records':records,'native_replay':replay,
        'independent_metrics':independent,'actual_queries':sum(s['oracle_queries'] for s in states),
        'completed_runs':sum(s['status']=='COMPLETED' for s in states),
        'operator_compiles_in_loop':sum(s.get('operator_compiles_in_loop',0) for s in states),
        'shared_assistant_context':True,'original_source_files_modified':False,
        'official_git_blob_checks':blobs,'tests':tests}
    receipt['native_execution_counts']={
        'LLINBO_accepted_LLM':sum(d['accepted_llm'] for s in states if s['method']=='LLINBO' for d in s['decisions']),
        'LLINBO_native_GP_UCB':sum(not d['accepted_llm'] for s in states if s['method']=='LLINBO' for d in s['decisions']),
        'LGBO_branches':dict(Counter(d['plan']['mode'] for s in states if s['method']=='LGBO' for d in s['decisions'])),
        'HEBO_affected_runs':[s['run_id'] for s in states if s['method']=='HEBO' and s['upstream_numerics']['random_prediction_fallbacks']],
        'HEBO_abandoned_fit_epochs':sum(s['upstream_numerics']['abandoned_fit_epochs'] for s in states if s['method']=='HEBO'),
        'HEBO_random_prediction_fallbacks':sum(s['upstream_numerics']['random_prediction_fallbacks'] for s in states if s['method']=='HEBO')}
    qa=run.read(run.RESEARCH/'figure_qa.json')
    assert qa['visual_status']=='PASS'
    for artifact in qa['artifacts']:
        assert sha256(Path(artifact['path']).read_bytes()).hexdigest()==artifact['sha256']
    receipt['figure_qa']=qa
    write_json_atomic(run.RESEARCH/'validation_receipt.json',receipt)
    return receipt
