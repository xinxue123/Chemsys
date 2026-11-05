"""Background-grounded independent LLM hypotheses on two paired trajectories."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from copy import deepcopy
import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
from filelock import FileLock
from chem.benchmarks import knowledge_gp as native
from kernel_contrast_qlogei import study as legacy
from kernel_contrast_revision import study as previous
from . import VARIANT,ARMS,MODEL,EFFORT,BACKEND,PARAMETERS as P
from . import background,hypotheses,experts,policy,prompts,transport

ROOT=legacy.ROOT;BASE=transport.BASE;REPORT=ROOT/'experiments/reports'/VARIANT
DESIGN=ROOT/'docs'/f'{VARIANT}.txt';SHORT=legacy.SHORT
JOBS=[(task,arm) for task in native.TASKS for arm in ARMS]
read,write,sha=legacy.read,legacy.write,legacy.sha


def check(protected=False):
    p=read(BASE/'protocol.json')
    assert p['protocol_sha256']==hypotheses.digest({k:v for k,v in p.items() if k!='protocol_sha256'})
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if protected else ()):
        for path,h in p[group].items():
            if sha(ROOT/path)!=h:raise ValueError('Frozen artifact changed: '+path)
    return p


def prepare():
    if (BASE/'protocol.json').exists():return check(True)
    if (BASE/'runs').exists() or (BASE/'requests').exists():raise ValueError('Register before new LLM or experiment requests')
    prior=previous.check(True)
    protected=dict(prior['protected_sha256']);protected.update(prior['sources_sha256'])
    protected.update(read(previous.REPORT/'participating_checkpoint_hashes.json'))
    for folder in (previous.REPORT,previous.BASE/'runs'):
        for path in folder.rglob('*'):
            if path.is_file() and (folder==previous.REPORT or path.suffix=='.json'):
                protected[path.relative_to(ROOT).as_posix()]=sha(path)
    source=list(Path(__file__).parent.glob('*.py'))+[DESIGN,ROOT/'code/experiment_tools/run_chemical_hypothesis_bo.py']
    sources={path.relative_to(ROOT).as_posix():sha(path) for path in source}
    sources['code/chemist_system/chem/benchmarks/knowledge_domain_context.py']=sha(ROOT/'code/chemist_system/chem/benchmarks/knowledge_domain_context.py')
    provenance={}
    for path in (ROOT/background.SOURCE).iterdir():
        if path.is_file():provenance[path.relative_to(ROOT).as_posix()]=sha(path)
    REPORT.mkdir(parents=True,exist_ok=True);cards=[]
    for task in native.TASKS:
        pool=native.load_task(ROOT,task,seed=0);card=background.context(pool)
        write(REPORT/'backgrounds'/f'{SHORT[task]}.json',card)
        cards.append(dict(task=task,background_sha256=hypotheses.digest(card),candidate_count=len(pool.ids),
            chemical_identities_decoded=pool.group!='P2',process_units_supplied=True))
    p=dict(variant=VARIANT,created_UTC=datetime.now(timezone.utc).isoformat(),jobs=JOBS,parameters=P,
        model=MODEL,effort=EFFORT,backend=BACKEND,fork_turns='none',initial=4,queries=16,seed=0,
        controls=prior['controls'],sources_sha256=sources,inputs_sha256=dict(prior['inputs_sha256'],**provenance),
        protected_sha256=protected,backgrounds=cards,source_registry=background.SOURCES,
        maximum_new_queries=256,maximum_logical_LLM_requests=32,shared_initial_hypotheses=True,
        native_GP_modified=False,development_not_held_out=True)
    p['protocol_sha256']=hypotheses.digest(p);write(BASE/'protocol.json',p)
    for path in source:
        target=BASE/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    return check(True)


def initialize():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.chdir(ROOT/'experiments/chemist_system');native.warm()


def folder_for(task,arm):return BASE/'runs'/SHORT[task]/arm


def save_arrays(folder,q,pool,context,values):
    stem=folder/'rounds'/f'{q:02d}';stem.parent.mkdir(parents=True,exist_ok=True)
    path=stem.with_name(stem.name+'_arrays.npz')
    if not path.exists():
        with path.open('xb') as f:np.savez_compressed(f,
            **{k:context['stats'][k] for k in ('mu_y','sd_y','sd_predictive_y')},
            **{k:context[k] for k in ('base_logei','available','eligible','tie_priority')},
            expert_mu=values['mu'],expert_logEI=values['scores'],mixture_logEI=values['mixture'],
            expert_ids=np.asarray(values['ids']),log_weights=np.asarray([values['log_weights'][i] for i in values['ids']]))
    csv=stem.with_name(stem.name+'_candidates.csv')
    if not csv.exists():
        table=dict(candidate_id=pool.ids,native_mean_raw=context['stats']['mu_y']*pool.scale,
            native_SD_raw=context['stats']['sd_y']*pool.scale,native_logEI=context['base_logei'],
            mixture_logEI=values['mixture'],available=context['available'],in_budget=context['eligible'])
        for k,i in enumerate(values['ids']):table[i+'_mean_raw']=values['mu'][k]*pool.scale
        pd.DataFrame(table).to_csv(csv,index=False,encoding='utf-8-sig')


def run_one(task,arm):
    p=check();pool=native.load_task(ROOT,task,seed=0);folder=folder_for(task,arm);folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.worker.lock'),timeout=1):
        initial=dict(task=task,arm=arm,initial_ids=list(pool.initial_ids),protocol_sha256=p['protocol_sha256'])
        initial['commit_sha256']=hypotheses.digest(initial);write(folder/'initial_commit.json',initial)
        sf=folder/'state.json'
        if sf.exists():state=read(sf);assert state['protocol_sha256']==p['protocol_sha256']
        else:
            state=dict(task=task,arm=arm,protocol_sha256=p['protocol_sha256'],status='RUNNING',
                history=[native.read_observation(ROOT,pool,k) for k in pool.initial_ids],notes=[],archive=[],
                decisions=[],calls=[],ledger=[],log_weights=dict(base=0.),initial_notes=None,initial_log_weights=None)
            write(sf,state)
        if state['status']=='COMPLETE':return dict(task=SHORT[task],arm=arm,status='COMPLETE')
        gp=native.GPAdapter(pool,BASE/'cache'/SHORT[task],seed=0)
        while len(state['decisions'])<16:
            q=len(state['decisions'])+1;cp=folder/'rounds'/f'{q:02d}_commit.json'
            if cp.exists():commit=read(cp)
            else:
                context=gp.context(state['history'],q);event=policy.refresh_event(pool,state,context,q)
                working=deepcopy(state);request=None
                if event['allowed']:
                    public=prompts.public(pool,state['history'],state['notes'],state['ledger'],event,credit=state['log_weights'])
                    request=transport.request(SHORT[task],'initial' if not state['calls'] else arm,q,public,
                        lambda raw:hypotheses.validate(raw,pool,state['history']))
                    notes=request['plan']['notes']
                    working['archive']+=deepcopy(state['notes']);working['notes']=notes
                    working['log_weights']=policy.initial(notes) if not state['calls'] else policy.edit_weights(state['log_weights'],notes)
                    working['calls'].append(q)
                    if not state['calls']:
                        working['initial_notes']=deepcopy(notes);working['initial_log_weights']=deepcopy(working['log_weights'])
                primary_hash=previous.models.state_hash(context['stats']['model'])
                values=experts.evaluate(gp,context,working['notes'],working['log_weights'])
                decision=policy.commit(pool,working,context,values,q)
                shadow=None
                if arm=='adaptive':
                    from scipy.special import logsumexp
                    prior=policy.initial(working['notes'])
                    shadow=dict(values,mixture=logsumexp(values['scores']+np.asarray([prior[i] for i in values['ids']])[:,None],axis=0))
                if shadow is not None:
                    selected_fixed=policy.choose(context,shadow,q)
                    decision['equal_prior_shadow_id']=pool.ids[selected_fixed]
                    decision['credit_changes_choice']=bool(q>1 and decision['selected_id']!=pool.ids[selected_fixed])
                else:decision['equal_prior_shadow_id']=None;decision['credit_changes_choice']=False
                assert previous.models.state_hash(context['stats']['model'])==primary_hash
                save_arrays(folder,q,pool,context,values)
                commit=dict(query=q,prestate_sha256=hypotheses.digest(state),decision=decision,
                    knowledge_state={k:working[k] for k in ('notes','archive','calls','log_weights','initial_notes','initial_log_weights')},
                    refresh=event,LLM_request=request['request'] if request else None,
                    primary_fit_key=context['stats']['fit_key'],native_state_sha256=primary_hash,counters=gp.counters)
                commit['commit_sha256']=hypotheses.digest(commit);write(cp,commit)
            assert commit['prestate_sha256']==hypotheses.digest(state)
            assert commit['commit_sha256']==hypotheses.digest({k:v for k,v in commit.items() if k!='commit_sha256'})
            d=commit['decision'];assert d['selected_id'] not in {h['candidate_id'] for h in state['history']}
            observation=native.read_observation(ROOT,pool,d['selected_id'])
            fb=policy.feedback(d,observation,pool.points[pool.ix[d['selected_id']]])
            write(folder/'rounds'/f'{q:02d}_feedback.json',dict(commit_sha256=commit['commit_sha256'],observation=observation,feedback=fb))
            before=max(h['raw_value'] for h in state['history'])
            state.update(deepcopy(commit['knowledge_state']));state['log_weights']=fb['log_weights_after']
            state['ledger']+=fb['ledger'];state['history'].append(observation)
            state['decisions'].append(dict(query=q,**observation,decision=d,incumbent_before=before,
                incumbent_after=max(before,observation['raw_value']),feedback=fb,refresh=commit['refresh']))
            state['status']='COMPLETE' if q==16 else 'RUNNING';write(sf,state,replace=True)
            print(json.dumps(dict(task=SHORT[task],arm=arm,query=q,changed=d['differs_from_BO'],
                value=observation['raw_value'],best=state['decisions'][-1]['incumbent_after'],LLM_calls=len(state['calls']))),flush=True)
            gp.release()
        return dict(task=SHORT[task],arm=arm,status='COMPLETE')


def report():
    p=check(True);REPORT.mkdir(parents=True,exist_ok=True);rows=[];rounds=[]
    for task in native.TASKS:
        bo=read(ROOT/p['controls'][task]['path']);b=legacy.summary(bo,'pure_BO',task);rows.append(b)
        for arm in ARMS:
            state=read(folder_for(task,arm)/'state.json');assert state['status']=='COMPLETE'
            assert [(h['candidate_id'],h['y']) for h in state['history'][:4]]==[(h['candidate_id'],h['y']) for h in bo['history'][:4]]
            result=legacy.summary(state,arm,task);result.update(delta_SearchAUC=result['SearchAUC']-b['SearchAUC'],
                interventions=sum(d['decision']['differs_from_BO'] for d in state['decisions']),LLM_calls=len(state['calls']),
                credit_changes_choice=sum(d['decision']['credit_changes_choice'] for d in state['decisions']))
            rows.append(result)
            for d in state['decisions']:
                packet=d['decision'];assert packet['native_logEI_gap']<=packet['epsilon']+1e-12
                if d['query']==16:assert not packet['differs_from_BO']
                rounds.append(dict(task=SHORT[task],arm=arm,query=d['query'],candidate_id=d['candidate_id'],
                    raw_value=d['raw_value'],best=d['incumbent_after'],changed=packet['differs_from_BO'],
                    weights=json.dumps(packet['log_weights_before']),native_replay_error=packet['native_replay_error'],
                    credit_changes_choice=packet['credit_changes_choice'],refresh=d['refresh']['allowed']))
        fixed=read(folder_for(task,'fixed')/'state.json');adaptive=read(folder_for(task,'adaptive')/'state.json')
        assert fixed['initial_notes']==adaptive['initial_notes']
        assert fixed['initial_log_weights']==adaptive['initial_log_weights']
        assert fixed['decisions'][0]['candidate_id']==adaptive['decisions'][0]['candidate_id']
    frame=pd.DataFrame(rows);frame.to_csv(REPORT/'results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rounds).to_csv(REPORT/'round_diagnostics.csv',index=False,encoding='utf-8-sig')
    text=['# Grounded chemical hypotheses and controlled mixture EI','',
        'Eight frozen tasks, seed0, identical four observations and sixteen queries. '
        'Fixed and adaptive arms share the same initial LLM response. The native GP and '
        'native acquisition budget remain unchanged. Hypothesis experts use bounded '
        'preferred-minus-comparison posterior mean shifts with unchanged covariance. '
        'Selection maximizes a weighted improvement mixture, not prediction disagreement.','',
        '| Task | BO SearchAUC | Fixed chemical knowledge | Adaptive chemical knowledge | Adaptive minus BO |',
        '|---|---:|---:|---:|---:|']
    for short in SHORT.values():
        f=frame[frame.task==short].set_index('method');a=f.loc['adaptive']
        text.append(f'| {short} | {f.loc["pure_BO","SearchAUC"]:.5f} | {f.loc["fixed","SearchAUC"]:.5f} | {a.SearchAUC:.5f} | {a.delta_SearchAUC:+.5f} |')
    text+=['','Task objectives have different units; do not average raw SearchAUC across tasks. '
        'The source setup/identity metadata is curated separately from all published outcomes. '
        'Public request packets contain physical identities and own observations, never full '
        'source articles, emulator kinetics or outcome tables. Private source hashes support '
        'audits. Unknown fixed details remain explicitly unknown.','',
        'The LLM backend is fresh independent gpt-6.1-terra medium agents, fork_turns=none. '
        'Actual spawn assignments and response receipts are retained. Token usage is unavailable. '
        'This is development evidence, not a stability claim or an unchanged official LGBO '
        'replication. Every result and abstention is retained.','',
        'Reproduce with run_chemical_hypothesis_bo.py prepare, preflight, run, pending, seal, '
        'and report. The runner waits for independently produced responses; it does not generate '
        'heuristic substitute responses or access credentials.']
    (REPORT/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(frame.to_string(index=False));return dict(status='REPORT_COMPLETE',rows=len(frame),decisions=len(rounds))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','run','pending','status','assign','seal','report'))
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--request');parser.add_argument('--receipt')
    args=parser.parse_args()
    if args.action=='prepare':p=prepare();value=dict(protocol_sha256=p['protocol_sha256'],jobs=len(JOBS))
    elif args.action=='preflight':
        from .validation import preflight
        value=preflight()
    elif args.action=='pending':value=transport.pending()
    elif args.action=='assign':transport.assign(args.request,read(args.receipt));value=dict(status='ASSIGNED')
    elif args.action=='seal':value=transport.seal(args.request)
    elif args.action=='report':value=report()
    elif args.action=='run':
        check(True)
        with ProcessPoolExecutor(max_workers=min(args.workers,len(JOBS)),initializer=initialize) as executor:
            futures=[executor.submit(run_one,*job) for job in JOBS]
            for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
        value=dict(status='COMPLETE')
    else:
        value=[dict(task=SHORT[t],arm=a,queries=len(read(folder_for(t,a)/'state.json')['decisions']),
            status=read(folder_for(t,a)/'state.json')['status']) for t,a in JOBS if (folder_for(t,a)/'state.json').exists()]
    print(json.dumps(value,ensure_ascii=True))


if __name__=='__main__':main()
