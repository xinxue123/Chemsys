"""Post-run custody and trust-state replay; never used by the optimizer."""
from copy import deepcopy
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from chem.benchmarks import stable_suzuki_study as s,stable_suzuki_engine as e,stable_suzuki_policy as policy
from chem.benchmarks.suzuki_domain import digest


def jnorm(value):return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))


def main():
    protocol=s.check();rows=[s.read(p) for p in sorted((s.ROOT/'runs').glob('*/state.json'))]
    assert len(rows)==16 and all(st['status']=='COMPLETE' for st in rows)
    choices=trust_updates=contexts=responses=controlled=0
    for state in rows:
        case=state['case'];pool=s.prior.inputs(case);lookup={k:i for i,k in enumerate(pool['ids'])}
        with np.load(s.prior.original.ROOT/'inputs'/f'case{case}'/'oracle.npz') as o:values=o['values'].copy()
        view={'history':deepcopy(state['history'][:4]),'knowledge':deepcopy(state['initial_knowledge']),
            'region_trust':{},'banks':{}}
        refresh={c['iteration']:c for c in state['calls']}
        checks=[(call['context_sha256'],'refresh',call) for call in state['calls']]
        for t,event in enumerate(state['decisions'],1):
            folder=s.ROOT/'runs'/state['run_id']/'rounds';commit=s.read(folder/f'{t:02d}_committed.json')
            with np.load(folder/f'{t:02d}_candidates.npz') as f:d={k:f[k].copy() for k in f.files}
            observed=state['history'][t+3];j=lookup[event['selected_id']]
            assert observed=={'candidate_id':event['selected_id'],'y':event['y']}
            assert observed['y']==float(values[j])
            assert commit['history']==view['history']
            available=d['available'];budget=commit['budget'];base=d['base_qLogEI']
            def permission():
                dg=e.original.diagnose(base,pool,view['knowledge'],budget,~available)
                region=e.canonical_region(view['knowledge'],list(dg.retained_ids));key=e.trust.geometry_key(region)
                bank=deepcopy(view['banks'].get(key,e.original.new_bank(t))) if key else None
                prior=e.original.forecast(bank,t) if bank else None
                conflict=bool(dg.revoked_ids) or dg.conflict_kind!='none'
                alpha=0. if conflict else prior['amplitude'] if prior else 1.
                trial={'region':region,'fallback':conflict,'amplitude':alpha}
                gate=e.trust.permission(view,trial,t)
                return gate,bank,prior
            gate,bank,prior=permission()
            if t in refresh:
                assert refresh[t]['knowledge_before']==view['knowledge']
                view['knowledge']=deepcopy(refresh[t]['validated']['plan']['knowledge'])
                gate,bank,prior=permission()
            assert jnorm(gate)==commit['trust']==event['trust']
            assert view['knowledge']==commit['knowledge']
            assert jnorm(view['region_trust'])==commit['region_trust_at_commit']
            assert bank==commit['bank_before'] and prior==commit['bank_prior']
            if bank:
                elapsed=t-bank['last_round'];forecast=np.array(bank['covariance'])+np.diag([.000025,.0025])*elapsed
                assert np.array_equal(forecast,np.array(prior['covariance']))
                assert prior['mean']==bank['mean']
                assert prior['amplitude']==float(np.clip(bank['mean'][1],0,1))
                raw=float(d['raw_shift_std'][j]*commit['scale_y'])
                after,measurement=e.original.update_bank(bank,prior,t,raw,float(d['mu_y'][j]),float(d['sd_predictive_y'][j]),observed['y'])
                assert after==event['bank_after']
                key=gate['geometry_key'];view['banks'][key]=after
                trial={'trust':gate,'selected':j,'stats':{'mu_y':d['mu_y']},'prior':prior,'_measurement':measurement}
                feedback=e.feedback(view,trial,t,observed['y'],after)
                assert jnorm(feedback)==event['trust_feedback']
                trust_updates+=1
            assert jnorm(view['region_trust'])==event['region_trust_after_feedback']
            assert abs(event['deployed_shift_y']-event['amplitude']*event['raw_shift_y'])<1e-12
            if commit['validation']:
                val=commit['validation'];checks.append((val['context_sha256'],'controlled_validation',val));controlled+=1
                anchor=view['history'][val['validated']['plan']['anchor_observation_index']-1]
                a=lookup[anchor['candidate_id']]
                assert int(np.sum(np.asarray(pool['z'][a])!=np.asarray(pool['z'][j])))==1
            view['history'].append(observed);choices+=1
        assert view['history']==state['history'] and view['banks']==state['banks']
        for key,phase,owner in checks:
            folder=s.ROOT/'requests'/key;context=s.read(folder/'context.json');answer=s.read(folder/'validated.json')
            raw_context=deepcopy(context);assert raw_context.pop('context_sha256')==key==digest(raw_context)
            if phase=='refresh':t=owner['iteration'];observations=context['public']['observations']
            else:t=context['public']['iteration'];observations=context['public']['observations']
            assert context['observed_prefix']==state['history'][:t+3]
            assert context['public']['task']==state['task'] and phase==context['phase']
            for i,(observed,history) in enumerate(zip(observations,context['observed_prefix']),1):
                assert observed['index']==i and observed['yield_percent']==round(history['y']*100,3)
                features=s.grounded.e.named_features(pool,lookup[history['candidate_id']])
                assert all(observed[k]==v for k,v in features.items())
            assert len(observations)==len(context['observed_prefix'])
            def validate(raw):
                if phase=='refresh':return s.grounded.e.validate_refresh(raw,owner['knowledge_before'],state['slots'],observations,pool)
                return policy.validate(raw,owner['prepared'],context['observed_prefix'])
            first_valid=None
            for attempt in sorted(folder.glob('attempt*')):
                if not (attempt/'response.json').exists():continue
                response=s.read(attempt/'response.json');body=s.read(attempt/'request.json');responses+=1
                assert json.loads(body['messages'][1]['content'])==context['public']
                assert digest(body['messages'][0]['content'])==context['system_sha256']
                assert {k:v for k,v in body.items() if k!='messages'}==protocol['request_parameters']
                if response.get('model')!='deepseek-flash' or response['choices'][0]['finish_reason']!='stop':continue
                try:raw=json.loads(response['choices'][0]['message']['content']);plan=validate(raw)
                except (ValueError,TypeError,KeyError,IndexError):continue
                if first_valid is None:first_valid=(int(attempt.name.removeprefix('attempt')),raw,jnorm(plan),s.filehash(attempt/'response.json'))
            assert first_valid is not None
            assert first_valid==(answer['accepted_attempt'],answer['raw'],answer['plan'],answer['response_sha256'])
            contexts+=1
    receipt={'status':'PASS','choices_and_oracle_labels_checked':choices,'trust_state_and_KF_bank_transitions_replayed':trust_updates,
        'own_prefix_LLM_contexts_checked':contexts,'received_responses_checked':responses,'exact_single_axis_tests_checked':controlled,
        'protocol_sha256':protocol['protocol_sha256'],'audit_script_sha256':s.filehash(__file__),
        'boundary':'post-run technical and custody audit; not an independent algorithm, performance tuning or generalization test'}
    s.write(s.OUT/'evidence_audit.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
