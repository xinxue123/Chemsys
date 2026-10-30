from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import debug_suzuki_case as debug
from scripts import suzuki_debug_view as view


def test_candidate_export_preserves_budget_rank_units_and_soft_region():
    pool = {'ids':[f'p{i:05}' for i in range(4)], 'z':[[.1,.2,.3,i] for i in range(4)],
            'points':[{'catalyst':c} for c in debug.engine.CATALYSTS[:4]]}
    state = {'history':[{'candidate_id':'p00000','y':.2}],
             'knowledge':{'confidence':.7,'clauses':[{'id':'H1','coordinate':3,'bounds':[],'categories':[0]}]}}
    decision = {'available':np.array([False,True,True,True]),'eligible':np.array([False,False,True,True]),
                'base':np.array([4.,1.,3.,3.]),'score':np.array([4.,9.,7.,7.]),'selected':2,'reference':2,
                'budget':0.,'retained':['H1'],'amplitude':.5,
                'raw_shift_std':np.full(4,.5),'shift_std':np.full(4,.25),
                'stats':{'mu_y':np.full(4,.2),'sd_y':np.full(4,.01),'sd_predictive_y':np.full(4,.02)}}
    cache = SimpleNamespace(field=lambda *args:(np.ones(4),None,None), model=lambda *args:('key',{'std':.1}))
    table, _ = debug.candidate_view(cache,state,pool,decision)
    assert table.reason.tolist()==['already_queried','outside_base_qLogEI_budget','selected','eligible_tie_larger_candidate_id']
    assert table.guided_rank_available.tolist()==[0,1,2,3]
    assert table.guided_rank_eligible.tolist()==[0,0,1,2]
    assert table[2].mean_after_region_percent==pytest.approx(25.)
    assert table[2].mean_after_KF_percent==pytest.approx(22.5)
    assert not table[2].matches_retained_region  # soft injection permits this
    assert not any('oracle' in name or 'observed' in name for name in table.dtype.names)


@pytest.fixture
def archived_refresh(tmp_path):
    knowledge={'confidence':.7,'clauses':[{'id':'H1','coordinate':0,'bounds':[.2,.8],'categories':[]}]}
    state={'knowledge':knowledge,'slots':{'H1':0},'history':[{'candidate_id':'p00000','y':.2}]}
    pool={'ids':['p00000','p00001'],'z':[[.1,.2,.3,0],[.5,.2,.3,1]],
          'points':[{'catalyst':c} for c in debug.engine.CATALYSTS[:2]]}
    public={'public':{'observed':state['history'],'catalysts':{0:'A',1:'B'},
                      'diagnosis_tuples':[('H1','H2')]},'history':state['history'],'protocol_sha256':'test'}
    context={**public,'context_sha256':debug.digest(public)}
    source=tmp_path/'archive'/'requests'/context['context_sha256']
    plan={'actions':[{'id':'H1','operation':'retain','bounds':[],'categories':[],
                      'reason':'Preserve the observed-prefix preference','support':[1]}],'summary':'Retain'}
    response={'choices':[{'message':{'content':debug.json.dumps(plan)}}]}
    response_path=source/'attempt01'/'response.json'
    debug.write_json(source/'context.json',context)
    debug.write_json(response_path,response)
    valid={'context_sha256':context['context_sha256'],'accepted_attempt':1,
           'response_sha256':debug.study.filehash(response_path),
           'plan':debug.engine.validate_refresh(plan,knowledge,state['slots'],1,pool)}
    debug.write_json(source/'validated.json',valid)
    return tmp_path,source,context,state,pool,valid


def test_replay_revalidates_exact_context_without_network(archived_refresh):
    root,source,context,state,pool,valid=archived_refresh
    calls=[]
    def original(client,ctx,st,p,folder):
        assert client is None and debug.study.read(folder/'context.json')==debug.json.loads(debug.json.dumps(context))
        calls.append(True)
        return debug.study.read(folder/'validated.json')
    answer=debug.replay_refresh(original,root/'archive',None,context,state,pool,root/'output')
    assert answer==valid and calls==[True]


def test_replay_rejects_changed_context_without_fallback(archived_refresh):
    root,source,context,state,pool,valid=archived_refresh
    changed=deepcopy(context);changed['public']['extra_unmatched_field']=1
    def forbidden(*args):raise AssertionError('must not call transport')
    with pytest.raises(debug.ReplayMismatch,match='exactly match'):
        debug.replay_refresh(forbidden,root/'archive',None,changed,state,pool,root/'output')
    assert not (root/'output').exists()


def test_replay_rejects_response_tampering(archived_refresh):
    root,source,context,state,pool,valid=archived_refresh
    debug.write_json(source/'attempt01'/'response.json',{'choices':[]})
    with pytest.raises(debug.ReplayMismatch,match='hash mismatch'):
        debug.replay_refresh(None,root/'archive',None,context,state,pool,root/'output')


def test_output_never_reuses_existing_or_scientific_paths(tmp_path):
    with pytest.raises(FileExistsError):debug.output_directory(str(tmp_path),2)
    with pytest.raises(ValueError):debug.output_directory(str(debug.REPO/'data'/'runs'/'new_debug'),2)


def test_default_covers_all_16_trajectories_and_full_schedule():
    args=debug.arguments([])
    assert args.case==2 and args.regimes==list(debug.study.REGIMES) and args.methods==list(debug.engine.METHODS)
    assert args.rounds==16 and args.llm=='replay' and not args.recompute
    args=debug.arguments(['--rounds','2'])
    assert debug.engine.common.regional.budget_at(4,args.rounds)==pytest.approx(np.log(4)*14/15)


def test_truth_annotation_is_display_only_and_preserves_input():
    table=np.rec.fromarrays([['p00000','p00001'],[True,True],[True,True],[20.,60.],[1.,2.],[2,1],[2,1]],
        names=['candidate_id','available','eligible','GP_mean_percent','region_unscaled_qLogEI','base_rank_available','guided_rank_available'])
    before=table.copy()
    pool={'points':[{'catalyst':'A','t_res':60.,'temperature':30.,'catalyst_loading':.5},
                    {'catalyst':'B','t_res':600.,'temperature':110.,'catalyst_loading':2.5}]}
    oracle=np.array([.95,.05]);original=oracle.copy()
    display=view.annotate(table,pool,oracle)
    assert np.array_equal(table,before) and np.array_equal(oracle,original)
    assert view.TRUTH not in table.dtype.names
    assert display[view.TRUTH].tolist()==[95.,5.]
    assert display.GP_mean_rank_available.tolist()==[2,1]  # oracle ranking cannot replace GP ranking


def test_filter_inspection_matches_native_joseph_update_without_mutating_inputs():
    bank=debug.engine.new_bank(1);prior=debug.engine.forecast(bank,1)
    bank_copy,prior_copy=deepcopy(bank),deepcopy(prior)
    info=view.filter_preview(bank,prior,1,.1,.2,.02,.25)
    after,m=debug.engine.update_bank(bank,prior,1,.1,.2,.02,.25)
    assert bank==bank_copy and prior==prior_copy
    assert info['deployed_alpha_this_round']==1.
    assert np.allclose(info['theta_after_preview'],after['mean'],rtol=0,atol=1e-14)
    assert np.allclose(info['P_after_preview'],after['covariance'],rtol=0,atol=1e-14)
    assert info['innovation']==pytest.approx(m['innovation'])


def test_staged_csv_keeps_true_yield_as_explicit_debug_column(tmp_path):
    import csv
    table=np.rec.fromarrays([['p00000','p00001'],[95.,5.],[20.,60.]],
                           names=['candidate_id',view.TRUTH,'GP_mean_percent'])
    view.csv_table(tmp_path/'gp.csv',table,list(table.dtype.names))
    rows=list(csv.DictReader((tmp_path/'gp.csv').open(encoding='utf-8-sig')))
    assert len(rows)==2 and float(rows[0][view.TRUTH])==95. and float(rows[0]['GP_mean_percent'])==20.


def test_actual_prompt_and_response_saved_without_console_dump(archived_refresh,capsys):
    root,source,context,state,pool,valid=archived_refresh
    options=SimpleNamespace(no_stage_csv=False,top=0,llm='replay',print_llm=False)
    book=view.DebugBook(root,options,np.array([.1,.2]),pool)
    state.update(case=2,regime='assistant_initial',method='G3_gate_KF_B4_event',run_id='test',decisions=[],initial_knowledge=state['knowledge'])
    book.begin(state)
    body={'model':'deepseek-flash','messages':[{'role':'system','content':'EXACT_SYSTEM_MARKER'},
                                            {'role':'user','content':'EXACT_OWN_OBSERVATIONS_MARKER'}]}
    debug.write_json(source/'attempt01'/'request.json',body)
    book.llm_start({'public':{'reasons':['stagnation']},'context_sha256':context['context_sha256']},body['messages'])
    assert (book.directory/'LLM_request_prompt.txt').exists()
    assert book.current['llm']['attempts']==[]
    book.llm_finish(source,valid=valid)
    prompt=(book.directory/'LLM_attempt01_prompt.txt').read_text(encoding='utf8')
    assert 'EXACT_SYSTEM_MARKER' in prompt and 'EXACT_OWN_OBSERVATIONS_MARKER' in prompt
    assert book.current['llm']['attempts'][0]['messages']==body['messages']
    assert 'EXACT_SYSTEM_MARKER' not in capsys.readouterr().out
    assert (book.directory/'LLM_attempt01_response.txt').exists()


def test_pause_modes_support_every_round_and_optional_round_filter():
    options=debug.arguments(['--pause','kf','round'])
    assert debug.pause_requested(options,'kf',1) and debug.pause_requested(options,'round',16)
    assert not debug.pause_requested(options,'stages',1)
    options.pdb_round=2
    assert debug.pause_requested(options,'kf',2) and not debug.pause_requested(options,'kf',1)
