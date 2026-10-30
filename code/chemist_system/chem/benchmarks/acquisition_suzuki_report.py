"""Evaluation, fixed-prefix diagnostics and independent receipts. No policy tuning."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import statistics as st
import time
import xml.etree.ElementTree as ET

import numpy as np

from . import acquisition_suzuki_study as s
from . import acquisition_suzuki_engine as e
from . import guarded_suzuki_engine as common
from .llinbo_guarded_report import metric,baseline
from .suzuki_domain import digest


def new_rows():
    return [s.read(f) for stage in ('development','main') for f in sorted((s.ROOT/stage/'runs').glob('*/state.json'))]


def main_rows():
    k=s.read(s.OUT/'selected_kappa.json')['kappa']
    return [s.read(s.state_path(c,seed,g,mode,k if mode=='fixed' else None))
            for c in (1,2,3,4) for seed in range(5) for g in ('none',*common.REGIMES) for mode in ('official','fixed')]


def stats(rows):
    values=[metric(r) for r in rows]
    return {'n':len(rows),'complete':sum(r['status']=='COMPLETE' for r in rows),
            'AUC_mean':st.mean(v['auc'] for v in values),'AUC_sd':st.stdev(v['auc'] for v in values),
            'final_mean':st.mean(v['final'] for v in values),'curve':[st.mean(v['curve'][i] for v in values) for i in range(16)],
            'by_case':{str(c):{'AUC_mean':st.mean(metric(r)['auc'] for r in rows if r['case']==c),
                              'final_mean':st.mean(metric(r)['final'] for r in rows if r['case']==c)} for c in sorted({r['case'] for r in rows})}}


def paired(a,b):
    delta=[metric(x)['auc']-metric(y)['auc'] for x,y in zip(a,b)]
    return {'pairs':len(delta),'mean_AUC_delta_pp':st.mean(delta),'wins':sum(v>1e-10 for v in delta),
            'losses':sum(v<-1e-10 for v in delta),'ties':sum(abs(v)<=1e-10 for v in delta),
            'by_case':{str(c):st.mean(v for r,v in zip(a,delta) if r['case']==c) for c in sorted({r['case'] for r in a})},
            'raw':[{'case':r['case'],'seed':r['seed'],'delta_pp':v} for r,v in zip(a,delta)]}


def report():
    p=s.check_protocol();rows=main_rows();assert len(rows)==200 and all(r['status'] in ('COMPLETE','FAILED') for r in rows)
    tables={};pairs={};knowledge={}
    for scope,cases in [('primary',(2,3,4)),('Case1_reference',(1,)),('all_four_exploratory',(1,2,3,4))]:
        tables[scope]={};pairs[scope]={};knowledge[scope]={}
        for g in ('none',*common.REGIMES):
            method='B0' if g=='none' else 'LGBO_region'
            groups={'qLogEI':[baseline(c,seed,g,method) for c in cases for seed in range(5)]}
            groups.update({f'UCB_{mode}':[r for r in rows if r['case'] in cases and r['regime']==g and r['mode']==mode] for mode in ('official','fixed')})
            for key in groups:groups[key]=sorted(groups[key],key=lambda r:(r['case'],r['seed']))
            tables[scope][g]={m:stats(v) for m,v in groups.items()}
            pairs[scope][g]={m:paired(groups[m],groups['qLogEI']) for m in ('UCB_official','UCB_fixed')}
            pairs[scope][g]['fixed_minus_official']=paired(groups['UCB_fixed'],groups['UCB_official'])
            if g!='none':
                knowledge[scope][g]={}
                for acq,group in groups.items():
                    b0=[baseline(c,seed,'none','B0') for c in cases for seed in range(5)] if acq=='qLogEI' else sorted(
                        [r for r in rows if r['case'] in cases and r['regime']=='none' and r['mode']==acq.removeprefix('UCB_')],key=lambda r:(r['case'],r['seed']))
                    knowledge[scope][g][acq]=paired(group,b0)|{'mean_clipped_downside_to_own_B0_pp':st.mean(max(metric(b)['auc']-metric(a)['auc'],0) for a,b in zip(group,b0))}
    allnew=new_rows();assert len(allnew)==210
    result={'protocol_sha256':p['protocol_sha256'],'selected_kappa':s.read(s.OUT/'selected_kappa.json')['kappa'],
            'new_unique_trajectories':len(allnew),'new_complete':sum(r['status']=='COMPLETE' for r in allnew),
            'new_logical_queries':sum(len(r['history']) for r in allnew),'main_UCB_trajectories':len(rows),
            'development_reused_in_main':5,'frozen_qLogEI_comparators':100,'new_LLM_answers':0,
            'tables':tables,'paired_vs_qLogEI':pairs,'region_minus_own_acquisition_B0':knowledge,
            'raw_new_metrics':[{k:r[k] for k in ('run_id','case','seed','regime','method','mode','kappa','stage','status')}|metric(r) for r in allnew],
            'limits':['q=1 finite pool, not batch BO.','Case1 selected kappa; primary comparison is Case2-4. Only three primary chemistries, descriptive repeated seeds.',
                      'Original synthetic stress knowledge is reference-aware; fixed assistant ROI is not fresh roundwise advice.',
                      'Current shared chat exposed to old test results, exploratory rather than blind confirmation.',
                      'Common ordinal catalyst code and GP misspecification unchanged.','UCB beta schedule alone does not reproduce full native LLINBO.']}
    s.write(s.OUT/'results.json',result)
    return {'completed':result['new_complete'],'queries':result['new_logical_queries'],'kappa':result['selected_kappa'],
            'primary_AUC':{g:{m:round(v['AUC_mean'],4) for m,v in t.items()} for g,t in tables['primary'].items()}}


def audit_group(job):
    case,seed=job;pool=s.read(s.parent.ROOT/'inputs'/f'case{case}'/'pool.json');ids=pool['ids'];ix={v:i for i,v in enumerate(ids)}
    y=np.load(s.parent.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values'];k=s.read(s.OUT/'selected_kappa.json')['kappa']
    cache=s.cache_for(case,seed,'prefix','audit_cache');records=[];numeric=[]
    for g in common.REGIMES:
        row=s.read(Path('data/runs/suzuki_llinbo_guarded/runs')/f'c{case}_s{seed}_{g}_LLINBO_matched'/'state.json')
        for t,event in enumerate(row['decisions'],1):
            h=row['history'][:t+3];queried={r['candidate_id'] for r in h};available=np.array([key not in queried for key in ids])
            q,meta=cache.score(h);b=common.argmax(q,available,ids);eligible=available&((q[b]-q)<=common.budget_at(4,t));n=int(eligible.sum())
            proposed=ix[event['proposed_id']];incumbent=max(r['y'] for r in h);comp=cache.components(h)
            assert bool(eligible[proposed])==event['accepted']
            record={'case':case,'seed':seed,'regime':g,'iteration':t,'qLogEI_key':meta['key'],'component_key':comp['metadata']['key'],
                    'proposal_id':ids[proposed],'capacity':n,'available':int(available.sum()),'qLogEI_accepted':event['accepted'],
                    'proposal_improves_incumbent':float(y[proposed])>incumbent+1e-10,
                    'proposal_incumbent_advantage_vs_qLogEI_reference_pp':(max(incumbent,float(y[proposed]))-max(incumbent,float(y[b])))*100,'UCB':{}}
            for mode in ('official','fixed'):
                beta=e.beta_at(mode,t,k);ucb=comp['mu']+np.sqrt(beta)*comp['sd'];mask=e.equal_capacity_mask(ucb,available,n,ids)
                ref=common.argmax(ucb,available,ids)
                record['UCB'][mode]={'accepted':bool(mask[proposed]),'reference_id':ids[ref],'beta':beta,
                                    'capacity':int(mask.sum()),'same_members':int((mask&eligible).sum()),
                                    'proposal_incumbent_advantage_vs_own_reference_pp':(max(incumbent,float(y[proposed]))-max(incumbent,float(y[ref])))*100}
            records.append(record)
    # The old B0 histories are also held fixed. These are not new trajectories.
    row=baseline(case,seed,'none','B0')
    for t in range(1,17):
        h=row['history'][:t+3];available=np.array([key not in {r['candidate_id'] for r in h} for key in ids])
        q,meta=cache.score(h);comp=cache.components(h,analytic_ei=True);a=int(common.argmax(q,available,ids))
        b=int(common.argmax(comp['analytic_logei'],available,ids))
        rec={'case':case,'seed':seed,'iteration':t,'qLogEI_key':meta['key'],'component_key':comp['metadata']['key'],
             'qLogEI_id':ids[a],'analytic_LogEI_id':ids[b],'same_argmax':a==b,
             'analytic_logEI_loss_of_MC_choice':float(comp['analytic_logei'][b]-comp['analytic_logei'][a]),
             'qLogEI_selected_sd':float(comp['sd'][a]),'UCB':{}}
        for mode in ('official','fixed'):
            beta=e.beta_at(mode,t,k);ucb=comp['mu']+np.sqrt(beta)*comp['sd'];j=int(common.argmax(ucb,available,ids))
            rec['UCB'][mode]={'id':ids[j],'same_argmax_as_qLogEI':j==a,'selected_sd':float(comp['sd'][j]),
                              'sd_minus_qLogEI':float(comp['sd'][j]-comp['sd'][a]),'mean_minus_qLogEI':float(comp['mu'][j]-comp['mu'][a])}
        numeric.append(rec)
    assert cache.counters['fits']==0
    result={'records':records,'numeric':numeric,'cache_counters':cache.counters,'operator_compiles_in_loop':sum(s.JIT.values())}
    s.write(s.ROOT/'prefix_audit'/f'c{case}_s{seed}.json',result);return result


def audit():
    s.check_protocol();jobs=[(c,seed) for c in (1,2,3,4) for seed in range(5)]
    with ProcessPoolExecutor(max_workers=4,initializer=s.worker_init) as executor:groups=list(executor.map(audit_group,jobs))
    records=[v for r in groups for v in r['records']];numeric=[v for r in groups for v in r['numeric']]
    result={'scope':'Frozen original prefixes, conditional diagnostics only; equal capacity is not native LLINBO psi or an EI-ratio bound.',
            'new_LLM_answers':0,'new_oracle_queries':0,'new_GP_fits':sum(g['cache_counters']['fits'] for g in groups),
            'LLINBO_prefixes':len(records),'B0_prefixes':len(numeric),'gate':{},'numeric':{}}
    for scope,cases in [('primary',(2,3,4)),('all_four',(1,2,3,4))]:
        result['gate'][scope]={}
        for g in common.REGIMES:
            subset=[r for r in records if r['case'] in cases and r['regime']==g];out={}
            for mode in ('qLogEI','official','fixed'):
                accept=lambda r:r['qLogEI_accepted'] if mode=='qLogEI' else r['UCB'][mode]['accepted']
                out[mode]={'prefixes':len(subset),'accepted':sum(accept(r) for r in subset),
                           'missed_proposal_improves_incumbent':sum(not accept(r) and r['proposal_improves_incumbent'] for r in subset),
                           'accepted_one_step_incumbent_worse_vs_common_qLogEI_reference':sum(accept(r) and r['proposal_incumbent_advantage_vs_qLogEI_reference_pp']<-1e-10 for r in subset)}
                if mode!='qLogEI':out[mode].update(changed_admissions=sum(accept(r)!=r['qLogEI_accepted'] for r in subset),
                      mean_admissible_set_overlap_fraction=st.mean(r['UCB'][mode]['same_members']/r['capacity'] for r in subset))
            result['gate'][scope][g]=out
        subset=[v for v in numeric if v['case'] in cases]
        result['numeric'][scope]={'prefixes':len(subset),'MC_vs_analytic_same_argmax':sum(v['same_argmax'] for v in subset),
              'max_analytic_logEI_loss_of_MC_choice':max(v['analytic_logEI_loss_of_MC_choice'] for v in subset),
              'mean_analytic_logEI_loss_of_MC_choice':st.mean(v['analytic_logEI_loss_of_MC_choice'] for v in subset),
              'UCB':{mode:{'same_argmax_as_qLogEI':sum(v['UCB'][mode]['same_argmax_as_qLogEI'] for v in subset),
                    'mean_selected_sd_difference':st.mean(v['UCB'][mode]['sd_minus_qLogEI'] for v in subset),
                    'mean_selected_mu_difference':st.mean(v['UCB'][mode]['mean_minus_qLogEI'] for v in subset),
                    'selects_higher_sd':sum(v['UCB'][mode]['sd_minus_qLogEI']>1e-10 for v in subset)} for mode in ('official','fixed')}}
    s.write(s.OUT/'prefix_audit.json',result);return result


def tree(folder):
    files=[f for f in sorted(folder.rglob('*')) if f.is_file()];h=sha256();size=0
    for f in files:
        size+=f.stat().st_size;h.update((f.relative_to(folder).as_posix()+'\0'+s.filehash(f)+'\n').encode('utf8'))
    return {'files':len(files),'bytes':size,'sha256':h.hexdigest()}


def verify():
    p=s.check_protocol();rows=new_rows();main=main_rows();assert len(rows)==210 and len(main)==200
    assert len({r['run_id'] for r in rows})==210
    decisions=0;initials=0
    for row in rows:
        assert row['protocol_sha256']==p['protocol_sha256'] and row['status']=='COMPLETE'
        assert len(row['history'])==20 and len(row['decisions'])==16 and row['operator_compiles_in_loop']==0
        case,seed,g=row['case'],row['seed'],row['regime'];folder=s.parent.ROOT/'inputs'/f'case{case}'
        pool=s.read(folder/'pool.json');ids=pool['ids'];ix={v:i for i,v in enumerate(ids)};y=np.load(folder/'oracle.npz')['values']
        assert [v['candidate_id'] for v in row['history'][:4]]==pool['initial_ids'][str(seed)]
        assert len({v['candidate_id'] for v in row['history']})==20;initials+=1
        advice=s.read(folder/'advice.json')[str(seed)].get(g,{'clauses':[],'confidence':.7})
        box=common.region_for(advice,[v['id'] for v in advice['clauses']]) if g!='none' else None
        cache=s.ROOT/'cache'/f'c{case}_s{seed}_{g}'
        for v in row['history']:assert v['y']==float(y[ix[v['candidate_id']]])
        for t,event in enumerate(row['decisions'],1):
            history=row['history'][:t+3];fit=digest({'history':sorted(history,key=lambda v:v['candidate_id']),'case':case,'seed':seed,'recipe':s.filehash(common.__file__)})
            meta=event['acquisition'];assert meta['fit_key']==fit
            key=digest({'fit':fit,'box':box,'confidence':advice['confidence'] if box else None,'analytic_ei':False,'recipe':s.filehash(e.__file__)})
            assert meta['key']==key
            if t==1:
                old=baseline(case,seed,g,'B0' if g=='none' else 'LGBO_region')['decisions'][0]
                native=old['base_acquisition'] if g=='none' else next(iter(old['guided_acquisitions'].values()))
                assert meta['fit_key']==native['fit_key'] and meta['regional_seed']==native['mc_seed']
                assert meta['delta']==native['delta'] and meta['lambda']==native['lambda']
                assert meta['common_covariance_jitter']==native['common_covariance_jitter']
            with np.load(cache/(key+'.npz')) as data:mu=data['mu'].copy();sd=data['sd'].copy()
            # Deliberately independent expression for beta and independent argmax.
            beta=2*np.log(t*4*np.pi**2/.1*6) if row['mode']=='official' else row['kappa']**2
            assert beta==meta['beta'];score=mu+np.sqrt(beta)*sd;observed={v['candidate_id'] for v in history}
            j=min([i for i,key in enumerate(ids) if key not in observed],key=lambda i:(-float(score[i]),ids[i]))
            assert event['selected_id']==ids[j]==row['history'][t+3]['candidate_id']
            assert event['score']==float(score[j]) and event['mean_std']==float(mu[j]) and event['sd_std']==float(sd[j]);decisions+=1
    selection=s.read(s.OUT/'selected_kappa.json');means={k:st.mean(metric(r)['auc'] for r in rows if r['stage']=='development' and r['kappa']==k) for k in e.KAPPAS}
    assert selection['kappa']==min(means,key=lambda k:(-means[k],k))
    protected={};cache_trees={}
    for label,path in s.PROTECTED.items():
        inv=s.read(path)
        for f in inv['files']:assert s.filehash(f['path'])==f['sha256'],f['path']
        protected[label]=len(inv['files'])
        for key,subdir in [('cache_tree','cache'),('audit_cache_tree','audit_cache')]:
            if key not in inv:continue
            actual=tree(Path('data/runs')/f'suzuki_{label}'/subdir);assert actual=={k:inv[key][k] for k in actual}
            cache_trees[label+'_'+subdir]=actual
    audit_groups=[s.read(f) for f in sorted((s.ROOT/'prefix_audit').glob('*.json'))]
    assert len(audit_groups)==20
    capacity_checks=0;numeric_checks=0
    k=selection['kappa']
    for group in audit_groups:
        assert group['cache_counters']['fits']==0 and group['operator_compiles_in_loop']==0
        for rec in group['records']:
            c,seed,t=rec['case'],rec['seed'],rec['iteration'];folder=s.parent.ROOT/'inputs'/f'case{c}'
            pool=s.read(folder/'pool.json');ids=pool['ids'];ix={v:i for i,v in enumerate(ids)}
            row=s.read(Path('data/runs/suzuki_llinbo_guarded/runs')/f"c{c}_s{seed}_{rec['regime']}_LLINBO_matched"/'state.json')
            h=row['history'][:t+3];available=np.array([key not in {v['candidate_id'] for v in h} for key in ids]);cache=s.ROOT/'audit_cache'/f'c{c}_s{seed}_prefix'
            with np.load(cache/(rec['qLogEI_key']+'.npz')) as data:q=data['scores'].copy()
            with np.load(cache/(rec['component_key']+'.npz')) as data:mu=data['mu'].copy();sd=data['sd'].copy()
            best=max(q[available]);mask=available&((best-q)<=np.log(4)*(16-t)/15);assert mask.sum()==rec['capacity']
            assert bool(mask[ix[rec['proposal_id']]])==rec['qLogEI_accepted']
            for mode in ('official','fixed'):
                beta=2*np.log(t*4*np.pi**2/.1*6) if mode=='official' else k**2;scores=mu+np.sqrt(beta)*sd
                ranked=sorted(np.flatnonzero(available),key=lambda j:(-float(scores[j]),ids[j]));chosen=ranked[:int(mask.sum())]
                assert len(chosen)==rec['UCB'][mode]['capacity']
                assert (ix[rec['proposal_id']] in chosen)==rec['UCB'][mode]['accepted'];capacity_checks+=1
        for rec in group['numeric']:
            c,seed,t=rec['case'],rec['seed'],rec['iteration'];pool=s.read(s.parent.ROOT/'inputs'/f'case{c}'/'pool.json');ids=pool['ids']
            h=baseline(c,seed,'none','B0')['history'][:t+3];available=[i for i,key in enumerate(ids) if key not in {r['candidate_id'] for r in h}]
            cache=s.ROOT/'audit_cache'/f'c{c}_s{seed}_prefix'
            with np.load(cache/(rec['component_key']+'.npz')) as d:loge=d['analytic_logei'].copy();mu=d['mu'].copy();sd=d['sd'].copy()
            j=min(available,key=lambda i:(-float(loge[i]),ids[i]));assert ids[j]==rec['analytic_LogEI_id']
            for mode in ('official','fixed'):
                scores=mu+np.sqrt(2*np.log(t*4*np.pi**2/.1*6) if mode=='official' else k**2)*sd
                j=min(available,key=lambda i:(-float(scores[i]),ids[i]));assert ids[j]==rec['UCB'][mode]['id']
            numeric_checks+=1
    result=s.read(s.OUT/'results.json');assert result['new_logical_queries']==4200 and result['new_complete']==210
    for g in ('none',*common.REGIMES):
        for mode in ('official','fixed'):
            subset=[r for r in main if r['case'] in (2,3,4) and r['regime']==g and r['mode']==mode]
            auc=st.mean(sum(max(v['y'] for v in r['history'][:i])*100 for i in range(5,21))/16 for r in subset)
            assert abs(auc-result['tables']['primary'][g]['UCB_'+mode]['AUC_mean'])<1e-10
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'new_unique_trajectories':210,'main_UCB_trajectories':200,
             'development_reuse':5,'logical_queries':4200,'independent_policy_and_lineage_checks':decisions,
             'initializations_verified':initials,'independent_equal_capacity_gate_checks':capacity_checks,'independent_numeric_prefix_checks':numeric_checks,
             'operator_compiles_in_loop':0,'protected_files_unchanged':protected,'protected_cache_trees_unchanged':cache_trees,
             'development_selection_recomputed':True,'results_sha256':s.filehash(s.OUT/'results.json'),'audit_sha256':s.filehash(s.OUT/'prefix_audit.json'),
             'boundary':'Implementation integrity only; scientific outcome requires the paired performance and diagnostic results.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def release():
    p=s.check_protocol();receipt=s.read(s.OUT/'validation_receipt.json');assert receipt['integrity_status']=='PASS'
    tests=ET.parse(s.OUT/'component_tests.xml').getroot();suites=list(tests.iter('testsuite'))
    counts={key:sum(int(v.attrib.get(key,0)) for v in suites) for key in ('tests','failures','errors','skipped')}
    assert counts['tests']>=6 and counts['failures']==counts['errors']==0
    files=[Path(__file__),Path(s.__file__),Path(e.__file__),Path('chem/benchmarks/suzuki_acquisition.py'),Path('chem/benchmarks/acquisition_suzuki_figures.py'),Path('tests/test_suzuki_acquisition.py'),Path('docs/suzuki_acquisition.md')]
    files += [f for f in s.ROOT.rglob('*') if f.is_file() and 'cache' not in str(f.relative_to(s.ROOT)).split('/')[0] and not any(part.endswith('cache') or part=='pytest_tmp' for part in f.relative_to(s.ROOT).parts)]
    files += [f for f in s.OUT.rglob('*') if f.is_file() and f.name not in ('release_inventory.json','release_receipt.json')]
    entries=[{'path':f.as_posix(),'bytes':f.stat().st_size,'sha256':s.filehash(f)} for f in sorted(set(files))]
    inv={'protocol_sha256':p['protocol_sha256'],'file_count':len(entries),'files':entries,
         'cache_tree':tree(s.ROOT/'cache'),'audit_cache_tree':tree(s.ROOT/'audit_cache'),'qualification_cache_tree':tree(s.ROOT/'qualification_cache'),
         'cache_digest_definition':'Sorted native Path relative POSIX path + NUL + file SHA256 + LF; pytest scratch excluded.'}
    s.write(s.OUT/'release_inventory.json',inv)
    result={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'integrity_status':'PASS',
            'protocol_sha256':p['protocol_sha256'],'new_unique_trajectories':210,'new_logical_queries':4200,'new_LLM_answers':0,
            'component_tests':counts,'operator_compiles_in_loop':0,'old_LLMinbo_prefixes':1280,'old_B0_numeric_prefixes':320,
            'protected_files_unchanged':receipt['protected_files_unchanged'],'validation_receipt_sha256':s.filehash(s.OUT/'validation_receipt.json'),
            'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),'scientific_result':'Read results.json: acquisition effects depend on UCB coefficient and knowledge condition.'}
    s.write(s.OUT/'release_receipt.json',result);return result
