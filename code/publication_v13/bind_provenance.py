"""Link every V13 GP commit to its saved checkpoint and add source provenance.

This appends a separate supplement to the sealed evidence archive; it never
rewrites the existing milestone, experiments, protocols, or model files.
"""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from build_evidence import read,sha,rel,write_csv,write_json,VARIANTS,REPORTS,SHORT

P=ROOT/'paper/v13'
M=ROOT/'experiments/milestones/v13_publication_20261103'


def main():
    manifest=read(M/'manifest.json');receipt=read(M/'receipt.json')
    rows=[];matched=0;no_data=0
    with (P/'data/metrics.csv').open(encoding='utf-8',newline='') as f:
        runs=[x for x in csv.DictReader(f) if x['method']=='v13']
    for m in runs:
        sf=ROOT/m['state_path'];state=read(sf);seed=int(m['seed']);regime=m['regime']
        variant=VARIANTS[0 if seed==0 else 1] if regime=='warm' else VARIANTS[2 if seed==0 else 3]
        for q in range(1,int(m['queries'])+1):
            folder=sf.parent/'rounds';cp=folder/f'{q:02}_commit.json';fp=folder/f'{q:02}_fit.json';gp=folder/f'{q:02}_GP.npz'
            c=read(cp);key=c['GP_fit_key'];empty=regime=='cold' and q==1
            model=ROOT/'experiments/runs'/variant/'cache'/state['task']/(key+'.pt')
            model_hash=''
            if empty:
                no_data+=1
            else:
                if not model.is_file():raise ValueError('Missing native GP checkpoint')
                model_hash=sha(model)
                if manifest['files_sha256'][rel(model)]!=model_hash:raise ValueError('Checkpoint not bound by sealed manifest')
                matched+=1
            rows.append(dict(regime=regime,task=m['task'],seed=seed,query=q,fit_key=key,
                stage='no_data' if empty else 'fitted_GP',
                checkpoint='' if empty else rel(model),checkpoint_sha256=model_hash,
                fit_record=rel(fp),fit_sha256=sha(fp),commit_path=rel(cp),commit_sha256=sha(cp),
                GP_arrays=rel(gp),GP_arrays_sha256=sha(gp)))
    assert (matched,no_data)==(1120,32)
    write_csv(P/'data/fit_inventory.csv',rows)
    for variant,report in zip(VARIANTS,REPORTS[:4]):
        dest=P/'protocols'/(variant+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'experiments/runs'/variant/'protocol.json',dest)
        shutil.copy2(ROOT/'experiments/reports'/report/'audit.json',P/'audit'/('native_'+variant+'.json'))
    shutil.copy2(M/'receipt.json',P/'audit/preservation_receipt.json')
    v=read(ROOT/'paper/v12_subagent/data/runtime_versions.json')
    write_json(P/'data/registered_runtime_versions.json',v)
    observed=dict(python=sys.version,platform=platform.platform(),scope='Publication-time snapshot of existing numerical interpreter; not an independent attestation of every historical run',
        packages={x:importlib.metadata.version(x) for x in ('numpy','scipy','torch','botorch','gpytorch','matplotlib')})
    write_json(P/'audit/authoring_versions.json',observed)
    cold=read(ROOT/'experiments/runs'/VARIANTS[2]/'protocol.json')
    datasets=[]
    for pool in cold['pools']:
        name=pool['task'];short=SHORT[name]
        file=next(ROOT/x for x in cold['inputs_sha256'] if x.endswith('pool.json') and (short.replace('c','case') in x if short.startswith('c') else name in x))
        data=read(file)
        datasets.append(dict(task=short,full_task=name,candidates=len(data['ids']),
            continuous=sum(x=='continuous' for x in pool['kinds']),categorical=sum(x=='categorical' for x in pool['kinds']),
            axes=json.dumps(pool['axes']),pool_path=rel(file),pool_sha256=sha(file),
            oracle_path=rel(file.with_name('oracle.npz')),oracle_sha256=sha(file.with_name('oracle.npz'))))
    write_csv(P/'data/datasets.csv',datasets)
    binding=dict(status='PASS_ALL_V13_GP_PREFIXES_BOUND',native_GP_decisions=matched,no_data_first_decisions=no_data,
        archive_GP_checkpoint_files=receipt['GP_checkpoint_files'],
        every_fitted_commit_has_preserved_checkpoint=True,
        fit_inventory_sha256=sha(P/'data/fit_inventory.csv'),
        scientific_source_files_modified=False,
        preservation_manifest_sha256=sha(M/'manifest.json'))
    write_json(P/'audit/model_binding.json',binding)
    # Preserve raw HPLC replicates, source licences and environment provenance
    # in a distinct additive bundle rather than rewriting an already sealed ZIP.
    extra=set()
    for folder in (ROOT/'experiments/datasets/continuous_20261029/sources',
                   ROOT/'experiments/datasets/reasoning_bo',
                   ROOT/'experiments/datasets/reaction_transfer'):
        extra.update(x.resolve() for x in folder.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    for name in ('source_receipt.json','snar_oracle_receipt.json','data_quality.json','validation_receipt.json','summary_provenance.json'):
        x=ROOT/'experiments/reports/continuous_20261029'/name
        if x.is_file():extra.add(x)
    for folder in (ROOT/'code/chemist_system/third_party/reasoning_bo_official',ROOT/'code/chemist_system/third_party/suzuki_official'):
        extra.update(x.resolve() for x in folder.rglob('*') if x.is_file() and x.name.lower().startswith(('license','notice','copying')))
    for x in (ROOT/'code/chemist_system/pyproject.toml',P/'data/registered_runtime_versions.json',P/'audit/model_binding.json',P/'data/fit_inventory.csv'):
        extra.add(x)
    if (M/'provenance_receipt.json').exists():
        old=read(M/'provenance_receipt.json')
        for name,h in old['artifact_sha256'].items():assert sha(M/name)==h
    else:
        index={rel(x):sha(x) for x in sorted(extra)}
        write_json(M/'source_provenance_manifest.json',dict(files_sha256=index,purpose='Raw source tables, licences, environment receipt and V13 GP-to-commit binding; additive to sealed evidence ZIP'))
        zpath=M/'source_provenance.zip'
        with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in index:z.write(ROOT/name,name)
        with zipfile.ZipFile(zpath) as z:
            for name,h in index.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        write_json(M/'provenance_receipt.json',dict(status='SEALED_AND_VERIFIED',files=len(index),main_evidence_manifest_sha256=sha(M/'manifest.json'),
            original_files_modified=False,artifact_sha256={n:sha(M/n) for n in ('source_provenance_manifest.json','source_provenance.zip')}))
    shutil.copy2(M/'provenance_receipt.json',P/'audit/source_provenance_receipt.json')
    print(json.dumps(binding))


if __name__=='__main__':main()
