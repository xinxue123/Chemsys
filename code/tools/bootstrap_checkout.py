"""Initialize ignored runtime directories for a source-only Git checkout.

This does not download data, create API credentials or manufacture archived
protocols. Existing migrated workspaces are left untouched.
"""
from hashlib import sha256
from pathlib import Path
import json
import os


def source_bindings(workspace):
    result=[]
    for base in [workspace/'code/chemist_system',workspace/'code/experiment_tools',workspace/'code/resources']:
        for path in sorted(base.rglob('*')):
            if not path.is_file() or {'__pycache__','.git','node_modules'}.intersection(path.parts) or path.suffix in {'.pyc','.pyo'}:continue
            result.append({'legacy_relative':path.relative_to(base).as_posix(),
                'canonical_relative':path.relative_to(workspace).as_posix(),
                'sha256':sha256(path.read_bytes()).hexdigest()})
    return result


def initialize(workspace):
    workspace=Path(workspace).resolve();meta=workspace/'runtime/layout_20261030'
    manifest=meta/'migration_manifest.json'
    if manifest.exists():return json.loads(manifest.read_text(encoding='utf-8'))
    view=workspace/'experiments/chemist_system'
    view.mkdir(parents=True,exist_ok=True)
    for name in ['data','research','outputs']:(view/name).mkdir(exist_ok=True)
    rows=source_bindings(workspace)
    for row in rows:
        source=workspace/row['canonical_relative'];target=view/row['legacy_relative']
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha256(target.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError(f'Existing runtime source differs: {target}')
            continue
        os.link(source,target)
    result={'layout_version':1,'status':'SOURCE_CHECKOUT_INITIALIZED','mode':'source_checkout',
        'legacy_path':'experiments/chemist_system','experiments_path':'experiments/chemist_system',
        'code_path':'code/chemist_system','source_files':rows,
        'archive_status':'Not included in Git: restore original pools, runs, protocols and receipts separately.'}
    meta.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    value=initialize(Path(__file__).resolve().parents[2])
    print(json.dumps({'status':value['status'],'source_files':len(value['source_files']),
                      'new_experiments':0,'new_LLM_calls':0},ensure_ascii=False))
