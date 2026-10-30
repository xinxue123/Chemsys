"""Read-only custody checks for the organized code and experiment layout."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
META=ROOT/'runtime/layout_20261030'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=read(META/'migration_manifest.json')
    assert manifest['status'] in {'MOVED_PENDING_VERIFICATION','COMPLETE_AND_VERIFIED'}
    legacy=ROOT/'chemist_system';archive=ROOT/manifest['experiments_path']
    assert legacy.resolve()==archive.resolve()
    for row in manifest['source_files']:
        source=ROOT/row['canonical_relative'];compat=legacy/row['legacy_relative']
        assert source.is_relative_to(ROOT/'code') and source.is_file()
        assert sha(source)==row['sha256'] and sha(compat)==row['sha256'],row['legacy_relative']
        assert os.path.samefile(source,compat),row['legacy_relative']
    for move in manifest['directory_moves']:
        assert (legacy/move['original_relative']).resolve()==(ROOT/move['canonical_relative']).resolve(),move
    navigation={'logs':'outputs','runs':'data/runs','reports':'research','datasets':'data/benchmarks'}
    for name,target in navigation.items():assert (ROOT/'experiments'/name).resolve()==(archive/target).resolve()
    # Full archived evidence checks, not a new optimization run.
    pins={}
    for name in ['research/continuous_r5_20261030/protected_snapshot.json',
                 'research/continuous_r5_20261030/release_inventory.json',
                 'research/expert_consultation_20261030/sources.json']:
        for path,value in read(legacy/name).items():
            if path in pins:assert pins[path]==value
            pins[path]=value
    for path,value in pins.items():
        target=Path(path) if Path(path).is_absolute() else legacy/path
        assert target.is_file() and sha(target)==value,path
    assert not any((ROOT/'code/chemist_system'/name).exists() for name in ['data','outputs','.venv-official'])
    for root,dirs,files in os.walk(ROOT/'code'):
        dirs[:]=[d for d in dirs if d!='__pycache__']
        # Official source distributions contain their own small data fixtures;
        # these are dependencies, not this project's experiment directories.
        assert not {'node_modules','.venv-official','candidate_tables'}.intersection(dirs),root
        assert not any(Path(f).suffix in {'.log','.pt','.npz','.pyd'} for f in files),root
    report={'status':'PASS','source_files_relocated':len(manifest['source_files']),
        'source_hashes_unchanged':True,'compatibility_names_share_source_bytes':True,
        'existing_evidence_files_verified':len(pins),'directory_moves_verified':len(manifest['directory_moves']),
        'experiment_navigation_links_verified':len(navigation),
        'code_has_no_experiment_logs_arrays_weights_or_environments':True,
        'code':str(ROOT/'code'),'experiments':str(archive),'environments':str(ROOT/'runtime/environments'),
        'new_experiments':0,'new_LLM_calls':0}
    (META/'layout_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
