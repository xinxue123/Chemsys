"""Add the reliability review and compact native reconstruction evidence."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
PORTABLE=ROOT/'Chemsys-portable'
VARIANT='llm_initialization_audit_20261104'


def main():
    target=PORTABLE/'results'/VARIANT
    target.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'experiments/reports'/VARIANT).iterdir():
        if source.is_file():
            shutil.copyfile(source,target/source.name)
    shutil.copyfile(ROOT/'experiments/runs'/VARIANT/'registration.json',target/'source_registration.json')
    for name in ('audit_llm_initialization_reliability.py','report_llm_initialization_audit.py'):
        shutil.copyfile(ROOT/'code/analysis'/name,PORTABLE/'code/analysis'/name)
    for name in ('README.md','RELIABILITY_REVIEW.md'):
        destination=PORTABLE/'results/llm_event_start_qlogei_20261104'/name
        source=ROOT/'experiments/reports/llm_event_start_qlogei_20261104'/name
        if name=='README.md' and destination.exists():
            local=destination.read_text(encoding='utf-8')
            suffix='\n## Portable workspace'+local.split('## Portable workspace',1)[1] if '## Portable workspace' in local else ''
            destination.write_text(source.read_text(encoding='utf-8')+suffix,encoding='utf-8')
        else:
            shutil.copyfile(source,destination)
    with (target/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## Portable evidence scope\n\n'
            'Compact reconstruction and prompt-anchoring evidence is included. Full native '
            'reconstruction uses the original thirty states and their per-round GP arrays; '
            'that archive remains with the original workspace. This addition changes no '
            'experimental data, optimizer or LLM response. The primary experiment reproduction '
            'instructions remain in the starting-point study report.\n')
    index=PORTABLE/'results/README.md'
    text=index.read_text(encoding='utf-8')
    if '## Initialization reliability review' not in text:
        text+='\n## Initialization reliability review\n\n'
        text+='The [fresh reconstruction audit](llm_initialization_audit_20261104/README.md) '
        text+='checks 15 distinct starts and 105 complete-pool qLogEI choices using 60 new native '
        text+='GP fits. All choices and score arrays match. The audit limits the conclusion '
        text+='to the particular prompt, fixed-prior GP and tie sequence; it does not establish '
        text+='an intrinsic disadvantage of LLM initialization.\n'
        index.write_text(text,encoding='utf-8')
    mp=PORTABLE/'MANIFEST.json'
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    manifest['LLM_initialization_reliability_audit']=dict(unique_initial_conditions=15,
        prefixes=105,fresh_native_GP_fits=60,exact_choice_matches=105,max_abs_logei_error=0.,
        old_model_cache_used_for_computation=False,original_results_preserved=True,
        full_reconstruction_requires_original_run_archive=True)
    files=[p for p in PORTABLE.rglob('*') if p.is_file() and p!=mp
        and not any(part in p.parts for part in ('__pycache__','.git','.venv','.pytest_cache'))
        and p.suffix not in ('.pyc','.lock')]
    manifest['sha256']={p.relative_to(PORTABLE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest['file_count_excluding_manifest']=len(files)
    manifest['bytes_excluding_manifest']=sum(p.stat().st_size for p in files)
    mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PACKAGED',files=len(files))))


if __name__=='__main__':
    main()
