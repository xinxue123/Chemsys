"""Copy the final-scope experiment; exclude the stopped screening runner."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
PORTABLE=ROOT/'Chemsys-portable'
VARIANT='llm_event_start_qlogei_20261104'


def main():
    paths=list((ROOT/'code/experiment_tools/llm_event_start').glob('*.py'))
    paths += [ROOT/'code/experiment_tools/llm_screen_qlogei'/name for name in
              ('__init__.py','knowledge.py','policy.py','validation.py')]
    paths += [ROOT/'code/experiment_tools/run_llm_event_start.py',ROOT/'code/tools/event_start_operations.py',
              ROOT/'docs'/f'{VARIANT}.txt']
    paths += [ROOT/'code/analysis'/name for name in ('analyze_llm_event_start.py',
              'check_llm_event_start_gp.py','plot_llm_event_start.py')]
    for source in paths:
        target=PORTABLE/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    result=PORTABLE/'results'/VARIANT
    result.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'experiments/reports'/VARIANT).iterdir():
        if source.is_file():
            shutil.copyfile(source,result/source.name)
    native=ROOT/'experiments/runs'/VARIANT
    for name in ('protocol.json','preflight.json','evaluation_endpoint.json','main_receipt.json','control_receipt.json'):
        shutil.copyfile(native/name,result/('source_'+name))
    reference=native/'runs/bo_s0/state.json'
    shutil.copyfile(reference,result/'initial_GP_reference.json')
    shutil.copyfile(native/'runs/event_s0/state.json',result/'event_reference.json')
    packets=[]
    for sf in sorted((native/'runs').glob('event_*/state.json')):
        state=json.loads(sf.read_text(encoding='utf-8'))
        for c in state['decisions']:
            if c['LLM'] is None:
                continue
            request=ROOT/c['LLM']['request_relpath']
            export=json.loads((request.parent/'export.json').read_text(encoding='utf-8'))
            packets.append(dict(seed=state['job']['seed'],query=c['query'],
                model='gpt-6.1-terra',effort='medium',
                input=json.loads((ROOT/export['input']).read_text(encoding='utf-8')),
                response=c['LLM']['raw'],source_request_sha256=c['LLM']['response_receipt']['request_sha256']))
    (result/'LLM_packets.json').write_text(json.dumps(packets,indent=2)+'\n',encoding='utf-8')

    # Relocation-only changes: capture local source/input paths, guard archives
    # that actually exist, and use the compact numerical reference for preflight.
    target=PORTABLE/'code/experiment_tools/llm_event_start/study.py'
    text=target.read_text(encoding='utf-8')
    text=text.replace("old=read(REFERENCE/'protocol.json')",'''pool=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    local_sources={}
    for directory in ('code/chemist_system/chem','code/experiment_tools'):
        for source in (ROOT/directory).rglob('*.py'):
            local_sources[source.relative_to(ROOT).as_posix()]=sha(source)
    old=dict(sources_sha256=local_sources,inputs_sha256={
        source.relative_to(ROOT).as_posix():sha(source)
        for source in (pool.source_path,pool.source_path.with_name('oracle.npz'))})''')
    text=text.replace("        protected[path.relative_to(ROOT).as_posix()]=sha(path)\n    p=dict(",
                      "        if path.exists():\n            protected[path.relative_to(ROOT).as_posix()]=sha(path)\n    p=dict(")
    text=text.replace("read(REFERENCE/'evaluation_endpoint.json')",
                      "read(ROOT/'results'/VARIANT/'source_evaluation_endpoint.json')")
    text=text.replace("for path in (REFERENCE/'cache').iterdir():","for path in (REFERENCE/'cache').glob('*'):")
    text=text.replace("read(REFERENCE/'runs/s4_middle_qlogei/state.json')",
                      "read(ROOT/'results'/VARIANT/'initial_GP_reference.json')")
    target.write_text(text,encoding='utf-8')
    with (result/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## Portable workspace\n\n'
            'This folder contains all compact results, all fourteen public LLM packets and '
            'their source receipts, the source protocol, numerical qualification and exact '
            'first-prefix GP reference. Full native state/GP-array custody remains in the '
            'original experiment archive. The aborted yield-stratified screening runner is '
            'excluded; its shared knowledge/KF math is included as a dependency.\n\n'
            'The scientific GP, proposal, projection, review, KF and prompt code is unchanged. '
            'The portable study registers local source/input hashes, guards only locally '
            'available archives, uses the bundled evaluator endpoint and compact GP reference, '
            'and treats a missing imported fit cache as empty. These are relocation adapters, '
            'not changes to the experiment algorithm. The source protocol is provenance, '
            'not a relocated execution protocol.\n\n'
            'Reproduction uses the pinned BO environment and a Codex orchestrator providing '
            'fresh isolated LLM responders. It is not an unattended API runner. Run prepare, '
            'preflight and the main phase, service prepared requests with event_start_operations.py, '
            'then run the control phase and analysis. No API key is bundled.\n\n'
            '~~~sh\n'
            'python code/experiment_tools/run_llm_event_start.py prepare\n'
            'python code/experiment_tools/run_llm_event_start.py preflight\n'
            'python code/experiment_tools/run_llm_event_start.py run --phase main --workers 6\n'
            'python code/experiment_tools/run_llm_event_start.py run --phase control --workers 6\n'
            'python code/analysis/analyze_llm_event_start.py\n'
            'python code/analysis/check_llm_event_start_gp.py\n'
            'python code/analysis/plot_llm_event_start.py\n'
            '~~~\n')
    index=PORTABLE/'results/README.md'
    text=index.read_text(encoding='utf-8')
    if '## Unrestricted LLM starting point' not in text:
        text+='\n## Unrestricted LLM starting point and confidence events\n\n'
        text+='The [Case 1 pilot](llm_event_start_qlogei_20261104/README.md) includes 30 complete '
        text+='eight-query trajectories. LLM chooses the first point without yield grouping; '
        text+='subsequent normal choices use full-pool qLogEI. KF triggers four later refreshes, '
        text+='all of which abstain and withdraw the failed knowledge. The matched LLM-start '
        text+='control isolates the zero incremental refresh effect from the initialization effect.\n'
        index.write_text(text,encoding='utf-8')
    index=PORTABLE/'README.md'
    text=index.read_text(encoding='utf-8')
    if '## LLM first point and sparse refresh' not in text:
        text+='\n## LLM first point and sparse refresh\n\n'
        text+='The [latest Case 1 experiment](results/llm_event_start_qlogei_20261104/README.md) '
        text+='includes code, all public LLM packets, round-level comparisons and numerical audits. '
        text+='Initialization is unrestricted; ordinary later queries use qLogEI without screening.\n'
        index.write_text(text,encoding='utf-8')
    mp=PORTABLE/'MANIFEST.json'
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    manifest['LLM_event_start_pilot']=dict(trajectories=30,observations=240,
        unrestricted_initial_points=True,initial_LLM_calls=10,refresh_calls=4,
        KF_updates=22,accepted_refresh_proposals=0,postinitial_qLogEI_deviations=0,
        same_LLM_start_control=True,documentation_language='English',
        source_protocol_sha256=json.loads((result/'source_protocol.json').read_text(encoding='utf-8'))['protocol_sha256'])
    local_protocol=PORTABLE/'experiments/runs'/VARIANT/'protocol.json'
    local_preflight=local_protocol.with_name('preflight.json')
    if local_protocol.exists():
        manifest['LLM_event_start_pilot']['portable_protocol_sha256']=json.loads(local_protocol.read_text(encoding='utf-8'))['protocol_sha256']
    if local_preflight.exists():
        manifest['LLM_event_start_pilot']['portable_preflight']=json.loads(local_preflight.read_text(encoding='utf-8'))['status']
    files=[p for p in PORTABLE.rglob('*') if p.is_file() and p!=mp
        and not any(part in p.parts for part in ('__pycache__','.git','.venv','.pytest_cache'))
        and p.suffix not in ('.pyc','.lock')]
    manifest['sha256']={p.relative_to(PORTABLE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest['file_count_excluding_manifest']=len(files)
    manifest['bytes_excluding_manifest']=sum(p.stat().st_size for p in files)
    mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PACKAGED',files=len(files),public_LLM_packets=len(packets))))


if __name__=='__main__':
    main()
