"""Copy corrected science code and compact evidence into the portable workspace."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
PORTABLE=ROOT/'Chemsys-portable'
VARIANT='fast_hit_qlogei_rollout_20261104'


def main():
    copies=list((ROOT/'code/experiment_tools/qlogei_rollout').glob('*.py'))
    copies += [ROOT/'code/experiment_tools/run_qlogei_rollout.py',ROOT/'docs'/f'{VARIANT}.txt']
    copies += [ROOT/'code/analysis'/name for name in ('analyze_qlogei_rollout.py',
        'check_qlogei_rollout_value.py','plot_qlogei_rollout.py')]
    for source in copies:
        target=PORTABLE/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    results=PORTABLE/'results'/VARIANT
    results.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'experiments/reports'/VARIANT).iterdir():
        if source.is_file():
            shutil.copyfile(source,results/source.name)
    for name in ('preflight.json','protocol.json','evaluation_endpoint.json','batch_receipt.json'):
        shutil.copyfile(ROOT/'experiments/runs'/VARIANT/name,results/('source_'+name))
    with (results/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## Portable execution\n\n'
            'The package includes the same scientific GP and qLogEI-rollout code, the Case 1 '
            'pool and response table, and compact completed exports. Native per-round '
            'commits, fit checkpoints and full GP arrays remain in the original run archive. '
            'The source protocol is retained as provenance, not as a relocated execution protocol.\n\n'
            'To reproduce from the package root with the pinned BO environment, first generate '
            'the matched control archive, then register and run the corrected rollout. '
            'The controls use the already registered portable fast-hit protocol; the corrected '
            'runner registers the local source hashes and freezes those generated controls.\n\n'
            '~~~sh\n'
            'python code/experiment_tools/run_fast_hit_bo.py prepare\n'
            'python code/experiment_tools/run_fast_hit_bo.py preflight\n'
            'python code/experiment_tools/run_fast_hit_bo.py run --workers 6\n'
            'python code/experiment_tools/run_qlogei_rollout.py prepare\n'
            'python code/experiment_tools/run_qlogei_rollout.py preflight\n'
            'python code/experiment_tools/run_qlogei_rollout.py run --workers 6\n'
            'python code/analysis/analyze_qlogei_rollout.py\n'
            'python code/analysis/check_qlogei_rollout_value.py\n'
            'python code/analysis/plot_qlogei_rollout.py\n'
            '~~~\n\n'
            'The legacy runner also regenerates its archived analytic-EI arm. Those outcomes '
            'remain a separate comparison and never enter corrected rollout decisions. '
            'Initialization and GP caches are keyed by observed history. No LLM or API key is needed.\n')
    addition=('\n## Corrected qLogEI-based rollout\n\n'
        'The [corrected Case 1 study](fast_hit_qlogei_rollout_20261104/README.md) '
        'uses MC512 qLogEI for both the current and virtual next action. It adds ten '
        'trajectories with the same one-plus-seven budget and ten shared low/middle starts. '
        'The frozen qLogEI controls reach 85.09% yield in 3/10 starts and the corrected '
        'rollout in 1/10. The previous analytic-EI rollout stays archived separately.\n')
    index=PORTABLE/'results/README.md'
    text=index.read_text(encoding='utf-8')
    if '## Corrected qLogEI-based rollout' not in text:
        index.write_text(text+addition,encoding='utf-8')
    index=PORTABLE/'README.md'
    text=index.read_text(encoding='utf-8')
    if '## qLogEI rollout correction' not in text:
        text+='\n## qLogEI rollout correction\n\n'
        text+='The [corrected rollout study](results/fast_hit_qlogei_rollout_20261104/README.md) '
        text+='includes its source, numerical qualification and complete compact outcomes. '
        text+='Both decision stages use qLogEI. The linked report gives the reproduction order.\n'
        index.write_text(text,encoding='utf-8')
    # Refresh after numerical verification as well, so this is safe to call twice.
    manifest_path=PORTABLE/'MANIFEST.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['qlogei_rollout_correction']=dict(new_trajectories=10,new_observations=80,
        frozen_qlogei_controls=10,total_budget=8,initial=1,qlogei_target_hits=3,
        corrected_rollout_target_hits=1,native_MC512_qlogei_both_stages=True,
        source_protocol_sha256=json.loads((results/'source_protocol.json').read_text(encoding='utf-8'))['protocol_sha256'],
        full_native_trajectories_require_original_run_archive=True)
    paths=[p for p in PORTABLE.rglob('*') if p.is_file() and p!=manifest_path
        and not any(x in p.parts for x in ('__pycache__','.git','.venv','.pytest_cache'))
        and p.suffix not in ('.pyc','.lock')]
    manifest['sha256']={p.relative_to(PORTABLE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(paths)}
    manifest['file_count_excluding_manifest']=len(paths)
    manifest['bytes_excluding_manifest']=sum(p.stat().st_size for p in paths)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PACKAGED',files=len(paths))))


if __name__=='__main__':
    main()
