"""Package the handoff controller and its non-performance qualification."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
PORTABLE=ROOT/'Chemsys-portable'
VARIANT='llm_bo_handoff_20261104'


def main():
    out=ROOT/'experiments/reports'/VARIANT
    checks=json.loads((out/'mechanism_checks.json').read_text(encoding='utf-8'))
    assert checks['status']=='PASS'
    doc=['# LLM first point and gradual transfer to pure BO','',
        'This is an implemented controller with mechanism checks. No new complete '
        'optimization trajectory has been run, and no SearchAUC improvement is claimed.','',
        'The first point is chosen by the LLM. Starting on query2, cached permitted '
        'knowledge can prefer regions within a qLogEI budget. The maximum authority '
        'ceiling shrinks with the remaining query budget. KF confidence can reduce '
        'the effective allowance inside that ceiling. Refreshing knowledge does not '
        'restart the handoff clock. The final query uses exactly the native full-pool '
        'qLogEI maximum and native tie priorities.','',
        'GP hyperparameter fitting and transfer of decision authority are separate. '
        'The controller leaves the GP and its qLogEI values unchanged. It does not '
        'keep the first four points fully LLM-controlled.','',
        'The linear ceiling and log(4) initial allowance are draft parameters. With '
        'fixed initial knowledge confidence, the maximum permitted acquisition '
        'deviation contracts as shown below. Changing confidence may change the '
        'effective deviation within the decreasing ceiling. These numbers are '
        'allowances, not probabilities of choosing the LLM.','',
        '| Query | Authority ceiling | Allowed log-qEI gap | Minimum exp(qLogEI) fraction |',
        '|---|---|---|---|']
    for row in checks['schedule']:
        doc.append(f"| {row['query']} | {row['LLM_authority_cap']:.4f} | {row['logei_gap']:.4f} | {row['minimum_exp_qLogEI_retained']:.4f} |")
    doc.extend(['',
        'Tests cover an early knowledge-induced change, contraction of the budget, '
        'an actual synthetic forecast/feedback/KF update that changes the following '
        'choice, paused-knowledge protection, no positive recommendation from an '
        'accurate failure forecast, refresh clock preservation and an exact BO '
        'endpoint with acquisition ties. A held Case 1 prefix also checks first-point '
        'custody and GP-array immutability. That shadow contains no new outcome and '
        'is not an optimization result.','',
        'Run the qualification from the workspace root:','',
        '~~~sh','python code/experiment_tools/run_llm_bo_handoff_check.py','~~~','',
        'The numerical check uses an existing archived prefix. In the portable '
        'workspace it can reconstruct that one-observation GP from the compact '
        'reference without fitting hyperparameters or calling the oracle. The '
        'full experiment runner has not yet been wired to this new controller.'])
    (out/'README.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
    sources=list((ROOT/'code/experiment_tools/llm_bo_handoff').glob('*.py'))+[
        ROOT/'code/experiment_tools/run_llm_bo_handoff_check.py',ROOT/'docs'/f'{VARIANT}.txt']
    for source in sources:
        target=PORTABLE/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    target=PORTABLE/'results'/VARIANT
    target.mkdir(parents=True,exist_ok=True)
    for source in out.iterdir():
        if source.is_file():
            shutil.copyfile(source,target/source.name)
    index=PORTABLE/'results/README.md'
    text=index.read_text(encoding='utf-8')
    if '## Gradual LLM-to-BO handoff' not in text:
        text+='\n## Gradual LLM-to-BO handoff controller\n\n'
        text+='The [handoff mechanism checks](llm_bo_handoff_20261104/README.md) '
        text+='implement one LLM first point, a contracting knowledge allowance from '
        text+='query2 and an exact pure-BO final choice. These are controller checks; '
        text+='no new full-trajectory performance result is available.\n'
        index.write_text(text,encoding='utf-8')
    mp=PORTABLE/'MANIFEST.json'
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    manifest['LLM_BO_handoff_controller']=dict(status='MECHANISM_CHECKS_PASS',
        performance_trajectories=0,first_point='LLM',transfer_starts_at_query=2,
        final_choice='exact native full-pool qLogEI and tie rule',GP_recipe_changed=False)
    files=[p for p in PORTABLE.rglob('*') if p.is_file() and p!=mp
        and not any(part in p.parts for part in ('__pycache__','.git','.venv','.pytest_cache'))
        and p.suffix not in ('.pyc','.lock')]
    manifest['sha256']={p.relative_to(PORTABLE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest['file_count_excluding_manifest']=len(files)
    manifest['bytes_excluding_manifest']=sum(p.stat().st_size for p in files)
    mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PACKAGED',controller_checks='PASS',performance_trajectories=0,files=len(files))))


if __name__=='__main__':
    main()
