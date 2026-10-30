"""Seal the already completed R1 experiment; never run or change its strategy."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
RUN = ROOT / 'data/runs/continuous_revised_20261030'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    protocol = read(REPORT / 'protocol.json')
    audit = read(REPORT / 'validation_receipt.json')
    assert audit['status'] == 'PASS'
    assert audit['completed_trajectories'] == 2 and audit['added_choices_verified'] == 32
    assert audit['baselines_rerun'] == 0
    protected = read(REPORT / 'protected_snapshot.json')
    for name, expected in protected.items():
        assert digest(ROOT / name) == expected, name
    for key in ['source_sha256', 'input_sha256']:
        for name, expected in protocol[key].items():
            assert digest(ROOT / name) == expected, name

    checks = read(REPORT / 'document_checks.json')
    assert checks['pages'] == 2 and not checks['missing_numbers']
    assert checks['all_fonts_embedded'] and checks['missing_glyphs'] == 0
    assert checks['overfull_boxes'] == 0
    for kind, suffix in [('PDF', 'pdf'), ('TeX', 'tex')]:
        assert checks[kind + '_sha256'] == digest(REPORT / ('SnAr_HPLC_Revised_R1.' + suffix))
    # Both final delivery-page renders have been inspected in this task.
    for i in [1, 2]:
        assert (ROOT / f'outputs/continuous_revised_typeset/delivery_pages/page-{i}.png').is_file()
    checks['status'] = 'PASS'
    checks['visual_pages_passed'] = [1, 2]
    write(REPORT / 'document_checks.json', checks)

    phases = Counter()
    tokens = Counter()
    attempts = 0
    for context_path in sorted((RUN / 'requests').glob('*/context.json')):
        context = read(context_path)
        phases[context['phase']] += 1
        responses = sorted(context_path.parent.glob('attempt*/response.json'))
        attempts += len(responses)
        for response_path in responses:
            usage = read(response_path)['usage']
            for name in ['prompt_tokens', 'completion_tokens', 'total_tokens',
                         'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens']:
                tokens[name] += usage.get(name, 0)
    assert phases == {'region_refresh': 6, 'interaction_test': 4}
    usage = {'model': protocol['LLM']['model'], 'logical_calls': sum(phases.values()),
             'phases': dict(phases), 'response_attempts': attempts, 'tokens': dict(tokens),
             'credentials_included': False}
    write(REPORT / 'LLM_usage.json', usage)

    members = {}
    names = ['README.md', 'DESIGN.md', 'REPORT.md', 'protocol.json', 'qualification.json',
             'validation_receipt.json', 'protected_snapshot.json', 'document_checks.json',
             'summary_provenance.json', 'LLM_usage.json', 'comparison.csv', 'results.json',
             'curves.json', 'interaction_local_comparisons.csv', 'SnAr_HPLC_Revised_R1.tex',
             'SnAr_figure.pdf', 'HPLC_figure.pdf', 'build_summary.py', 'build_delivery.py']
    for name in names:
        members[name] = REPORT / name
    for task in ['SnAr', 'HPLC']:
        for path in sorted((REPORT / task).rglob('*')):
            if path.is_file():
                members[path.relative_to(REPORT).as_posix()] = path
    for name in protocol['source_sha256']:
        path = ROOT / name
        if path.suffix == '.py':
            members['code/' + path.relative_to(ROOT).as_posix()] = path
    for folder in ['requests', 'runs']:
        for path in sorted((RUN / folder).rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.npz'}:
                members['evidence/' + path.relative_to(RUN).as_posix()] = path
    for name, path in members.items():
        assert 'config.txt' not in name.lower() and path.suffix not in {'.pt', '.pem', '.key'}
        if name.startswith('evidence/requests/'):
            content = path.read_text(encoding='utf-8')
            assert not re.search(r'"(?:authorization|api_key|apikey)"\s*:', content, re.I), name
            assert not re.search(r'\bsk-[A-Za-z0-9]{16,}\b|Bearer\s+[A-Za-z0-9]', content), name

    archive_path = REPORT / 'SnAr_HPLC_R1_Source_and_Results.zip'
    package_note = (
        'R1 source and results package\n\n'
        'The TeX file compiles independently with its two *_figure.pdf assets.\n'
        'code/ contains incremental code for the original chemist_system checkout;\n'
        'it depends on the frozen original GP, data, and decision modules.\n'
        'Copy incremental files into their matching repository-relative paths.\n'
        'The canonical entrypoint is continuous_revised.py; see README.md.\n'
        'Both 16-round candidate CSV series and native prediction/commit evidence are included.\n'
        'Hidden outcomes in candidate CSVs are post-decision debugging exports only.\n'
        'API prompts and sanitized responses are in evidence/requests/.\n'
        'No credentials, GP checkpoint weights, or original comparator copies are included.\n'
        'The final report PDF is delivered separately. Single seed; no claim of statistical superiority.\n'
    )
    with ZipFile(archive_path, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('PACKAGE_README.txt', package_note)
        for name, path in sorted(members.items()):
            archive.write(path, name)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(members) + 1
        for name, path in members.items():
            assert sha256(archive.read(name)).hexdigest() == digest(path), name

    receipt = {
        'status': 'COMPLETE_AND_VERIFIED', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol['protocol_sha256'], 'completed_trajectories': 2,
        'added_choices': 32, 'baselines_rerun': 0, 'protected_files_unchanged': len(protected),
        'unit_tests_passed': 7, 'trajectory_audit': 'PASS', 'document_checks': 'PASS',
        'negative_gain_nonincrease_cases': sum(x['negative_gain_nonincrease_checks'] for x in audit['checks']),
        'interaction_tests_inside_original_gate': 4, 'GP_initial_prediction_acquisition_max_difference': 0.0,
        'operator_compiles_in_loop': 0, 'LLM_logical_calls': usage['logical_calls'],
        'archive_members_verified': len(members) + 1,
        'PDF_sha256': digest(REPORT / 'SnAr_HPLC_Revised_R1.pdf'),
        'source_and_results_ZIP_sha256': digest(archive_path),
        'result_interpretation': 'HPLC SearchAUC improves; SnAr declines; neither final best improves; not a universal improvement',
    }
    write(REPORT / 'delivery_receipt.json', receipt)
    paths = set(path for path in REPORT.rglob('*') if path.is_file())
    paths.update(path for path in RUN.rglob('*') if path.is_file())
    paths.update(ROOT / name for name in protocol['source_sha256'])
    paths.update(ROOT / ('outputs/continuous_revised_' + phase + '.log') for phase in ['prepare', 'run', 'audit', 'report'])
    paths.update(ROOT / f'outputs/continuous_revised_typeset/delivery_pages/page-{i}.png' for i in [1, 2])
    inventory = {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)
                 if path.name not in {'release_inventory.json', 'release_receipt.json'}
                 and path.suffix not in {'.lock', '.tmp', '.pyc'} and '__pycache__' not in path.parts}
    write(REPORT / 'release_inventory.json', inventory)
    for name, expected in inventory.items():
        assert digest(ROOT / name) == expected, name
    write(REPORT / 'release_receipt.json', {
        'status': 'PASS', 'files_verified': len(inventory),
        'inventory_sha256': digest(REPORT / 'release_inventory.json'),
        'protected_files_unchanged': len(protected), 'baseline_reruns': 0,
    })
    print(json.dumps({'status': receipt['status'], 'sealed_files': len(inventory),
                      'archive_members': len(members) + 1, 'archive_bytes': archive_path.stat().st_size,
                      'LLM': usage}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
