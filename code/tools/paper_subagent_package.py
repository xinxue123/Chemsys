"""Create a typesetting/data archive only after the complete scientific gate.

An explicit allowlist excludes credentials, prompts, runtime files and old
manuscript artifacts. This is a local package, not a public upload.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/v12_subagent'
REPORT = ROOT / 'experiments/reports/paper_v12_subagent'
ARCHIVE = ROOT / 'paper/v12_subagent_source.zip'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def inputs():
    selected = {}
    for pattern in ('*.tex', '*.bib', 'README.txt', 'verify_results.py', 'evidence_manifest.json',
                    'generated/*.tex', 'figures/*.pdf', 'figures/*.png', 'data/*.csv', 'data/*.json', 'data/README.txt'):
        for path in PAPER.glob(pattern):
            selected['v12_subagent/' + path.relative_to(PAPER).as_posix()] = path
    for name in ('audit.json', 'manuscript_validation.json', 'compile_receipt.json', 'pdf_qa.json',
                 'text_generation.json', 'figure_generation.json', 'mechanism_generation.json', 'catalogue_receipt.json', 'catalogue_field_check.json', 'pdf_completeness_regression.json', 'contrast_trace_review.json', 'hplc_panel_review_68.json'):
        selected['v12_subagent/audit_receipts/' + name] = REPORT / name
    selected['v12_subagent/protocol/protocol.json'] = ROOT / 'experiments/runs/paper_v12_subagent/protocol.json'
    for name in ('paper_v12_subagent_protocol_20261101.txt', 'paper_v12_subagent_decision_instructions.txt'):
        selected['v12_subagent/protocol/' + name] = ROOT / 'docs' / name
    for path in REPORT.glob('incident_r000176*.json'):
        selected['v12_subagent/audit_receipts/' + path.name] = path
    for path in (REPORT/'transport_continuations').glob('*/*.json'):
        selected['v12_subagent/audit_receipts/' + path.relative_to(REPORT).as_posix()] = path
    return dict(sorted(selected.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true', help='List package candidates without producing an archive or claiming completion.')
    args = parser.parse_args()
    if args.inspect:
        paths = inputs()
        print(json.dumps({'archive_created': False, 'candidate_files': len(paths),
                          'missing': [name for name, path in paths.items() if not path.is_file()],
                          'candidate_bytes': sum(path.stat().st_size for path in paths.values() if path.is_file())}, indent=2))
        return
    subprocess.run([sys.executable, str(ROOT / 'code/tools/paper_subagent_validate.py'), '--require-complete'], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / 'code/tools/paper_subagent_contrast_review.py'), '--require-complete'], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / 'code/tools/paper_subagent_hplc_review.py')], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(PAPER / 'verify_results.py'), '--require-complete'], check=True, cwd=PAPER)
    paths = inputs()
    hashes = {name: digest(path) for name, path in paths.items()}
    validation = json.loads((REPORT / 'manuscript_validation.json').read_text(encoding='utf-8'))
    if not validation['completion_gate_satisfied']:
        raise ValueError('Complete evidence gate is required before creating the release archive')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'content_sha256': hashes,
                'validation': validation, 'pdf_sha256': {name: digest(PAPER / name) for name in ('main.pdf', 'supplement.pdf')},
                'scope': 'Portable LaTeX sources, figures, exported results, arithmetic verifier and audit receipts. The native experiment archive is a separate project dependency.'}
    temporary = ARCHIVE.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in paths.items():
            archive.write(path, name)
        archive.writestr('v12_subagent/package_manifest.json', json.dumps(manifest, indent=2))
    if {name: digest(path) for name, path in paths.items()} != hashes:
        raise ValueError('Package inputs changed during creation; temporary archive is not a release')
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        for name, expected in hashes.items():
            if sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Archive byte mismatch: ' + name)
    temporary.replace(ARCHIVE)
    receipt = {'status': 'PASS_COMPLETE_LOCAL_PACKAGE', 'archive': ARCHIVE.relative_to(ROOT).as_posix(),
               'archive_sha256': digest(ARCHIVE), 'files': len(hashes) + 1, 'manifest': manifest}
    (REPORT / 'package_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'manifest'}, indent=2))


if __name__ == '__main__':
    main()
