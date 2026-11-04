"""Metadata writer compatibility for the registered, unchanged experiment adapter.

The numerical preflight completed24/24 pairs before its parent summary writer
referenced old.write, which does not exist. Supply the already-native writer;
this touches neither a registered source byte nor a scientific calculation.
"""
from pathlib import Path
from unittest.mock import patch
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/chemist_system'), str(ROOT / 'code/experiment_tools')]
import run_v13_initialization_seeds as cli
from v13_initialization_seeds import study
from state_split_v13_experiment import transport, study as frozen


def main():
    transport.write_once(study.directory(ROOT) / 'metadata_writer_adapter.json', dict(
        purpose='Bind_missing_parent_summary_writer_to_existing_native_atomic_writer',
        registered_source_modified=False, GP_or_policy_modified=False,
        helper_sha256=transport.file_sha(__file__)))
    with patch.object(study.old, 'write', frozen.write, create=True):
        if len(sys.argv) > 1 and sys.argv[1] == 'recover-preflight':
            p = study.check(ROOT, True)
            if any((study.base(ROOT, *job) / 'requests').exists() for job in study.JOBS):
                raise ValueError('Preflight recovery must precede every new LLM request')
            rows = [transport.read(study.base(ROOT, *job) / 'preflight/receipt.json') for job in study.JOBS]
            if len(rows) != 24 or any(r['status'] != 'PASS_EXACT' or r['baseline_queries_audited'] != 16 for r in rows):
                raise ValueError('All numerical preflights must already be complete')
            result = dict(status='PASS', protocol_sha256=p['protocol_sha256'], rows=rows,
                numerical_cache_copy_summary='original parent summary failed after copying and24 numerical checks',
                metadata_writer_recovery=True, new_queries=0, LLM_calls=0)
            frozen.write(study.directory(ROOT) / 'preflight.json', result)
            print(json.dumps(dict(status='PASS', pairs=24, BO_queries_audited=384,
                new_GP_fits_in_recovery=0, new_queries=0, LLM_calls=0)))
        else:
            cli.main()


if __name__ == '__main__':
    main()
