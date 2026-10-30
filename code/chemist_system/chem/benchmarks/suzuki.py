"""Run `python -m chem.benchmarks.suzuki --help` for the isolated study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"


def main():
    parser = argparse.ArgumentParser(description="LLINBO / LGBO / official HEBO, only Suzuki Cases 1-4")
    parser.add_argument("--root", type=Path, default=Path("data/runs/suzuki_baselines"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export-oracles")
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--methods", nargs="+", choices=("LLINBO", "LGBO", "HEBO"), default=("LLINBO", "LGBO", "HEBO"))
    pending_parser = sub.add_parser("pending")
    pending_parser.add_argument("--compact", action="store_true")
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--answers", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--verify-selections", action="store_true")
    args = parser.parse_args()
    if args.command == "export-oracles":
        # Existing frozen exports are immutable, including ZIP timestamps/hashes.
        if (args.root / "protocol.json").exists():
            from hashlib import sha256
            p = json.loads((args.root / "protocol.json").read_text(encoding="utf-8"))
            for rel, expected in p["oracle_files_sha256"].items():
                if sha256((args.root / rel).read_bytes()).hexdigest() != expected:
                    raise ValueError("frozen oracle changed; refusing to overwrite or silently repair it")
            result = json.loads((args.root / "oracles/qualification.json").read_text(encoding="utf-8"))["cases"]
        else:
            from .suzuki_domain import export_oracles
            result = export_oracles(args.root / "oracles")
    elif args.command == "report":
        from .suzuki_report import report
        result = report(args.root, verify_selections=args.verify_selections)
    else:
        from . import suzuki_runner as runner
        if args.command == "prepare":
            p = runner.prepare(args.root)
            result = {"protocol_sha256": p["protocol_sha256"], "planned_runs": 60}
        elif args.command == "run":
            result = runner.run(args.root, args.workers, args.methods)
        elif args.command == "pending":
            result = runner.pending(args.root, args.compact)
        else:
            result = runner.submit(args.answers, args.root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
