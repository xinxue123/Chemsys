"""Rebuild the saved analyses in a disposable copy and compare with the submission."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("development", "heldout", "screen", "first_points", "knowledge")
TOLERANCE = 1e-9


def check_value(expected, observed, location):
    if isinstance(expected, dict) and isinstance(observed, dict):
        if expected.keys() != observed.keys():
            raise ValueError(f"{location}: JSON fields differ")
        for key, value in expected.items():
            check_value(value, observed[key], f"{location}.{key}")
    elif isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            raise ValueError(f"{location}: different number of values")
        for i, (left, right) in enumerate(zip(expected, observed)):
            check_value(left, right, f"{location}[{i}]")
    elif isinstance(expected, float) and type(observed) in (int, float):
        if not math.isclose(expected, observed, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise ValueError(f"{location}: {expected!r} != {observed!r}")
    elif type(expected) is not type(observed) or expected != observed:
        raise ValueError(f"{location}: {expected!r} != {observed!r}")


def compare_file(expected, observed, rel):
    if expected.read_bytes() == observed.read_bytes():
        return "exact"
    if observed.suffix == ".json":
        left = json.loads(expected.read_text(encoding="utf-8-sig"))
        right = json.loads(observed.read_text(encoding="utf-8-sig"))
        check_value(left, right, rel)
    elif observed.suffix == ".csv":
        with expected.open(encoding="utf-8-sig", newline="") as a:
            with observed.open(encoding="utf-8-sig", newline="") as b:
                left, right = list(csv.reader(a)), list(csv.reader(b))
        if len(left) != len(right) or left[0] != right[0]:
            raise ValueError(f"{rel}: CSV header or row count differs")
        for i, (a, b) in enumerate(zip(left[1:], right[1:]), 2):
            if len(a) != len(b):
                raise ValueError(f"{rel}:{i}: different number of columns")
            for j, (x, y) in enumerate(zip(a, b), 1):
                if x == y:
                    continue
                try:
                    a_float, b_float = float(x), float(y)
                except ValueError as error:
                    raise ValueError(f"{rel}:{i}:{j}: {x!r} != {y!r}") from error
                check_value(a_float, b_float, f"{rel}:{i}:{j}")
    else:
        raise ValueError(f"{rel}: changed non-table output")
    return "equivalent"


def make_report_directory(out):
    if out is not None:
        out = out.resolve()
        if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "output"):
            raise ValueError("Use a fresh directory under output/ or outside the submission.")
        out.mkdir(parents=True, exist_ok=False)
        return out
    parent = ROOT / "output"
    parent.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="reproduction-", dir=parent))


def rebuild(part, out):
    report_dir = make_report_directory(out)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    with tempfile.TemporaryDirectory(prefix="saved-inputs-", dir=report_dir) as tmp:
        work = Path(tmp)
        for name in ("archives", "data", "experiments", "results", "code"):
            shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__"))
        (work / "tools").mkdir()
        shutil.copy2(ROOT / "tools/rebuild_results.py", work / "tools/rebuild_results.py")
        before = {
            p.relative_to(work): p.stat().st_mtime_ns
            for p in (work / "results").rglob("*") if p.is_file()
        }
        completed = subprocess.run(
            [sys.executable, "-B", str(work / "tools/rebuild_results.py"), part],
            cwd=work, env=env, check=False, capture_output=True, encoding="utf-8",
        )
        (report_dir / "rebuild.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        completed.check_returncode()
        jobs = PARTS if part == "all" else (part,)
        if any(f"PASS {job}" not in completed.stdout.splitlines() for job in jobs):
            raise ValueError("A requested saved analysis did not complete.")
        checked = {}
        for p in sorted((work / "results").rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(work)
            if before.get(rel) == p.stat().st_mtime_ns:
                continue
            reference = ROOT / rel
            if not reference.is_file():
                raise ValueError(f"{rel}: rebuilt output has no frozen reference")
            checked[rel.as_posix()] = compare_file(reference, p, rel.as_posix())
            dest = report_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        if not checked:
            raise ValueError("No rebuilt result was detected.")
    report = {
        "status": "PASS",
        "analyses": list(jobs),
        "relative_and_absolute_numeric_tolerance": TOLERANCE,
        "result_files": checked,
        "new_model_calls": 0,
        "new_gp_fits": 0,
        "new_optimization_queries": 0,
    }
    (report_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {len(checked)} results across {len(jobs)} analyses")
    print(report_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=[*PARTS, "all"], default="all", nargs="?")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rebuild(args.part, args.out)


if __name__ == "__main__":
    main()
