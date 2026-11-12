"""Compile the submitted TeX sources into a separate directory with Tectonic."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/hypothesis_scope_jcim"
STEMS = ("si", "main", "si_zh", "main_zh")
FONT_FILES = (
    "arial.ttf", "arialbd.ttf", "arialbi.ttf", "ariali.ttf",
    "cour.ttf", "courbd.ttf", "courbi.ttf", "couri.ttf",
    "simhei.ttf", "simkai.ttf", "simsun.ttc",
    "times.ttf", "timesbd.ttf", "timesbi.ttf", "timesi.ttf",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", help="Tectonic executable. Defaults to tectonic on PATH.")
    parser.add_argument("--only-cached", action="store_true")
    parser.add_argument("--english-only", action="store_true")
    parser.add_argument("--fonts-dir", type=Path, help="Optional local font files for the Chinese TeX.")
    parser.add_argument("--out", type=Path, help="Fresh build directory. Defaults to output/.")
    args = parser.parse_args()
    executable = args.tectonic or shutil.which("tectonic")
    if not executable:
        parser.error("Tectonic is required. Install it or use --tectonic.")
    executable = str(Path(executable).resolve())
    if args.out is not None:
        out = args.out.resolve()
        if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "output"):
            parser.error("Use a fresh directory under output/ or outside the submission.")
        out.mkdir(parents=True, exist_ok=False)
    else:
        parent = ROOT / "output"
        parent.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="tex-", dir=parent))
    stems = STEMS[:2] if args.english_only else STEMS
    for stem in stems:
        shutil.copy2(PAPER / f"{stem}.tex", out)
    for name in ("figures", "generated", "generated_zh"):
        shutil.copytree(PAPER / name, out / name)
    shutil.copy2(PAPER / "references.bib", out)
    if args.fonts_dir and not args.english_only:
        font_out = out / "fonts"
        font_out.mkdir()
        for name in FONT_FILES:
            shutil.copy2(args.fonts_dir / name, font_out)
    options = ["--keep-logs", "--keep-intermediates"]
    if args.only_cached:
        options.append("--only-cached")
    for stem in stems:
        completed = subprocess.run(
            [executable, "-X", "compile", f"{stem}.tex", *options],
            cwd=out, capture_output=True, check=False,
        )
        (out / f"{stem}.build.txt").write_bytes(completed.stdout + completed.stderr)
        completed.check_returncode()
        if not (out / f"{stem}.pdf").is_file():
            raise ValueError(f"Missing PDF for {stem}.")
        print(f"PASS {stem}.pdf", flush=True)
    print(out)


if __name__ == "__main__":
    main()
