"""Verify and install the data attachment for the versioned evidence checkout."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_DATA.json"
BLOCK_SIZE = 1024 * 1024


def stream_hash(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(BLOCK_SIZE), b""):
        digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    with path.open("rb") as handle:
        return stream_hash(handle)


def safe_member(info):
    rel = info.filename
    pure = PurePosixPath(rel)
    return (
        pure.parts and pure.as_posix() == rel and not rel.startswith("/")
        and ".." not in pure.parts and "\\" not in rel and ":" not in rel
        and not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16)
        and pure.parts[0] in {"archives", "experiments", "evidence"}
        and pure.suffix in {".json", ".csv", ".npz", ".zip"}
    )


def install(archive_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if archive_path.stat().st_size != manifest["bytes"] or file_hash(archive_path) != manifest["sha256"]:
        raise ValueError("The Release archive differs from RELEASE_DATA.json.")
    expected = {row["path"]: row for row in manifest["files"]}
    with ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) != len(expected) or {row.filename for row in entries} != set(expected):
            raise ValueError("The Release archive member list differs from RELEASE_DATA.json.")
        pending = []
        for info in entries:
            rel = info.filename
            row = expected[rel]
            if not safe_member(info) or info.file_size != row["bytes"]:
                raise ValueError(f"Unexpected data member {rel!r}")
            destination = ROOT / rel
            if destination.resolve() != destination:
                raise ValueError(f"Destination is redirected for {rel!r}")
            if destination.exists():
                if not destination.is_file() or file_hash(destination) != row["sha256"]:
                    raise ValueError(f"Existing file differs from the Release data at {rel!r}")
                continue
            with archive.open(info) as handle:
                if stream_hash(handle) != row["sha256"]:
                    raise ValueError(f"Data member checksum failed for {rel!r}")
            pending.append((info, destination))
        for info, destination in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("xb") as target:
                try:
                    shutil.copyfileobj(source, target, length=BLOCK_SIZE)
                except BaseException:
                    target.close()
                    destination.unlink()
                    raise
    return {"status": "PASS", "files": len(expected), "installed": len(pending)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(install(args.archive.resolve())))


if __name__ == "__main__":
    main()

