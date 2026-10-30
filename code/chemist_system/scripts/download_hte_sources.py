"""Download pinned original HTE workbooks, without changing the BO environment."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
from pathlib import Path
import json
import time
import httpx

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / 'data/benchmarks/reaction_transfer/raw'
COMMIT = 'd9e6b87ce1b881978490d68bfc00021e3b48127a'
FILES = (
    ('Perera_Suzuki.xlsx', 'data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx', 'cd1b57e308fbc584666af0d7a26fe3041c14caaf', 628918),
    ('Doyle_Buchwald_Hartwig.xlsx', 'data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx', '36487d7ed4f9d5c0d24d3684d7a219a75be5e64f', 2134762),
)


def download(spec):
    name, relative, expected, size = spec
    url = f'https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/{COMMIT}/{relative}'
    target = ROOT / name
    if target.exists():
        content = target.read_bytes()
    else:
        answer = httpx.get(url, timeout=120, follow_redirects=False)
        answer.raise_for_status()
        content = answer.content
    assert len(content) == size and content.startswith(b'PK')
    blob = sha1(b'blob ' + str(len(content)).encode('ascii') + b'\0' + content).hexdigest()
    assert blob == expected
    if not target.exists():
        target.write_bytes(content)
    return {'name': name, 'source_url': url, 'repository': 'rxn4chemistry/rxn_yields',
        'commit': COMMIT, 'original_path': relative, 'bytes': size,
        'git_blob_sha1': blob, 'sha256': sha256(content).hexdigest(),
        'scope': 'Raw workbooks retained unchanged; sheet selection and row exclusions are recorded separately.'}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(download, FILES))
    manifest = {'status': 'DOWNLOADED_AND_GIT_BLOB_VERIFIED', 'source_commit': COMMIT,
        'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'files': rows}
    path = ROOT / 'source_manifest.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf8'))
        assert old['files'] == rows
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': manifest['status'], 'files': [{'name': v['name'], 'bytes': v['bytes'], 'sha256': v['sha256']} for v in rows]}))


if __name__ == '__main__':
    main()
