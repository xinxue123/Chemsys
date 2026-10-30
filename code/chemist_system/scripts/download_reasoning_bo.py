"""Pin upstream algorithm/data; sanitize auxiliary external telemetry literals."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
from pathlib import Path
import json
import re
import httpx

BASE = Path(__file__).resolve().parents[1]
DEST = BASE / 'third_party/reasoning_bo_official'
META = BASE / 'research/reaction_transfer/reasoning_bo_upstream_manifest.json'


def main():
    metadata = json.loads(META.read_text(encoding='utf8'))
    DEST.mkdir(exist_ok=True)
    def fetch(spec):
        path = DEST / spec['path']
        url = f"https://raw.githubusercontent.com/{metadata['repository']}/{metadata['commit']}/{spec['path']}"
        existing = path.exists()
        if existing:
            return None
        response = httpx.get(url, timeout=120, follow_redirects=False)
        response.raise_for_status()
        raw = response.content
        blob = sha1(b'blob '+str(len(raw)).encode('ascii')+b'\0'+raw).hexdigest()
        assert blob == spec['git_blob_sha1'] and len(raw) == spec['bytes']
        sanitizations = []
        if path.suffix in ('.py', '.json', '.md'):
            text = raw.decode('utf8')
            # No telemetry endpoint or published token is reused by the adapter.
            for label, pattern in [('external_webhook', r'https://open\.(?:feishu|larksuite)\.(?:cn|com)/open-apis/bot/v2/hook/[^\s\"\x27]+'),
                                   ('external_api_token', r'sk-[A-Za-z0-9_-]{12,}')]:
                text, n = re.subn(pattern, '[REDACTED_'+label.upper()+']', text)
                if n:
                    sanitizations.append({'kind': label, 'occurrences': n})
            raw = text.encode('utf8')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {**spec, 'url': url, 'local_sha256': sha256(raw).hexdigest(), 'sanitizations': sanitizations}
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(fetch, metadata['files']))
    provenance = DEST / 'SOURCE_PROVENANCE.json'
    if provenance.exists():
        old = json.loads(provenance.read_text(encoding='utf8'))
        assert old['commit'] == metadata['commit']
        for r in old['files']:
            assert sha256((DEST/r['path']).read_bytes()).hexdigest() == r['local_sha256']
    else:
        assert all(row is not None for row in rows)
        provenance.write_text(json.dumps({**{k:v for k,v in metadata.items() if k!='files'}, 'files': rows,
            'omitted': ['scripts/start_optimization.sh (external telemetry credential)'],
            'execution': 'Only reviewed optimizer/reasoner functions are called; no SwanLab, webhook, or DB connection is launched.'}, indent=2), encoding='utf8')
    print(json.dumps({'status': 'PINNED', 'commit': metadata['commit'], 'files': len(metadata['files']),
        'sanitized_files': [r['path'] for r in rows if r and r['sanitizations']]}))


if __name__ == '__main__':
    main()
