"""Read-only verification and recoverable source/evidence snapshot for M1."""
from hashlib import sha256
from pathlib import Path
import json
import time
import zipfile

BASE = Path(__file__).resolve().parents[1]
OLD = BASE / 'research/suzuki_grounded_regional'
DEST = BASE / 'research/milestones/M1_Reizman_seed0_20261028'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def hashed(path):
    return sha256(path.read_bytes()).hexdigest()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else BASE / value


def main():
    if (DEST / 'milestone.json').exists():
        saved = read(DEST / 'milestone.json')
        assert hashed(DEST / saved['archive']['name']) == saved['archive']['sha256']
        for row in saved['files']:
            assert hashed(resolve(row['path'])) == row['sha256'], row['path']
        print(json.dumps({'status': 'ALREADY_SAVED_AND_VERIFIED', 'files': len(saved['files'])}))
        return
    receipt = read(OLD / 'release_receipt.json')
    inventory = read(OLD / 'release_inventory.json')
    protocol = read(OLD / 'protocol.json')
    assert receipt['integrity_status'] == 'PASS' and receipt['completed'] == 48
    assert hashed(OLD / 'release_inventory.json') == receipt['inventory_sha256']
    paths = {}
    def add(path, expected=None):
        path = resolve(path).resolve()
        assert path.is_relative_to(BASE.resolve())
        key = path.relative_to(BASE).as_posix()
        value = hashed(path)
        if expected is not None:
            assert value == expected, key
        paths[key] = {'path': key, 'bytes': path.stat().st_size, 'sha256': value}
    for row in inventory['files']:
        add(row['path'], row['sha256'])
    for path, value in {**protocol['input_sha256'], **protocol['protected_files']}.items():
        add(path, value)
    add(OLD / 'release_receipt.json')
    add(OLD / 'release_inventory.json')
    # Keep exact baseline decisions in the snapshot, rather than aggregate values only.
    for case in (1, 2, 3, 4):
        add(f'data/runs/suzuki_mixed_kalman/base/c{case}_s0/state.json')
        for regime in protocol['regimes']:
            for method in ('LGBO_region', 'LLINBO_matched'):
                add(f'data/runs/suzuki_mixed_comparison/runs/c{case}_s0_{regime}_{method}/state.json')
    caches = {}
    for name, expected in receipt['cache_trees'].items():
        folder = BASE / 'data/runs/suzuki_grounded_regional' / name
        rows = [{'path': p.relative_to(folder).as_posix(), 'bytes': p.stat().st_size, 'sha256': hashed(p)}
                for p in sorted(folder.rglob('*')) if p.is_file()]
        actual = {'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                  'sha256': sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()}
        assert actual == expected, name
        caches[name] = actual
    DEST.mkdir(parents=True)
    note = '''# M1：Reizman Suzuki，seed=0，修复方案完整验证

保存日期：2026-10-28。

完成 Case 1–4 × 四种知识条件 × R1/R2/R3，共 48 条完整新轨迹、768 次新增查询、960 个逻辑观测；141 次真实 deepseek-flash 刷新全部有效。36 项测试、768 次独立选择复核、1775 条引用事实通过；循环内编译为 0。

主测试 Case 2–4、四条件平均 SearchAUC：原 G3=68.7523，R1=68.3683，R2=69.0376，R3=68.6610，仅预算门限=69.4085。R2 相对原 G3 有小幅收益，但尚未超过仅预算门限；R3 新区幅度折扣没有通过整体增量检验。不把实现通过写成研究优于基线。

冻结选择：保留具名输入和数值证据校验；R2 作为继续检验的候选；R3 不默认启用。单 seed 描述性结论，人工知识压力条件使用参考结果信息。

ZIP 保存代码、协议、结果、逐轮决策、可见 LLM 请求/响应、固定输入及基线决策。缓存留在原实验目录，已按现有缓存树哈希逐一校验；不复制环境、凭据或缓存。此 ZIP 是源代码和证据快照，运行依赖仍使用工作区既有环境及官方代码依赖。

第二实验使用独立 data/runs 与 research 命名空间接入 Perera Suzuki 和 Doyle Buchwald–Hartwig；不覆盖 M1 数据、代码和报告。里程碑完整清单及 ZIP 的 SHA256 见 milestone.json。
'''
    (DEST / 'MILESTONE.md').write_text(note, encoding='utf8')
    archive = DEST / 'M1_Reizman_seed0_20261028.zip'
    with zipfile.ZipFile(archive.with_suffix('.tmp'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for key in sorted(paths):
            target.write(BASE / key, arcname=key)
        target.write(DEST / 'MILESTONE.md', arcname='MILESTONE.md')
    archive.with_suffix('.tmp').replace(archive)
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None and len(target.namelist()) == len(paths) + 1
    saved = {'milestone': 'M1_Reizman_seed0_20261028', 'status': 'SAVED_AND_VERIFIED',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'parent_protocol_sha256': protocol['protocol_sha256'], 'release_receipt_sha256': hashed(OLD / 'release_receipt.json'),
        'results_sha256': receipt['results_sha256'], 'planned': 48, 'completed': 48,
        'archive': {'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': hashed(archive)},
        'files': [paths[key] for key in sorted(paths)], 'cache_trees_verified': caches,
        'cache_copied': False, 'credentials_copied': False,
        'scope': 'Complete source/evidence snapshot; runtime environments and vendor dependencies remain in original workspace.'}
    (DEST / 'milestone.json').write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': saved['status'], 'files': len(paths), 'archive_bytes': archive.stat().st_size,
        'archive_sha256': saved['archive']['sha256'], 'cache_files_verified': sum(v['files'] for v in caches.values())}))


if __name__ == '__main__':
    main()
