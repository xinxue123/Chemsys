"""All prescribed tie seeds must pass; never select a favorable subset."""
import argparse
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from chem.benchmarks import reasoning_bo_data as data
OUT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');a=ap.parse_args()
    rows=[];missing=[];sources=[];normalized=[]
    for seed in [0,1,2]:
        variant=a.group+'_t'+str(seed);folder=OUT/variant
        if not (folder/'release_receipt.json').exists():missing.append(variant);continue
        release=data.read(folder/'release_receipt.json');inv=data.read(folder/'release_inventory.json')
        assert data.filehash(folder/'release_inventory.json')==release['inventory_sha256']
        for f in inv['files']:assert data.filehash(f['path'])==f['sha256'],f['path']
        result=data.read(folder/'results.json');protocol=data.read(folder/'protocol.json')
        assert release['status']=='PASS_COMPLETE' and data.read(folder/'validation_receipt.json')['choices']==32
        assert protocol['seed']==0 and protocol['initial_queries']==4 and protocol['additional_queries']==16
        assert protocol['config']['tie_seed']==seed
        config={k:v for k,v in protocol['config'].items() if k not in ('id','tie_seed')}
        normalized.append(config)
        source={k:v for k,v in protocol['source_sha256'].items() if '/plans/' not in k}
        sources.append(source)
        rows.extend({**r,'tie_seed':seed} for r in result['metrics'])
    if normalized:assert all(c==normalized[0] for c in normalized)
    if sources:assert all(s==sources[0] for s in sources)
    checks={}
    for task,floor,best in [('Perera_Suzuki',98.,99.),('Doyle_Buchwald_Hartwig',91.4567435325,98.28722602)]:
        rs=[r for r in rows if r['task']==task]
        if not rs:continue
        auc=[r['SearchAUC'] for r in rs];y=[r['final_best_percent'] for r in rs]
        checks[task]={'count':len(rs),'AUC_min':min(auc),'AUC_mean':statistics.mean(auc),'AUC_max':max(auc),
            'AUC_range':max(auc)-min(auc),'minimum_final_best_percent':min(y),
            'floor_all_pass':all(v+1e-9>=floor for v in auc) and all(v+1e-9>=best for v in y),
            'range_pass':max(auc)-min(auc)<=1.+1e-9}
    passed=not missing and len(checks)==2 and all(c['count']==3 and c['floor_all_pass'] and c['range_pass'] for c in checks.values())
    result={'group':a.group,'status':'STABILITY_PASS' if passed else 'INCOMPLETE' if missing else 'STABILITY_FAIL',
        'fixed_tie_seeds':[0,1,2],'optimizer_seed':0,'missing':missing,'checks':checks,'metrics':rows,
        'scope':'Three prescribed tie-order checks on known seed0 benchmarks; not independent initialization/generalization evidence.'}
    data.write(OUT/(a.group+'_stability.json'),result)
    text=[f'**{a.group}：{result["status"]}**','',
        '优化seed=0，初始化与4＋16预算不变。全部并列seed0/1/2均须达标；不筛选其中最好的一次。','',
        '| 数据集 | 并列seed | SearchAUC | 最终最佳/% | 首达实验序号 |','|---|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['task'],r['tie_seed'])):
        text.append(f"| {r['task']} | {r['tie_seed']} | {r['SearchAUC']:.4f} | {r['final_best_percent']:.4f} | {r['first_final_best_experiment']} |")
    text += ['', '验收：Perera每次AUC≥98、最佳≥99%；Doyle每次AUC≥91.4567435325、最佳≥98.28722602%；每数据集AUC极差≤1。',
        '', '未完成：'+', '.join(missing) if missing else '全部预定轨迹完成。','',
        '结果来自已分析基准上的开发。即使三次通过，也只支持这里规定的有限排序稳定性。前序负面版本完整保留。']
    (OUT/(a.group+'_stability.md')).write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
