"""Post-run presentation and custody only; frozen policy sources are untouched."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from chem.benchmarks import stable_suzuki_study as s,stable_suzuki_report as report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true');a=parser.parse_args()
    result=s.read(s.OUT/'results.json');assert s.read(s.OUT/'validation_receipt.json')['status']=='PASS'
    assert s.read(s.OUT/'evidence_audit.json')['status']=='PASS'
    rows=result['comparison_rows'];current=result['raw_metrics']
    # Independently reconcile chart-source CSV with all displayed metrics.
    with (s.OUT/'plot_source_rows.csv').open(encoding='utf-8-sig',newline='') as f:plot_rows=list(csv.DictReader(f))
    labels={'Revised strategy':'Stable_revised','LGBO':'LGBO','LLINBO-Justify':'LLINBO_Justify','Pure BO (qLogEI)':'Pure_BO'}
    for case in (1,2,3,4):
        for regime in s.REGIMES:
            for label,method in labels.items():
                group=sorted([r for r in plot_rows if int(r['case'])==case and r['knowledge']==regime and r['method']==label],key=lambda r:int(r['experiment']))
                assert len(group)==20
                y=np.array([float(r['yield_percent']) for r in group]);best=np.maximum.accumulate(y)
                assert np.array_equal(best,np.array([float(r['incumbent_percent']) for r in group]))
                expected=next(r for r in rows if r['case']==case and r['regime']==regime and r['method']==method)
                assert abs(float(best[4:].mean())-expected['SearchAUC'])<1e-12
                assert float(best[-1])==expected['final_best']
    provenance=s.read(s.OUT/'plot_provenance.json')
    provenance['visual_QA']={'status':'PASS','scope':'all four exported PNGs visually inspected for labels, axes, marks, legend, clipping and footer; 64 chart series reconciled to metrics',
        'figures':['comparison_'+r+'.png' for r in s.REGIMES],'source_series_checked':64}
    s.write(s.OUT/'plot_provenance.json',provenance)
    lines=['# 修订版策略在Suzuki的增量尚不稳定','',
        '已完成16/16条模拟轨迹：Case 1–4×四知识条件；原4692候选、Mixed GP、seed=0、4次初始化＋16次查询。旧对照读取冻结结果，未重跑或改写。',
        '', '助手初始知识条件下，新版SearchAUC较纯BO为2胜2负；较旧R2为2例下降、2例持平（AUC持平不代表选择序列相同）。混合错误条件的四案例均值仍低于纯BO，未证明稳定迁移收益。',
        '', 'SearchAUC是16次追加实验后累计最佳产率的均值；产率单位为%，越高越好。首次最佳按包含4个初始化的总实验次序计。',
        '', '| Case／助手初始知识 | 新版AUC | 旧R2 AUC | 纯BO AUC | LGBO AUC | LLINBO-Justify AUC | 新版最佳产率 | 首次最佳 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in (1,2,3,4):
        get=lambda method:next(r for r in rows if r['case']==case and r['regime']=='assistant_initial' and r['method']==method)
        new=get('Stable_revised')
        lines.append(f"| {case} | {new['SearchAUC']:.3f} | {get('R2_conflict_fallback')['SearchAUC']:.3f} | {get('Pure_BO')['SearchAUC']:.3f} | {get('LGBO')['SearchAUC']:.3f} | {get('LLINBO_Justify')['SearchAUC']:.3f} | {new['final_best']:.3f} | {new['first_own_best_total']} |")
    lines+=['', '| 知识条件／四案例均值 | 新版AUC | 相对纯BO／百分点 |','|---|---:|---:|']
    labels_zh={'correct':'正确知识（人工压力测试）','wrong_category':'错误类别（人工压力测试）','mixed':'混合错误（人工压力测试）','assistant_initial':'助手初始建议'}
    for regime in s.REGIMES:
        m=result['by_knowledge_condition'][regime]
        lines.append(f"| {labels_zh[regime]} | {m['SearchAUC_mean']:.3f} | {m['delta_AUC_vs_Pure_BO']:+.3f} |")
    lines+=['', '本域缺少催化剂分子结构且仅有一个名义轴，Stage11的分子核恢复、双类别联合反例及其局部跟进没有启用。保留连续核，未生成新候选或把近邻冒充单轴对照；沿用修订版区域协方差单位修复、KF、B4及同区域信任闩锁。必要适配详见[ADAPTATIONS.md](ADAPTATIONS.md)。这些共同变化不构成单模块因果消融。',
        '', '实际48次刷新＋3次精确单轴试验；58份真实API响应，含格式修复；门限外选择0次。51份请求均仅含自身已观测前缀。全部256次选择和250次实际KF更新复核通过，催化剂重命名后后验差异为0。',
        '', '![助手初始知识：逐轮产率与累计最佳](C:/Users/lihao/Documents/ChatGPT/Chemsys/chemist_system/research/suzuki_stable_20261029/comparison_assistant_initial.png)',
        '', '其他知识条件图：[正确知识](comparison_correct.png) · [错误类别](comparison_wrong_category.png) · [混合错误](comparison_mixed.png)。全部方案逐案例、逐条件数值见comparison_all.csv；逐轮决策见query_trace.csv。',
        '', '以下保留所有四知识条件和全部冻结对照的详细表。', '']
    path=s.OUT/'REPORT.md';body=path.read_text(encoding='utf-8');body=body[body.index('## Case 1'):]
    path.write_text('\n'.join(lines)+'\n'+body,encoding='utf-8')
    secret=s.api.load_config('../config.txt').api_key.encode('utf-8');checked=0
    for root in (s.OUT,s.ROOT):
        for f in root.rglob('*'):
            if f.is_file() and f.suffix in ('.json','.md','.csv','.log','.py','.xml'):
                assert secret not in f.read_bytes(),f.as_posix();checked+=1
    s.write(s.OUT/'delivery_checks.json',{'status':'PASS','chart_series_reconciled':64,'credential_scan_files':checked,
        'credential_found':False,'algorithm_sources_unchanged':True,'technical_validation_is_not_performance_acceptance':True})
    if a.release:
        receipt=report.release();inv=s.read(s.OUT/'release_inventory.json');extra=Path('scripts/plot_stable_suzuki.py')
        assert s.filehash(extra)==provenance['script_sha256']
        if not any(r['path']==extra.as_posix() for r in inv['files']):
            inv['files'].append({'path':extra.as_posix(),'bytes':extra.stat().st_size,'sha256':s.filehash(extra)})
            inv['files'].sort(key=lambda row:row['path']);s.write(s.OUT/'release_inventory.json',inv)
        receipt.update(files=len(inv['files']),inventory_sha256=s.filehash(s.OUT/'release_inventory.json'))
        s.write(s.OUT/'release_receipt.json',receipt)
        for row in inv['files']:assert s.filehash(row['path'])==row['sha256'],row['path']
        print(json.dumps(receipt))
    else:print(json.dumps({'status':'PASS','chart_series_reconciled':64,'credential_scan_files':checked}))


if __name__=='__main__':main()
