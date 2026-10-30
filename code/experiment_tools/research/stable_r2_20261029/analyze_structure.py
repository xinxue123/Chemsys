"""Diagnose sealed structure_t0 choices without querying counterfactual outcomes."""
from pathlib import Path
import json
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from chem.benchmarks import reasoning_bo_data as data


def main():
    out=Path(__file__).resolve().parent;root=Path('data/runs/stable_r2_20261029/structure_t0')
    pool=data.load_pool('Perera_Suzuki');p=next((root/'runs').glob('Perera*/state.json'));s=data.read(p)
    axis='Nucleophile_SMILES';levels=pool['levels'][axis];code=np.array([levels.index(x[axis])+1 for x in pool['points']])
    ix={k:i for i,k in enumerate(pool['ids'])};rows=[]
    for d in s['decisions']:
        t=d['iteration'];path=p.parent/'rounds'/f'{t:02d}_candidates.npz'
        with np.load(path) as z:
            av=z['available'];E=z['eligible'];base=z['base_qLogEI'];maximum=base[av].max();selected=ix[d['selected_id']]
            row={'query':t,'own_observed_yield_percent':d['y']*100,'selected_category':int(code[selected]),
                'base_best_categories':sorted(set(code[av&(base==maximum)].tolist())),
                'base_max_ties':int((av&(base==maximum)).sum()),'eligible_by_category':{str(c):int((E&(code==c)).sum()) for c in (1,2,3)},
                'minimum_base_loss_by_category':{str(c):float(maximum-base[av&(code==c)].max()) for c in (1,2,3)},
                'budget':d['budget'],'amplitude':d['amplitude'],'B4_or_trust_fallback':d['fallback'],
                'execution_mode':d['execution_mode'],'selected_id':d['selected_id'],'candidate_table_sha256':data.filehash(path)}
            rows.append(row)
    result={'variant':'structure_t0','source_state_sha256':data.filehash(p),'axis':axis,'categories':{str(i+1):v for i,v in enumerate(levels)},
        'rows':rows,'scope':'Own committed predictions, own observed chosen points and public structures only; no counterfactual oracle queries.',
        'findings':['First base maximum unique: ID tie collapse is removed at the initial decision.',
            'All16 choices use category1; alternate categories remain gate-eligible in queries1..10,12..14.',
            'No controlled comparisons run because incumbent already exceeds frozen95% skip threshold.',
            'Refresh5/7/9 narrows already supported regions; B4/trust fallback10..16 removes knowledge influence, but base maximum stays category1.',
            'Removing ties and making input similarity chemical is insufficient for productive disambiguation of confounded categories.']}
    data.write(out/'structure_diagnosis.json',result)
    text=['**结构核试验的逐轮瓶颈：预测有区分，搜索仍封闭。**','',
        'structure_t0的32次选择与KF更新已回放封存。Perera AUC96.57／最佳96.57%，Doyle92.7455144125／98.28722602%；前者失败，未运行并列seed1/2。','',
        '只使用本轨迹已承诺的全候选预测、真实已选点及公开结构；未查询任何反事实产率。类别1是硼酸，类别2是Bpin，类别3是有机三氟硼酸盐，代码仅作显示。','',
        '| 查询 | 实际/% | 选中类别 | 最高基础qLogEI并列数 | 预算内类别1/2/3数量 | KF幅度 | B4或信任回退 |','|---:|---:|---:|---:|---|---:|---|']
    for r in rows:
        counts='/'.join(str(r['eligible_by_category'][str(c)]) for c in (1,2,3))
        text.append(f"| {r['query']} | {r['own_observed_yield_percent']:.2f} | {r['selected_category']} | {r['base_max_ties']} | {counts} | {r['amplitude']:.3f} | {r['B4_or_trust_fallback']} |")
    text += ['','第1轮已没有最高基础分并列，但没有找到新高产类别。前9轮另外两类别均有预算内候选；因此不能把这段缺少探索全部归因于机会损失门限。结构GP本身每轮最高分都在类别1，区域知识又偏好该类；缺少独立反例时，二者相互确认。','',
        '第5/7/9轮刷新仍收窄已支持区域。KF在同一区域实测反馈后调小幅度，换区域时又从新bank开始；它校正被实际采样的残差，没有为从未采样的替代类别创造证据。第10轮起知识影响因诊断/信任关闭，基础GP依然偏向类别1。','',
        '普通反例因已有95.57%初始化超过95%门槛而全部跳过。末期q11、15、16预算也确实排除了全部替代类别，收缩门限进一步限制了补救窗口。','',
        '下一项据此只检验受控、以预期收益排序的联合反例，在冻结的16次查询内部主动获得替代类别的真实证据；不继续扫描指纹权重或挑并列种子。','',
        '分子特征使用[RDKit Morgan指纹接口](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html)，固定半径2、2048bit并包含手性。类别索引仅查表，相似度来自结构。数值性质通过不等于搜索达标。','',
        '原结果：[structure_t0/report.md](structure_t0/report.md)；逐轮机器表：[structure_diagnosis.json](structure_diagnosis.json)。']
    (out/'structure_diagnosis.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'queries':len(rows),'chosen_categories':sorted(set(r['selected_category'] for r in rows))}))


if __name__=='__main__':main()
