"""Static scientific comparison of the reviewed intervention means."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path('research/suzuki_diagnosis')
data=json.loads((root/'comparison_diagnosis.json').read_text(encoding='utf8'))
font=Path('C:/Windows/Fonts/msyh.ttc')
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
    'font.size':12,'axes.unicode_minus':False})
g=data['groups']
fig,axes=plt.subplots(2,1,figsize=(8.5,7.9))
panels=[
    ('原 B0：保持 GP 与采集函数，改变候选池和初始化',
     ['旧初始化 · 512 候选','旧初始化 · 4672 候选','新初始化 · 512 候选','新初始化 · 4672 候选'],
     ['legacy_original','expanded_pool_old_init','old_pool_new_init','expanded_pool_new_init']),
    ('官方方法：配对关闭知识引导',
     ['LLINBO：关闭 LLM','LLINBO：完整方法','LGBO：区域偏移为 0¹','LGBO：完整方法'],
     ['LLINBO_without_LLM','LLINBO','LGBO_zero_regional_lift','LGBO'])]
for ax,(title,labels,keys) in zip(axes,panels):
    values=[g[k]['search_auc_pct'] for k in keys]
    ax.barh(range(4),values,color='#286a9c',height=.52)
    ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('SearchAUC（平均最佳产率，%）')
    ax.set_title(title,loc='left',fontsize=12.5,pad=13)
    ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
    for i,v in enumerate(values):ax.text(v+1.1,i,f'{v:.2f}',ha='left',va='center',fontsize=12)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Suzuki 表现差异的实验诊断',fontsize=15,y=.985)
fig.text(.5,.027,'Case 2–4，每个设置 15 条轨迹，均为 4+16 次查询；冻结模拟器、事后诊断、非盲评。\n¹固定原偏好分支序列，仅将区域均值偏移设为 0。官方源码未修改。',ha='center',fontsize=9.5)
fig.tight_layout(rect=(0,.095,1,.955),h_pad=2.2)
fig.savefig(root/'controlled_comparison.png',dpi=180)
fig.savefig(root/'controlled_comparison.svg')
plt.close(fig)
