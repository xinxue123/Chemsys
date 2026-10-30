"""Scientific plots from the frozen matched-study results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path('research/suzuki_guarded');r=json.loads((root/'results.json').read_text(encoding='utf8'))
font=Path('C:/Windows/Fonts/msyh.ttc');font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
    'font.size':11.5,'axes.unicode_minus':False})
methods=['B0','LGBO_region','B3_gate','LGBO_gate','B4_base','B4_LGBO','A2_LGBO']
labels=['无知识 BO','区域 LGBO','单候选审查','仅预算门限','原硬筛选 B4','组合 B4 + 区域','全部撤销']
regimes=['correct','wrong_category','mixed','assistant_initial']
values=np.array([[r['tables'][g][m]['AUC_mean'] for g in regimes] for m in methods])
fig,(ax,bx)=plt.subplots(2,1,figsize=(8.8,9.2),gridspec_kw={'height_ratios':[1.5,1.]})
image=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
ax.set_yticks(range(len(methods)),labels)
ax.set_xticks(range(4),['正确知识','错误类别','混合错误','助手初始建议'])
ax.tick_params(length=0,pad=8)
for i in range(len(methods)):
    for j in range(4):
        rgb=image.cmap(image.norm(values[i,j]))[:3]
        luminance=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
        ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if luminance<.48 else 'black')
ax.set_title('统一底座下的平均 SearchAUC（%）',loc='left',fontsize=13,pad=15)
for spine in ax.spines.values():spine.set_visible(False)
fig.colorbar(image,ax=ax,fraction=.028,pad=.025,label='SearchAUC（%）')
controls=['LGBO_region','B3_gate','LGBO_gate','A2_LGBO']
yl=['相对区域 LGBO','相对单候选审查','相对仅预算门限','相对全部撤销']
all_delta=[]
for i,control in enumerate(controls):
    per_case=[np.mean([r['paired_comparisons'][g+'_B4_LGBO_minus_'+control]['by_case'][str(c)]
                      for g in ('wrong_category','mixed')]) for c in (2,3,4)]
    mean=float(np.mean(per_case));lo=min(per_case);hi=max(per_case);all_delta.extend(per_case)
    bx.hlines(i,lo,hi,color='#8c939b',linewidth=2)
    bx.scatter(per_case,[i]*3,color='#8c939b',s=24,zorder=3)
    bx.scatter([mean],[i],marker='D',s=48,color='#246c9f' if mean>=0 else '#b75d34',zorder=4)
    bx.text(mean,i-.15,f'{mean:+.2f}',ha='center',va='bottom',fontsize=11)
span=max(all_delta+[0])-min(all_delta+[0]);pad=max(.6,span*.13)
bx.set_xlim(min(all_delta+[0])-pad,max(all_delta+[0])+pad);bx.set_ylim(3.55,-.6)
bx.set_yticks(range(4),yl);bx.axvline(0,color='#555555',linewidth=1,linestyle=':')
bx.grid(axis='x',alpha=.15);bx.spines[['top','right']].set_visible(False)
bx.set_title('组合方法在错误/混合知识下的增量',loc='left',fontsize=13,pad=15)
bx.set_xlabel('SearchAUC 差值（百分点；正值更好）')
fig.suptitle('B4 联合修订与区域引导：冻结比较结果',fontsize=15,y=.985)
fig.text(.5,.025,'Case 2–4，每项 15 条轨迹，统一 4+16 次查询；预算仅由 Case 1 选择。\n菱形为均值，灰线/圆点为三个案例的范围/均值，并非置信区间。人工压力测试与初始建议回放。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.96),h_pad=2.4)
fig.savefig(root/'matched_results.png',dpi=180);fig.savefig(root/'matched_results.svg');plt.close(fig)
