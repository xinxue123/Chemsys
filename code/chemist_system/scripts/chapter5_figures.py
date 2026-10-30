"""Export scientific figures and complete tables from independently audited data."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chem.decision.replay import write_json_atomic

COLORS = {'B0': '#606060', 'B1': '#D55E00', 'B2': '#009E73', 'B3': '#0072B2', 'B4': '#7B3294'}
METHODS = ('B0', 'B1', 'B2', 'B3', 'B4')
STYLES = {'B0': '--', 'B1': '-.', 'B2': ':', 'B3': '--', 'B4': '-'}
REGIMES = ('relevant', 'single_harmful', 'joint_only_harmful', 'mixed')
TITLES = ('Relevant priors', 'One harmful prior', 'Joint-only harmful priors', 'Mixed priors')


def load(name):
    path = ROOT / 'data/runs/chapter5' / name / 'independent_audit.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['status'] != 'PASS':
        raise ValueError('figures require a complete independently audited suite')
    return data


def save(fig, name, out):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out / (name + '.' + extension), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    output = ROOT / 'research/chapter5/figures'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    math, reaction, assistant = [load(n) for n in ('math_test', 'summit_test', 'assistant_test')]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey='row')
    for row_number, (data, label) in enumerate(((math, 'Mathematical tasks'), (reaction, 'Reaction emulators'))):
        for column, regime in enumerate(REGIMES):
            ax = axes[row_number, column]
            for method in METHODS:
                selected = [r for r in data['rows'] if r['method'] == method and r['regime'] == regime]
                curves = np.array([r['best_curve'] for r in selected])
                ax.plot(np.arange(4, 21), curves[:, 3:].mean(axis=0), color=COLORS[method],
                        linestyle=STYLES[method], linewidth=2.2 if method == 'B4' else 1.6, label=method)
            ax.set_title(TITLES[column] if row_number == 0 else f'{len(selected)} paired campaigns / method')
            ax.grid(alpha=.2)
            ax.set_xticks([4, 8, 12, 16, 20])
            if row_number == 1:
                ax.set_xlabel('Total queried candidates')
            if column == 0:
                ax.set_ylabel(label + '\nBest normalized objective')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(.5, .96))
    fig.suptitle('Fixed artificial priors: all planned trajectories retained', y=1.005)
    fig.subplots_adjust(top=.82, hspace=.4, wspace=.18)
    save(fig, 'fixed_prior_curves', output)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for case, ax in zip((2, 3, 4), axes):
        for method in METHODS:
            rows = [r for r in assistant['rows'] if r['method'] == method and r['task_id'].endswith(f'_{case}')]
            curves = np.array([r['best_curve'] for r in rows])
            ax.plot(np.arange(4, 21), curves[:, 3:].mean(axis=0), color=COLORS[method],
                    linestyle=STYLES[method], linewidth=2.2 if method == 'B4' else 1.6, label=method)
            ax.scatter(np.full(5, 20.), curves[:, -1], color=COLORS[method], alpha=.4, s=15)
        ax.set_title(f'Case {case}: five paired seeds')
        ax.set_xlabel('Total queried candidates')
        ax.set_xticks([4, 8, 12, 16, 20])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Best normalized emulator yield')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(.5, 1.02))
    fig.subplots_adjust(top=.76, wspace=.25)
    save(fig, 'assistant_case_curves', output)

    summaries = {n: json.loads((ROOT / 'research/chapter5' / (n + '_results.json')).read_text(encoding='utf-8'))
                 for n in ('math_test', 'summit_test', 'assistant_test')}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    contrast = next(r for r in summaries['math_test']['paired_comparisons']
                    if r['regime'] == 'all_regimes' and r['comparison'] == 'B4_minus_B3')
    axes[0].scatter([r['search_auc'] for r in contrast['clusters']], range(5), color=COLORS['B4'], s=55)
    axes[0].set_yticks(range(5), [r['family'] for r in contrast['clusters']])
    axes[0].set_title('Mathematics: family mean differences')
    for name, offset, marker, color, label in (
            ('summit_test', -.1, 's', COLORS['B3'], 'Fixed artificial priors'),
            ('assistant_test', .1, 'o', COLORS['B4'], 'Actual assistant')):
        contrast = next(r for r in summaries[name]['paired_comparisons']
                        if r['regime'] == 'all_regimes' and r['comparison'] == 'B4_minus_B3')
        axes[1].scatter([r['search_auc'] for r in contrast['clusters']], np.arange(3) + offset,
                        color=color, marker=marker, s=55, label=label)
    axes[1].set_yticks(range(3), ['Case 2', 'Case 3', 'Case 4'])
    axes[1].set_title('Reaction scenarios: case mean differences')
    axes[1].legend(fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.3), ncol=2, frameon=False)
    for ax in axes:
        ax.axvline(0., color='#222222', linewidth=1.)
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('B4 minus B3 SearchAUC (positive favors B4)')
    fig.tight_layout()
    save(fig, 'paired_auc_effects', output)

    table = ['# 第5章完整结果表', '', '由独立轨迹核验自动生成。SearchAUC 为 Best@5 至 Best@20 的算术均值；Regret20 为冻结候选池最优值减最终最优观测。',
             '停止轨迹沿用 incumbent 至第20个预算位置。反应目标均为模拟器预测产率除以100，不是实验产率。', '']
    for name, summary in summaries.items():
        table += [f'## {name}', '', '| 先验 | 方法 | 轨迹 | 提前停止 | SearchAUC | Regret20 | 预算违例轮次 |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for r in summary['tables']:
            table.append(f"| {r['regime']} | {r['method']} | {r['campaigns']} | {r['stopped']} | {r['search_auc']:.6f} | {r['final_regret']:.6f} | {r['budget_violations']} |")
        table += ['', '任务级差值、家族聚类区间、恢复率分母见同名 results.json。', '']
    (ROOT / 'research/chapter5/results_tables.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    sources = [ROOT / 'data/runs/chapter5' / n / 'independent_audit.json'
               for n in ('math_test', 'summit_test', 'assistant_test')]
    receipt = {'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sources},
               'figure_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*'))},
               'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
               'shading_or_error_bars': 'none; case-curve endpoint dots show all five seeds',
               'caption_scope': 'finite-pool simulator and synthetic optimization; no laboratory performance claim'}
    write_json_atomic(ROOT / 'research/chapter5/figure_receipt.json', receipt)
    print(json.dumps({'figures': 3, 'formats_per_figure': 3, 'complete_method_tables': True}))


if __name__ == '__main__':
    main()
