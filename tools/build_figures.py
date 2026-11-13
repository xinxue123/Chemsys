"""Rebuild Figures 2–5 from the frozen plot data and saved results."""

import argparse
import csv
import json
import statistics as st
import tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
DATA = ROOT / 'data/figure_data'
BLUE, ORANGE, GREEN, RED, PINK, GREY, INK = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#8C8C8C', '#222222'
T90_DF7 = 1.894579
plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
                     'xtick.major.width': .7, 'ytick.major.width': .7, 'pdf.fonttype': 42, 'lines.linewidth': 1.4, 'legend.frameon': False})
rng = np.random.default_rng(20261106)


def rows_of(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def save(fig, name, out):
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def panel(ax, letter):
    ax.text(-.02, 1.05, letter, transform=ax.transAxes, fontsize=9.5, fontweight='bold', ha='right', va='bottom')


def bootstrap_mean(values, n=10000):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return np.quantile(means, [.05, .95])


def development(out):
    rows = rows_of(R/'hypothesis_scope_method_evidence_20261105/method_by_task.csv')
    names = (('lgbo', 'LGBO', '19'), ('hyp', 'Controller', '2.9'), ('first', 'LLM-start BO', '0'), ('rbo', 'Reasoning BO', '19'), ('bo', 'Random-start BO', '0'))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.45), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=.34))
    for i, (key, label, calls) in enumerate(names):
        values = [float(r['auc_pct_of_max']) for r in rows if r['method'] == key]
        mean = st.mean(values)
        half = T90_DF7*st.stdev(values)/len(values)**.5
        y = len(names) - 1 - i
        color = BLUE if key == 'hyp' else INK
        a.errorbar(mean, y, xerr=half, fmt='o', color=color, ms=4.5, capsize=2, lw=1.1)
        a.text(101.5, y, calls, va='center', ha='right', fontsize=7.5, color=color)
    a.text(101.5, len(names) - .35, 'LLM calls', ha='right', fontsize=7, color=GREY)
    a.set_yticks(range(len(names)), [n[1] for n in names][::-1])
    a.set_xlim(60, 102)
    a.set_ylim(-.6, len(names) - .1)
    a.set_xlabel('Mean SearchAUC (% of pool maximum)')
    panel(a, 'a')
    rows = rows_of(DATA / 'fig2_development/panel_b_running_best_gap.csv')
    query = [int(r['query']) for r in rows]
    for task in ('Reizman 1', 'Reizman 2', 'Reizman 3', 'Reizman 4', 'Perera', 'Doyle', 'SnAr', 'HPLC'):
        b.plot(query, [float(r[task]) for r in rows], color=GREY, lw=.6, alpha=.65)
    b.plot(query, [float(r['mean_of_eight_tasks']) for r in rows], color=BLUE, lw=2, label='Mean of eight tasks')
    b.plot([], [], color=GREY, lw=.8, label='Individual tasks')
    b.axhline(0, color=INK, lw=.7)
    b.set_xlim(1, 20)
    b.set_xticks([1, 5, 10, 15, 20])
    b.set_xlabel('Query')
    b.set_ylabel('LLM-start minus random-start BO\n(running best, % of maximum)')
    b.legend(loc='upper right')
    panel(b, 'b')
    save(fig, 'fig2_development', out)


def band(ax, data, color, label, ls='-'):
    q = np.arange(1, 21)
    low, high = np.array([bootstrap_mean(data[:, i]) for i in range(20)]).T
    ax.fill_between(q, low, high, color=color, alpha=.16, lw=0)
    ax.plot(q, data.mean(axis=0), color=color, ls=ls, label=label)


def screen(out):
    """New-task screen: LLM pick percentile against start gain on every task, and running best on the three new tasks."""
    names = dict(c1='Reizman 1', c2='Reizman 2', c3='Reizman 3', c4='Reizman 4', perera='Perera', doyle='Doyle', snar='SNAr', hplc='HPLC',
                 aryl='Direct Arylation', dfl='Deoxyfluor.', bor='Borylation', chx='Pd coupling')
    result = json.loads((R/'hypothesis_scope_v2_screen_20261107/screen.json').read_text(encoding='utf-8'))['tasks']
    aliases = dict(zip(
        ('Reizman 1', 'Reizman 2', 'Reizman 3', 'Reizman 4', 'Perera', 'Doyle', 'SnAr', 'HPLC',
         'Direct Arylation', 'Deoxyfluorination', 'Borylation', 'Pd coupling'),
        ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc', 'aryl', 'dfl', 'bor', 'chx'),
    ))
    points = [(aliases[r['task']], float(r['llm_pick_pool_percentile']),
               float(r['llm_start_minus_random_start_searchauc']), r['role'] == 'development')
              for r in rows_of(DATA / 'fig4_screen/panel_a_pick_percentile_vs_gain.csv')]
    runs = defaultdict(dict)
    for r in rows_of(R/'hypothesis_scope_v2_screen_20261107/trajectories.csv'):
        runs[r['task'], r['arm'], int(r['stream'])][int(r['query'])] = float(r['running_best'])
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.0), gridspec_kw=dict(wspace=.40, hspace=.46))
    axes = axes.ravel()
    a = axes[0]
    names['c2'] = 'Reizman 1–3'
    offsets = {'c2': (-24, 5), 'snar': (-14, 3.5), 'perera': (-9, 4), 'c4': (-10, -8), 'chx': (-36, -8), 'doyle': (-8, -8)}
    for task, x, y, dev in points:
        a.scatter(x, y, s=16, marker='o' if dev else 's', color=GREY if dev else RED, lw=0, alpha=.9, zorder=3)
        if task not in ('c1', 'c3'):
            a.annotate(names[task], (x, y), xytext=offsets.get(task, (3, 1)), textcoords='offset points', fontsize=5.6, color=INK, ha='left')
    a.axhline(0, color=GREY, lw=.8, ls='--')
    a.axvline(90, color=BLUE, lw=.8, ls=':')
    a.set_xlim(0, 104)
    a.set_ylim(-19, 24)
    a.set_xlabel('LLM first pick, pool percentile')
    a.set_ylabel('LLM-start minus random-start BO\n(SearchAUC, % of maximum)')
    a.scatter([], [], s=16, marker='o', color=GREY, label='Development')
    a.scatter([], [], s=16, marker='s', color=RED, label='New or held out')
    a.legend(loc='upper left', fontsize=6, handletextpad=.1, borderpad=.2)
    panel(a, 'a')
    for ax, task, letter in zip(axes[1:], ('dfl', 'bor', 'chx'), 'bcd'):
        for arm, label, color, ls in (('controller', 'Controller', BLUE, '-'), ('bo', 'Random-start BO', INK, '-'), ('llm', 'LLM-start BO', RED, '--')):
            band(ax, np.array([[runs[task, arm, s][q] for q in range(1, 21)] for s in range(24)]), color, label, ls)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 102)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel('Query')
        ax.set_title(names[task] + f" (pick {result[task]['mean_pick_percentile']:.0f}th)", loc='left', fontsize=7.5)
        panel(ax, letter)
        if task == 'chx':
            ax.tick_params(labelleft=False)
    axes[1].set_ylabel('Mean running best (%)')
    axes[2].set_ylabel('Mean running best (%)')
    axes[3].legend(loc='upper left', fontsize=5.8, handlelength=1.6)
    save(fig, 'fig4_screen', out)


def first_point(out):
    rows = rows_of(R/'hypothesis_scope_v2_gp_followups_20261106/first_point_runs.csv')
    result = json.loads((R/'hypothesis_scope_v2_gp_followups_20261106/gp_followups.json').read_text(encoding='utf-8'))['test_F']
    panels = (('aryl', 'Direct Arylation: best solvent'), ('doyle', 'Doyle: best base'), ('perera', 'Perera: best electrophile'))
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.75), sharey=True, gridspec_kw=dict(wspace=.09, width_ratios=[1.32, 1, 1]))
    for ax, (task, title), letter in zip(axes, panels, 'abc'):
        spec = result[task]
        level_labels = {'aryl': ('Best solvent', 'Other solvents'),
                        'doyle': ('Best base', 'Other bases'),
                        'perera': ('Best\nelectrophile', 'Other\nelectrophiles')}[task]
        points = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r['task'] == task:
                points[r['group']][r['first_id']].append(float(r['SearchAUC_pct']))
        labels = []
        for x, (group, color, share) in enumerate((('inside', RED, spec['share_inside']), ('outside', BLUE, spec['share_outside']))):
            values = [st.mean(v) for v in points[group].values()]
            ax.scatter(x + rng.uniform(-.15, .15, len(values)), values, s=9, color=color, alpha=.55, lw=0)
            low, high = bootstrap_mean(values)
            mean = st.mean(values)
            ax.errorbar(x + .27, mean, yerr=[[mean - low], [high - mean]], fmt='o', color=INK, ms=4, capsize=2, lw=1.1)
            labels.append('Low-yield start:\n' + level_labels[x] + f'\n{share:.0f}%')
        ticks = [0, 1]
        if task == 'aryl':
            llm = result['llm_points']
            ax.scatter([2 + d for d in (-.12, 0, .12)], [r['mean_auc'] for r in llm.values()], s=22, marker='D', color=PINK, lw=0, zorder=3)
            ticks.append(2)
            labels.append('LLM\nfirst picks')
            ax.set_xlim(-.5, 2.5)
        else:
            ax.set_xlim(-.5, 1.6)
        ax.set_xticks(ticks, labels)
        for tick_label in ax.get_xticklabels()[:2]:
            tick_label.set_fontsize(6.5)
        ax.set_title(title, loc='left')
        gap = f"{spec['gap_outside_minus_inside']:+.1f}".replace('-', '−')
        ax.text(.03, .93, f"gap {gap}, p = {spec['permutation_p']:.3f}", transform=ax.transAxes, fontsize=7.5)
        ax.set_ylim(15, 102)
        panel(ax, letter)
    axes[0].set_ylabel('SearchAUC (% of maximum)')
    save(fig, 'fig5_first_point', out)


def main():
    from build_figure3 import render

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Fresh output directory.')
    args = parser.parse_args()
    if args.out is not None:
        out = args.out.resolve()
        if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / 'output'):
            parser.error('Use a fresh directory under output/ or outside the submission.')
        out.mkdir(parents=True, exist_ok=False)
    else:
        parent = ROOT / 'output'
        parent.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix='figures-', dir=parent))
    development(out)
    render(out)
    screen(out)
    first_point(out)
    print('PASS Figures 2-5:', out)


if __name__ == '__main__':
    main()
