"""Convert the reviewed report into an editable standalone XeLaTeX document.

Only presentation changes: text/numbers come from REPORT.md; readable chart
panels are regenerated from the same frozen histories. No optimization runs.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / 'REPORT.md'
ROOT = HERE.parents[2]
TMP = ROOT / 'outputs' / 'p0_p1_typeset'
ASSETS = HERE / 'assets'
INPUT_HASHES = {}


def h(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    INPUT_HASHES[str(path)] = h(path)
    return json.loads(path.read_text(encoding='utf-8'))


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, encoding='utf-8',
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def charts():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#77818b', 'text.color': '#27323b',
                         'savefig.facecolor': 'white'})
    records = []
    def values(path, label):
        state = read(path)
        assert state['status'] == 'COMPLETE' and len(state['history']) == 20
        y = np.array([x['y'] * 100 for x in state['history']])
        records.append({'source': str(path), 'label': label,
                        'yield_percent': y.tolist(),
                        'cumulative_best_percent': np.maximum.accumulate(y).tolist()})
        return y
    def finish(fig, axes, path, objective):
        for i, ax in enumerate(axes):
            ax.axvspan(.5, 4.5, color='#e9edf2', zorder=0)
            ax.set(xlim=(.5, 20.5), ylim=(0, 103), xticks=[1, 4, 8, 12, 16, 20],
                   yticks=[0, 25, 50, 75, 100],
                   xlabel='Experiment (4 initialization + 16 queries)',
                   ylabel=objective if i == 0 else 'Best value so far (%)')
            ax.grid(axis='y', color='#dfe4e8', lw=.55)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(labels),
                   bbox_to_anchor=(.5, 1.005), frameon=False, fontsize=8.3,
                   handlelength=2.1, columnspacing=1.15)
        fig.subplots_adjust(left=.075, right=.99, bottom=.18, top=.80, wspace=.27)
        fig.savefig(path, dpi=300)
        plt.close(fig)
    styles = {
        'Revised Agent': ('#2563a6', 'o', '-'),
        'LGBO': ('#c37713', 's', '--'),
        'LLINBO-Justify': ('#7c852b', '^', '-.'),
        'Pure BO': ('#42474d', 'x', ':'),
    }
    for case in range(1, 5):
        fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.70))
        paths = {
            'Revised Agent': ROOT / 'data/runs/suzuki_stable_20261029/runs' / f'c{case}_s0_assistant_initial_stable/state.json',
            'LGBO': ROOT / 'data/runs/suzuki_mixed_comparison/runs' / f'c{case}_s0_assistant_initial_LGBO_region/state.json',
            'LLINBO-Justify': ROOT / 'data/runs/suzuki_mixed_comparison/runs' / f'c{case}_s0_assistant_initial_LLINBO_matched/state.json',
            'Pure BO': ROOT / 'data/runs/suzuki_mixed_kalman/base' / f'c{case}_s0/state.json'}
        for label, path in paths.items():
            y = values(path, label)
            color, marker, line = styles[label]
            for ax, v in zip(axes, (y, np.maximum.accumulate(y))):
                ax.plot(range(1, 21), v, label=label, color=color, marker=marker,
                        ls=line, lw=1.15, markersize=2.8)
        axes[0].set_title(f'Case {case}: observed')
        axes[1].set_title(f'Case {case}: cumulative best')
        finish(fig, axes, ASSETS / f'p0_case{case}.png', 'Simulated yield (%)')
    for task, short in [('Perera_Suzuki', 'Perera Suzuki'),
                        ('Doyle_Buchwald_Hartwig', 'Doyle Buchwald-Hartwig')]:
        fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.85))
        paths = [('New tie0', ROOT / 'data/runs/stable_r2_20261029/feedback_t0/runs' / f'{task}_s0_feedback_t0/state.json', '#2563a6', '-'),
                 ('New tie1', ROOT / 'data/runs/stable_r2_20261029/feedback_t1/runs' / f'{task}_s0_feedback_t1/state.json', '#4b91ad', '--'),
                 ('New tie2', ROOT / 'data/runs/stable_r2_20261029/feedback_t2/runs' / f'{task}_s0_feedback_t2/state.json', '#87bfc7', '-.'),
                 ('Pure BO', ROOT / 'data/runs/hte_pointwise_seed0/runs' / f'{task}_s0_Pure_BO/state.json', '#42474d', ':'),
                 ('Reasoning BO', ROOT / 'data/runs/hte_pointwise_seed0/runs' / f'{task}_s0_ReasoningBO_point/state.json', '#c37713', '--')]
        for label, path, color, line in paths:
            y = values(path, label)
            for ax, v in zip(axes, (y, np.maximum.accumulate(y))):
                ax.plot(range(1, 21), v, label=label, color=color, ls=line,
                        lw=1.2, marker='o' if label == 'New tie0' else None, markersize=2.5)
        axes[0].set_title(short + ': observed', fontsize=9)
        axes[1].set_title('Cumulative best', fontsize=9)
        finish(fig, axes, ASSETS / ('p1_perera.png' if task == 'Perera_Suzuki' else 'p1_doyle.png'),
               'UV product area (%)' if task == 'Perera_Suzuki' else 'Reported yield (%)')
    write_json(TMP / 'figure_source_rows.json', records)
    return records


def plain(node):
    if isinstance(node, list):
        return ''.join(plain(v) for v in node)
    if not isinstance(node, dict):
        return ''
    if node.get('t') == 'Str':
        return node['c']
    if node.get('t') in ('Space', 'SoftBreak', 'LineBreak'):
        return ' '
    return plain(node.get('c', []))


def transform(ast):
    result = []
    section = ''
    headings = {}
    table_number = 0
    trace_table = False
    for position, block in enumerate(ast['blocks']):
        if block['t'] == 'Header':
            level, attr, content = block['c']
            if level == 1:
                continue
            block['c'][0] = level - 1
            section = plain(content)
            headings[level] = section
            headings = {k: v for k, v in headings.items() if k <= level}
            if level == 2 and section.startswith('附录B'):
                result.append({'t': 'RawBlock', 'c': ['latex', r'\Needspace{360pt}']})
            elif level == 2 and section.startswith(('附录A', '附录C')):
                result.append({'t': 'RawBlock', 'c': ['latex', r'\Needspace{10\baselineskip}']})
            if level == 3 and re.match(r'A\.\d+\s+Case', section):
                result.append({'t': 'RawBlock', 'c': ['latex', r'\Needspace{270pt}']})
            following = ast['blocks'][position+1] if position+1 < len(ast['blocks']) else None
            if level == 4 and following and following['t'] == 'Table':
                result.append({'t': 'RawBlock', 'c': ['latex', r'\Needspace{235pt}']})
                block = {'t': 'RawBlock', 'c': ['latex',
                    '\\par\\addvspace{0.8em}\\noindent{\\normalsize\\bfseries\\heiti ' +
                    section + '}\\par\\nobreak\\smallskip']}
        if block['t'] == 'Table':
            table_number += 1
            specs = block['c'][2]
            count = len(specs)
            headers = [plain(cell[-1]) for cell in block['c'][3][1][0][1]]
            header_text = ' '.join(headers)
            cells = block['c'][3][1][0][1]
            for cell, value in zip(cells, headers):
                if value == 'AUC较纯BO/百分点':
                    cell[-1] = [{'t': 'Plain', 'c': [{'t': 'Str', 'c': 'ΔAUC/pp'}]}]
            if count == 8:
                widths = [.18, .065, .065, .16, .12, .145, .145, .12]
                trace_table = True
            elif count == 5 and 'SearchAUC' in header_text:
                widths = [.36, .14, .14, .17, .19]
            elif count == 5 and '四Case' in header_text:
                widths = [.40, .15, .15, .15, .15]
            elif count == 5:
                widths = [.25, .27, .15, .16, .17]
            elif count == 4:
                widths = [.25, .25, .25, .25]
            elif count == 3:
                widths = [.19, .46, .35]
            elif count == 2:
                widths = [.29, .71]
            else:
                widths = [1 / count] * count
            for spec, width in zip(specs, widths):
                spec[1] = {'t': 'ColWidth', 'c': width}
            # Break source labels at underscores using URL's line-breaking rules.
            # Numeric tables retain the exact reviewed text.
            caption = section.replace('#### ', '').replace('### ', '')
            scope = ' '.join(headings.values())
            case_scope = ' '.join(v for k, v in headings.items() if k >= 3)
            cases = re.findall(r'Case\s*(\d)', case_scope)
            if cases and ('四、P0' in scope or '附录A' in scope):
                condition = '助手初始建议' if '4.1' in scope else section
                caption = f'P0 Case {cases[-1]}  {condition}  方法比较'
            block['c'][1] = [None, [{'t': 'Plain', 'c': [
                {'t': 'Str', 'c': f'{caption}  数据表'}]}]]
        is_image = block['t'] in ('Para', 'Figure') and (
            (block['t'] == 'Para' and any(x.get('t') == 'Image' for x in block['c']))
            or block['t'] == 'Figure')
        if is_image:
            description = plain(block)
            payload = json.dumps(block, ensure_ascii=False)
            if 'comparison_assistant_initial.png' in payload:
                for case in range(1, 5):
                    result.append({'t': 'RawBlock', 'c': ['latex',
                        '\\par\\addvspace{1em}\\noindent\\begin{minipage}{\\linewidth}\\centering\n'
                        f'\\includegraphics[width=\\linewidth]{{assets/p0_case{case}.png}}\n'
                        f'\\captionof{{figure}}{{P0 Case {case}的逐轮产率与累计最佳。助手初始建议，seed=0，灰色区为4次共同初始化。}}\n'
                        '\\end{minipage}\\par\\medskip\n']})
                result.append({'t': 'RawBlock', 'c': ['latex', r'\FloatBarrier']})
                continue
            if 'stable_strategy_comparison.png' in payload:
                for name, text in [('perera', 'Perera Suzuki（UV产物峰面积）'),
                                   ('doyle', 'Doyle Buchwald--Hartwig（报告产率）')]:
                    result.append({'t': 'RawBlock', 'c': ['latex',
                        '\\par\\addvspace{1em}\\noindent\\begin{minipage}{\\linewidth}\\centering\n'
                        f'\\includegraphics[width=\\linewidth]{{assets/p1_{name}.png}}\n'
                        f'\\captionof{{figure}}{{P1 {text}的逐轮值与累计最佳。优化seed=0，修订版并列seed=0、1、2；完全重合的曲线表示相同观测或累计最佳。}}\n'
                        '\\end{minipage}\\par\\medskip\n']})
                result.append({'t': 'RawBlock', 'c': ['latex', r'\FloatBarrier']})
                continue
        result.append(block)
    ast['blocks'] = result
    # Replace textual Unicode math characters without touching existing Math.
    replacements = {
        'α': r'\alpha', 'θ': r'\theta', 'σ': r'\sigma',
        'λ': r'\lambda', 'ε': r'\epsilon', 'μ': r'\mu',
        'ν': r'\nu', 'Δ': r'\Delta', '≥': r'\geq', '≤': r'\leq',
        '×': r'\times', '∈': r'\in', '…': r'\ldots',
    }
    def walk(node):
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if node.get('t') == 'Math':
            return node
        if node.get('t') == 'Link':
            target = node['c'][2][0]
            if re.match(r'^[A-Za-z]:/', target):
                node['c'][2][0] = 'file:///' + target
        if node.get('t') == 'Str':
            text = node['c'].replace('–', '-').replace('—', '-').replace('‑', '-')
            # These fixed textual labels occur outside displayed equations.
            text = text.replace('⁻¹²', '^{-12}').replace('⁻⁴', '^{-4}').replace('⁶', '^{6}').replace('⁻³', '^{-3}')
            if any(c in text for c in replacements) or any(c in text for c in '₀₁₂₃₄₅₆₇₈₉ⱼ') or '^{' in text:
                escaped = (text.replace('\\', r'\textbackslash{}').replace('&', r'\&')
                           .replace('%', r'\%').replace('#', r'\#').replace('$', r'\$'))
                for k, v in replacements.items():
                    escaped = escaped.replace(k, '\\ensuremath{' + v + '}')
                for i, c in enumerate('₀₁₂₃₄₅₆₇₈₉'):
                    escaped = escaped.replace(c, '\\ensuremath{_{' + str(i) + '}}')
                escaped = escaped.replace('ⱼ', r'\ensuremath{_{j}}')
                escaped = re.sub(r'\^\{(-?\d+)\}', lambda m: r'\ensuremath{^{'+m.group(1)+'}}', escaped)
                escaped = escaped.replace('_', r'\_')
                # Undo escaping in the subscript math fragments.
                escaped = escaped.replace(r'\ensuremath{\_{', r'\ensuremath{_{')
                return {'t': 'RawInline', 'c': ['latex', escaped]}
            node['c'] = text
        for key in list(node):
            if key != 't':
                node[key] = walk(node[key])
        return node
    ast = walk(ast)
    return ast, table_number, trace_table


HEADER = r"""
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{booktabs,longtable,array}
\usepackage{etoolbox}
\usepackage{placeins}
\usepackage{fancyhdr}
\usepackage{caption}
\usepackage{xurl}
\usepackage{needspace}
\geometry{a4paper,left=23mm,right=23mm,top=23mm,bottom=23mm,headheight=14pt,headsep=8mm}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.35em}
\linespread{1.13}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.22}
\setlength{\tabcolsep}{4pt}
\AtBeginEnvironment{longtable}{\fontsize{9.2pt}{12.6pt}\selectfont\renewcommand{\arraystretch}{1.05}\setlength{\parskip}{0.1em}}
\setlength{\LTpre}{0.5em}
\setlength{\LTpost}{0.7em}
\captionsetup{font=small,labelfont=bf,labelsep=quad,skip=5pt,hypcap=false}
\ctexset{
 section={format=\Large\bfseries\heiti,beforeskip=1.8ex,afterskip=1.1ex},
 subsection={format=\large\bfseries\heiti,beforeskip=1.5ex,afterskip=0.8ex},
 subsubsection={format=\normalsize\bfseries\heiti,beforeskip=1.2ex,afterskip=0.6ex}
}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small P0与P1两级实验数据报告}
\fancyhead[R]{\small Agent逻辑与实验对照}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\AtBeginDocument{\hypersetup{
 pdftitle={P0与P1两级实验数据报告},pdfauthor={},pdfsubject={Agent逻辑 新方案特征及对照结果}}}
\let\OriginalTOC\tableofcontents
\renewcommand{\tableofcontents}{\pagenumbering{Roman}\OriginalTOC\clearpage\pagenumbering{arabic}}
"""

FRONT = r"""
\begin{titlepage}
\thispagestyle{empty}
\vspace*{22mm}
\begin{center}
{\Huge\bfseries\heiti P0与P1两级实验数据报告\par}
\vspace{12mm}
{\Large Agent执行逻辑与修订方案对照结果\par}
\vspace{20mm}
{\large P0  Reizman Suzuki Case 1至4\par}
\vspace{4mm}
{\large P1  Perera Suzuki与Doyle Buchwald--Hartwig\par}
\vspace{18mm}
{\normalsize 固定候选集\quad Mixed GP\quad seed 0\quad 4次初始化加16次查询\par}
\vfill
{\large 2026年10月29日\par}
\vspace{10mm}
\end{center}
\begin{quote}\small
依据封存轨迹整理。P0含四种知识条件，P1含三组并列顺序检查。两级采用不同任务和目标口径，结果分别列示，不合成跨数据集总分。
\end{quote}
\end{titlepage}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--cached', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    TMP.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    source_hash = h(REPORT)
    records = charts()
    ast = json.loads(run(['pandoc', REPORT, '-f', 'markdown+tex_math_dollars', '-t', 'json']).stdout)
    ast, tables, trace_table = transform(ast)
    # Explicit font files let Tectonic fetch TeX fonts without system installation.
    for name, stem in [('mainfont', 'texgyretermes'), ('sansfont', 'texgyreheros'),
                       ('monofont', 'texgyrecursor')]:
        ast['meta'][name] = {'t': 'MetaString', 'c': stem + '-regular.otf'}
        ast['meta'][name + 'options'] = {'t': 'MetaList', 'c': [
            {'t': 'MetaString', 'c': 'BoldFont=' + stem + '-bold.otf'},
            {'t': 'MetaString', 'c': 'ItalicFont=' + stem + '-italic.otf'},
            {'t': 'MetaString', 'c': 'BoldItalicFont=' + stem + '-bolditalic.otf'}]}
    write_json(TMP / 'report_ast.json', ast)
    (TMP / 'header.tex').write_text(HEADER, encoding='utf-8')
    (TMP / 'front.tex').write_text(FRONT, encoding='utf-8')
    output = run(['pandoc', TMP / 'report_ast.json', '-f', 'json', '-t', 'latex', '-s',
                  '--pdf-engine=xelatex', '--toc', '--toc-depth=2', '-V', 'toc-title=目录',
                  '-V', 'documentclass=ctexart', '-V', 'classoption=fontset=fandol',
                  '-V', 'fontsize=11pt',
                  '--include-in-header', TMP / 'header.tex',
                  '--include-before-body', TMP / 'front.tex'])
    tex = '\\PassOptionsToPackage{hypertexnames=false}{hyperref}\n' + output.stdout
    (HERE / 'P0_P1_Report.tex').write_text(tex, encoding='utf-8')
    assert h(REPORT) == source_hash
    for path, expected in INPUT_HASHES.items():
        assert h(path) == expected, path
    write_json(HERE / 'build_provenance.json', {
        'source_markdown': str(REPORT), 'source_markdown_sha256': source_hash,
        'tex_sha256': h(HERE / 'P0_P1_Report.tex'), 'tables_preserved': tables,
        'figure_series': len(records), 'figures': 6, 'inputs': INPUT_HASHES,
        'changes': ['Chinese XeLaTeX layout', 'captioned long tables',
                    'same reviewed chart data in readable per-case panels',
                    'textual Unicode math made explicit TeX'],
        'new_experiments': 0, 'new_API_calls': 0})
    print(json.dumps({'tex_created': str(HERE / 'P0_P1_Report.tex'),
                      'tables': tables, 'figures': 6}, ensure_ascii=True), flush=True)
    if args.compile:
        engine = ROOT / 'outputs/tex_tools/tectonic/tectonic.exe'
        command = [engine, '--keep-logs', '--keep-intermediates', '--print',
                   '--outdir', TMP, HERE / 'P0_P1_Report.tex']
        if args.cached:
            command.insert(1, '--only-cached')
        env = os.environ.copy()
        env['TECTONIC_CACHE_DIR'] = str(ROOT / 'outputs' / 'tex_cache')
        with (TMP / 'compile_console.log').open('wb') as log:
            completed = subprocess.run([str(x) for x in command], cwd=HERE, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        print((TMP / 'compile_console.log').read_text(encoding='utf-8', errors='replace')[-7000:], flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        shutil.copy2(TMP / 'P0_P1_Report.pdf', HERE / 'P0_P1_Report.pdf')
        print(json.dumps({'pdf_created': str(HERE / 'P0_P1_Report.pdf'),
                          'pdf_sha256': h(HERE / 'P0_P1_Report.pdf')}), flush=True)


if __name__ == '__main__':
    main()
