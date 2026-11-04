"""Typeset the audited confirmation as a standalone LaTeX addendum."""
from pathlib import Path
import csv
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ID = 'program_kf_confirmation_20261104'
REPORT = ROOT/'experiments/reports'/ID
PAPER = ROOT/'paper/v13_confirmation'
LOGS = ROOT/'experiments/logs'/ID/'typeset'
CELLS = {'perera1':'Perera Suzuki 1', 'hplc1':'HPLC 1', 'aryl0':'Direct Arylation 0'}
ARMS = {'bo':'BO','reasoning_bo':'Reasoning BO','v13':'V13','fixed':'Fixed weight','dynamic':'Dynamic KF'}


def read_csv(name):
    with (REPORT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def fmt(x, signed=False):
    return format(float(x), '+.2f' if signed else '.2f')


def write(name, s):
    (PAPER/name).write_text(s+'\n', encoding='utf-8')


def table(caption, columns, headers, rows):
    return '\n'.join([
        r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',
        r'\begin{tabular}{'+columns+'}',r'\toprule',' & '.join(headers)+r' \\',
        r'\midrule',*[' & '.join(map(str,row))+r' \\' for row in rows],
        r'\bottomrule',r'\end{tabular}',r'\end{table}'])


def main():
    audit = json.loads((REPORT/'audit.json').read_text(encoding='utf-8'))
    if audit['status']!='PASS_COMPLETE':
        raise ValueError('Completed replay and arithmetic verification required')
    pfx = audit['prefix']
    switches = sum(len(x['switched']) for x in pfx.values())
    positive = sum(len(x['positive']) for x in pfx.values())
    negative = sum(len(x['negative']) for x in pfx.values())
    decision_count = f'{switches} decision' + ('' if switches==1 else 's')
    write('abstract_results.tex',
        f'Across 27 locked prefixes, deploying the dynamic weight changed {decision_count}, '
        f'with {positive} positive and {negative} negative immediate improvement contrasts. '
        f'The dynamic controller had a material SearchAUC gain in {audit["material_gain_cells"]} '
        f'of three matched trajectory comparisons and a material loss in {audit["material_loss_cells"]}.')
    prefix_rows = [[CELLS[cell],x['all_GP_tied'],x['all_LLM_tied'],x['mathematical_opportunities'],
                   len(x['switched']),len(x['positive']),len(x['negative']),len(x['neutral'])]
                   for cell,x in pfx.items()]
    prefix = table('Nine locked prefixes per cell. Possible denotes an alternative winner '
        'somewhere on the allowed weight interval. Gain, loss and neutral classify the change '
        'in immediate improvement over the shared incumbent.', 'lrrrrrrr',
        ['Cell','GP ties','LLM ties','Possible','Switched','Gain','Loss','Neutral'], prefix_rows)
    prefix += (f'\nThe offline evaluator looked up {2*switches} labels after locking all decisions. '
               'The paired prefix stage used zero new online queries.\n')
    sw = [r for r in read_csv('prefix_pairs.csv') if r['switched']=='True']
    if sw:
        prefix += table(r'All observed switches. Objective values use native units. '
            r'$\Delta I$ is the dynamic minus fixed change in improvement over $M$.','lrrrrrr',
            ['Cell','Query','$M$','Fixed $y$','Dynamic $y$','Dynamic $w$',r'$\Delta I$'],
            [[CELLS[r['cell']],r['query'],fmt(r['incumbent']),fmt(r['fixed_outcome']),
              fmt(r['dynamic_outcome']),f"{float(r['dynamic_weight']):.3f}",fmt(r['delta_improvement'],True)]
              for r in sw])
    write('prefix_results.tex',prefix)
    dash = r'\textemdash{}'
    metrics = read_csv('metrics.csv')
    effects = read_csv('paired_effects.csv')
    traces = table('All six new trajectories and nine frozen reference trajectories. A dash '
        'denotes a target unreached within twenty queries.', 'llrrrr',
        ['Cell','Method','SearchAUC','Final best','Target','Hit query'],
        [[CELLS[r['cell']],ARMS[r['arm']],fmt(r['SearchAUC']),fmt(r['final_best']),
          fmt(r['target_95']),r['first_target'] or dash] for r in metrics])
    traces += '\n'+table('Paired dynamic minus fixed results under the registered controller.',
        'lrrrr',['Cell',r'$\Delta$SearchAUC',r'Relative (\%)',r'$\Delta$final best','First divergence'],
        [[CELLS[r['cell']],fmt(r['SearchAUC_delta'],True),fmt(float(r['SearchAUC_relative_delta'])*100,True),
          fmt(r['final_best_delta'],True),r['first_divergence'] or dash] for r in effects])
    if (REPORT/'first_divergences.csv').exists():
        traces += '\n'+table('First action divergence under the same observed history, GP, '
            'candidate menu and raw response. Fixed weight is 0.4. '
            'The last column measures the change in improvement over the common incumbent.',
            'lrrrrrr',['Cell','Query','$M$','Fixed $y$','Dynamic $y$',
                       'Dynamic $w$',r'$\Delta I$'],
            [[CELLS[r['cell']],r['query'],fmt(r['incumbent']),fmt(r['fixed_value']),
              fmt(r['dynamic_value']),f"{float(r['dynamic_weight']):.3f}",
              fmt(r['immediate_improvement_delta'],True)] for r in read_csv('first_divergences.csv')])
    write('trace_results.tex',traces)
    text = (f'Positive locked-prefix switches were observed in {audit["positive_switch_cells"]} cells. '
            f'The prespecified mechanism confirmation criterion was '
            f'{"met" if audit["mechanism_confirmed"] else "unmet"}. '
            f'The prespecified cross-task performance criterion was '
            f'{"met" if audit["cross_task_performance_gate"] else "unmet"}. ')
    if not audit['cross_task_performance_gate']:
        text += ('The evidence supports reporting action sensitivity, outcome-dependent switch '
                 'value and their limitations. A general improvement claim would require '
                 'replication across more independent tasks and language realizations.')
    write('discussion_results.tex',text)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TECTONIC_CACHE_DIR=str(ROOT/'runtime/cache/tex'))
    tool = ROOT/'runtime/tools/tex_tools/tectonic/tectonic.exe'
    p = subprocess.run([str(tool),'--outdir',str(PAPER),str(PAPER/'confirmation.tex')],
                       env=env,cwd=PAPER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       encoding='utf-8',errors='replace')
    (LOGS/'tectonic.log').write_text(p.stdout,encoding='utf-8')
    p.check_returncode()
    subprocess.run(['pdftoppm','-png','-r','115',str(PAPER/'confirmation.pdf'),str(LOGS/'page')],check=True)


if __name__=='__main__':
    main()
