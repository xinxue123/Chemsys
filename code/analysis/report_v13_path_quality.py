"""English report and figures from the completed read-only path audit."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/v13_path_quality_audit_20261104'
TASKS=('c1','c2','c3','c4','perera','doyle','snar','hplc')
LABELS=dict(c1='Suzuki Case 1',c2='Suzuki Case 2',c3='Suzuki Case 3',c4='Suzuki Case 4',
            perera='Perera Suzuki',doyle='Doyle Buchwald-Hartwig',snar='SnAr',hplc='HPLC')


def rows(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def num(value,digits=2):
    return f'{float(value):.{digits}f}' if value not in ('',None) else 'not reached'


def main():
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    assert audit['status']=='PASS_READ_ONLY'
    paths=rows('v13_trajectory_profiles.csv');pairs=rows('high_low_pairs.csv')
    design=rows('design_prefixes.csv');gp=rows('GP_prefixes.csv');rounds=rows('v13_rounds.csv')
    common=rows('paired_GP_common_pool.csv')
    (OUT/'cases').mkdir(exist_ok=True)
    facts={}
    for regime in ('cold','warm'):
        count=0
        for pair in [r for r in pairs if r['regime']==regime]:
            h=next(r for r in design if (r['regime'],r['task'],r['seed'],r['observations'])==
                   (regime,pair['task'],pair['high_seed'],'8'))
            l=next(r for r in design if (r['regime'],r['task'],r['seed'],r['observations'])==
                   (regime,pair['task'],pair['low_seed'],'8'))
            count+=float(h['mean_Gower_distance'])>float(l['mean_Gower_distance'])
        facts[regime+'_higher_quality_more_Gower_dispersion_at8']=int(count)
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        for task,axis in zip(TASKS,axes.flat):
            pair=next(r for r in pairs if r['regime']==regime and r['task']==task)
            high,low=pair['high_seed'],pair['low_seed']
            for r in [r for r in paths if r['regime']==regime and r['task']==task and r['seed'] not in (high,low)]:
                best=np.array(json.loads(r['best_sequence']))
                start=0 if regime=='cold' else 3
                axis.plot(range(start+1,21),best[start:],color='#bfc4c7',lw=1,alpha=.7,zorder=1)
            for seed,color,label in [(high,'#287b65','Higher terminal / faster tie'),(low,'#b65b41','Lower terminal / slower tie')]:
                r=next(r for r in paths if (r['regime'],r['task'],r['seed'])==(regime,task,seed))
                best=np.array(json.loads(r['best_sequence']))
                start=0 if regime=='cold' else 3
                axis.plot(range(start+1,21),best[start:],color=color,lw=2,
                          label=f"seed{seed}: AUC {float(r['SearchAUC']):.1f}",zorder=3)
            r=next(r for r in paths if (r['regime'],r['task'],r['seed'])==(regime,task,high))
            maximum=float(r['final_best'])/float(r['final_fraction_of_max'])
            axis.set_ylim(0,maximum*1.07)
            axis.set_xlim(1 if regime=='cold' else 4,20)
            axis.set_xticks([1,5,10,15,20] if regime=='cold' else [4,8,12,16,20])
            title=LABELS[task]+(' (equal final)' if pair['comparison'].startswith('speed_only') else '')
            axis.set_title(title,fontsize=11)
            axis.grid(axis='y',alpha=.18)
            axis.spines[['top','right']].set_visible(False)
            axis.legend(fontsize=8,frameon=False,loc='lower right')
            axis.set_ylabel('Cumulative best (task units)',fontsize=9)
            axis.set_xlabel('Total observations',fontsize=9)
        fig.suptitle('V13 archived trajectories: '+('zero start, 0 + 20' if regime=='cold' else 'four initial observations + 16 queries'),fontsize=16)
        fig.text(.5,.012,'Green/red: largest/smallest terminal objective, ties broken by SearchAUC; gray: other seeds.\nDescriptive outcome-selected comparisons. Units differ by task; cohorts are kept separate.',ha='center',fontsize=10)
        fig.tight_layout(rect=[0,.065,1,.95])
        fig.savefig(OUT/f'{regime}_path_comparison.png',dpi=170)
        plt.close(fig)
    for pair in pairs:
        regime,task=pair['regime'],pair['task']
        high,low=pair['high_seed'],pair['low_seed']
        a=[r for r in rounds if (r['regime'],r['task'],r['seed'])==(regime,task,high)]
        b=[r for r in rounds if (r['regime'],r['task'],r['seed'])==(regime,task,low)]
        lines=[f'# {LABELS[task]}: {regime} path comparison','',
            f"Higher terminal / faster tied path: seed{high}; lower terminal / slower tied path: seed{low}.",
            f"Comparison type: {pair['comparison']}. These are outcome-selected examples, not matched-prefix counterfactuals.",'',
            f"Final objectives: {num(pair['high_final'])} / {num(pair['low_final'])}; SearchAUC: {num(pair['high_SearchAUC'])} / {num(pair['low_SearchAUC'])}.",'',
            '| Total observations | High selected | High objective / incumbent after | High own-GP reference / menu best | Low selected | Low objective / incumbent after | Low own-GP reference / menu best |',
            '|---|---|---:|---:|---|---:|---:|']
        for ra,rb in zip(a,b):
            assert ra['total_observations']==rb['total_observations']
            lines.append(f"| {ra['total_observations']} | {ra['candidate_id']} | {num(ra['objective'])} / {num(ra['cumulative_best'])} | {num(ra['reference_objective'])} / {num(ra['menu_best'])} | {rb['candidate_id']} | {num(rb['objective'])} / {num(rb['cumulative_best'])} | {num(rb['reference_objective'])} / {num(rb['menu_best'])} |")
        lines.extend(['','Reference/menu yields are retrospectively evaluated at each path\'s own history. They were not given to the LLM and are not alternative full trajectories. Complete conditions, public rationales, KF and event flags are in v13_rounds.csv.'])
        (OUT/'cases'/f'{regime}_{task}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    lines=['# What distinguishes high- and low-reward V13 paths?','',
        'This read-only audit checks all64 archived V13 trajectories across eight tasks and four starting seeds, with192 total V13/BO/Reasoning-BO paths as context. Cold0+20 and warm4+16 are separate cohorts. No GP fit, LLM call or sequential experiment is rerun. V13 uses its historical native empirical-scaling GP, not the later eight-query fixed50%-prior recipe.','',
        '**The logs show multiple mechanisms, not a universal diversity rule: early outcome range and GP uncertainty, selection among the actual offered candidates, and whether later experiments escape the initial low-value region. KF activity alone does not identify successful paths.**','',
        'Within each task/cohort the displayed extremes maximize/minimize terminal objective, with SearchAUC as a tie-break. Low means lowest among these four observed paths, not a declared chemical failure. Final-quality ties are explicitly treated as speed comparisons. One LLM realization per seed combines initialization and response variation; these examples are observational, not causal module ablations.','',
        '## Cold-start comparisons','',
        '| Task | High / low seeds | Terminal objective, high / low | SearchAUC, high / low | First90%-of-max observation, high / low |',
        '|---|---|---:|---:|---:|']
    for r in pairs:
        if r['regime']!='cold':continue
        label=LABELS[r['task']]+(' [terminal tie]' if r['comparison'].startswith('speed_only') else '')
        lines.append(f"| [{label}](cases/cold_{r['task']}.md) | {r['high_seed']} / {r['low_seed']} | {num(r['high_final'])} / {num(r['low_final'])} | {num(r['high_SearchAUC'])} / {num(r['low_SearchAUC'])} | {r['high_first90'] or 'not reached'} / {r['low_first90'] or 'not reached'} |")
    lines.extend(['','Objectives retain task units: Suzuki emulator yield, Perera product UV area, Doyle yield, SnAr productivity and HPLC peak area. They are not averaged across tasks. SearchAUC averages20 running maxima in the cold cohort,16 post-initial maxima in the warm cohort.','',
        '![Cold paths](cold_path_comparison.png)','',
        '## 1. Early outcome range and uncertainty can reinforce the path','',
        'Cold Suzuki1/seed2 starts35.41,8.21,44.01,57.20, while seed0 starts6.01,11.62,11.37,12.10. At four observations, the first path has empirical response SD17.93 percentage points and common-unqueried-pool mean latent SD19.07; the slow path has response SD2.48 and mean latent SD1.58. The categorical kernel correlations are0.000033 and0.999979, respectively. The slow path is already nearly treating catalysts as equivalent while giving narrow uncertainty after seeing only low outcomes.',
        'The old native implementations center and scale responses by their empirical prefix SD and then fit GP hyperparameters. In physical units, posterior SD equals that prefix output scale times standardized posterior SD (apart from the separately declared singleton/constant-output reporting convention). A narrow observed response range and a fitted weak signal can therefore produce unjustifiably narrow physical uncertainty outside the observed region. A richer outcome range is not itself proof of representative data; the reported calibration failures are evaluated against the frozen pool.',
        'HPLC offers a second example with similar first responses: seed2 starts44.55, while seed0 starts49.99. Yet their first-four best values are1088.92 versus58.59. At that prefix the response SDs are436.60 versus23.28, and mean common-pool latent SDs490.48 versus24.79. Empirical +/-1.96 latent-SD coverages are91.8% versus34.9%. Thus an initially higher response does not determine which path later supplies the GP with useful scale and condition information.',
        'Early category merging is not universally permanent: among16 cold Suzuki paths it occurs in5/16 at four observations,1/16 at eight, and0/16 at twelve/nineteen, using the descriptive0.99 correlation cutoff. Additional data often change the fitted kernel, but a recovered late model cannot retroactively recover the lost SearchAUC area.','',
        '## 2. GP candidate supply and LLM choice must be distinguished','',
        'In slow Suzuki1/seed0, a >=95%-of-pool-max candidate first appears in the GP top-five menu on query7; it is not selected until query20. On query8 the own-GP reference would give84.16%, while V13 selects46.51%. Across that whole path there are five missed own-GP record opportunities; the faster seed2 path has zero. These are one-step offline opportunities under the actual prefix, not the effect of swapping a point and continuing.',
        'Perera/seed2 reaches only34.88 after four observations. Queries3-5 perform focused single-factor comparisons around this weak incumbent, obtaining22.24,24.48 and39.21. On query7 the own-GP reference is95.34 while V13 chooses73.70. That is an actual improvement over its47.47 incumbent, so a binary improvement-based feedback system can reinforce the choice despite missing a much better offered point. The wider electrophile note also does not distinguish the whole menu. This is a decision-value/credit-assignment issue, not evidence that the KF arithmetic failed.',
        'For low-terminal HPLC/seed0, no >=95%-of-global-max point ever appears in its20 menus. However its early menus still contain1125.24 on query1 and1489.94 on query3, while V13 chooses49.99 and58.59. Both the evolving GP shortlist and its LLM reranking contribute possible bottlenecks. Successful paths also miss offered good points; success does not certify every earlier action.','',
        '## 3. High terminal quality can come from a late escape','',
        'Doyle/seed1 remains at34.44 through query8, then improves to83.31 on query9,89.56 on query12,92.35 on query13 andapproximately100 on query14. It has no structured-note forecast or KF update anywhere. Doyle/seed3 obtains87.90 by query7 and93.31 on query11, but never exceeds that level. Seed1 has the higher terminal value yet the LOWER SearchAUC70.56 versus76.50.',
        'The query9 jump changes three categorical factors relative to seed1\'s current incumbent. Queries11-14 each change one factor relative to the updated incumbent. This observed sequence moves from a new joint combination into focused comparisons after finding a better reference; it does not isolate the individual causal effects of the three simultaneous query9 changes.',
        'Doyle/seed0 supplies a different success: query6 selects99.62 while its own-GP reference would give1.92. This path also has no KF feedback. These examples preserve the LLM\'s useful candidate-level interventions while showing that the distinctive KF loop is not necessary for every observed gain. They do not establish why a freshly sampled alternative LLM response would choose the same point.',
        'SnAr\'s displayed high/low terminal objectives differ by onlyabout0.6% of the pool maximum, yet SearchAUC differs substantially because the high path reaches its best on query7 and the lower path onlyapproaches its terminal result on query20. The low label should not be read as a poor final chemical objective in that case.','',
        '## 4. Geometric diversity and contrast counts have no uniform ordering','',
        f"At eight observations, the higher-terminal/faster-tied cold path has greater nominal-aware Gower dispersion in only{facts['cold_higher_quality_more_Gower_dispersion_at8']}/8 comparisons; the warm cohort gives{facts['warm_higher_quality_more_Gower_dispersion_at8']}/8. Categories are compared by equality, never by numerical ID distance.",
        'Cold Suzuki1\'s faster path is less dispersed (0.242 versus0.412), with no exact one-factor pair in either first-eight prefix. Perera\'s stronger-terminal path is also less dispersed (0.429 versus0.571) but has more one-factor pairs (12 versus6). Conversely Doyle\'s highest-terminal path has onlytwo first-eight one-factor pairs, versusseven in the lower-terminal path. A blanket requirement to add dispersion or another contrast is not supported.',
        'What the contrast tests, whether its reference is already promising, and the opportunity cost within the remaining budget matter. This is consistent with the separate single-catalyst-control experiment: a contrast that reveals a useful difference can repair a path, whereas comparing two similarly low-performing catalysts can consume an early high-value opportunity. The retrospective pattern is not a causal decomposition of diversity versus outcomes.','',
        '## 5. KF and editor counts do not separate good and bad paths','',
        'Cold Suzuki1\'s faster and slower paths both have three KF updates. Doyle/seed1 reaches the highest terminal value withzero updates, while lower-terminal seed3 hasnine. HPLC/seed0 is poor withzero updates as well. Thus neither absence nor presence of KF explains all path outcomes. More scoring can also reflect repeated exposure to an unresolved weak region.',
        'Permissions, calibration and event execution replay correctly, but a note must cover actual alternatives and affect a useful selection to earn a mechanism claim. A positive Brier update or editor call is not itself evidence of improved next-choice ranking. Existing historical ablations remain relevant only within their own matched-prefix and response-backend scope.','',
        '## Warm-start cross-check','',
        '| Task | High / low seeds | Initial best, high / low | Terminal objective, high / low | SearchAUC, high / low |',
        '|---|---|---:|---:|---:|---:|'])
    for pair in [r for r in pairs if r['regime']=='warm']:
        h=next(r for r in paths if (r['regime'],r['task'],r['seed'])==('warm',pair['task'],pair['high_seed']))
        l=next(r for r in paths if (r['regime'],r['task'],r['seed'])==('warm',pair['task'],pair['low_seed']))
        lines.append(f"| [{LABELS[pair['task']]}](cases/warm_{pair['task']}.md) | {pair['high_seed']} / {pair['low_seed']} | {num(h['initial_best'])} / {num(l['initial_best'])} | {num(pair['high_final'])} / {num(pair['low_final'])} | {num(pair['high_SearchAUC'])} / {num(pair['low_SearchAUC'])} |")
    lines.extend(['','Warm SnAr is an important counterexample to an initial-value-only story: the higher-terminal seed0 starts from a four-point best3133.27, BELOW lower-terminal seed3\'s4701.07, but reaches11559.61 versus10938.30. Warm HPLC starts from fairly similar initial best1879.79 versus1786.32, yet ends2569.88 versus1871.86. The initial value often helps, but the later path remains consequential.',
        'Warm Suzuki1/2 also tie in final quality across all four seeds. Warm Suzuki4 shows another rank reversal: the lowest terminal objective can accompany a higher SearchAUC because it starts with an excellent incumbent. Warm seed0 Reasoning-BO controls have a historical API backend, unlike the fresh-subagent V13 runs; other methods are contextual references, not clean module ablations.','',
        '![Warm paths](warm_path_comparison.png)','',
        '## Evidence and limits','',
        f"The audit checks{audit['method_trajectories_checked']} original method states and{audit['measured_history_records_checked']} stored measured-history records, replays all{audit['V13_commit_feedback_replays']} V13 prestate/commit/feedback transitions, and reads{audit['GP_prefix_profiles']} archived GP prefixes. Source/state/GP-array hashes and available-candidate masks are checked. No model is refitted and no candidate is newly queried.",
        'Covariance decoding is checked against a native synthetic two-input kernel evaluation, with no fitting. Empirical interval coverage on the finite unqueried pool is a diagnostic, not a formal coverage guarantee. Good-region underestimation and common-pool comparisons are in paired_GP_common_pool.csv.',
        'Four seeds per task, with independent LLM responses, do not isolate initialization, response noise, GP scaling and knowledge mechanisms. The displayed paths are chosen by observed outcomes, and post-hoc contrasts are not randomized interventions. Findings explain these recorded paths; they do not validate a new universal policy.',
        'All64 V13 profiles and full original public rationales are retained in v13_trajectory_profiles.csv and v13_rounds.csv. All16 displayed pairs have separate chronological tables under cases/. Other files: all_method_metrics.csv, high_low_pairs.csv, design_prefixes.csv, GP_prefixes.csv, paired_GP_common_pool.csv, audit.json and source_hashes.json.',
        'Reproduce: run code/analysis/audit_v13_path_quality.py, then code/analysis/report_v13_path_quality.py with runtime/environments/.venv-official/Scripts/python.exe.'])
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'descriptive_facts.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',report=str(OUT/'README.md'),case_tables=16,**facts)))


if __name__=='__main__':
    main()
