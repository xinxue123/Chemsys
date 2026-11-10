"""Mechanism study of the hypothesis-scope controller, from saved evidence.

Standard library only. It issues no model calls, queries or GP fits. Rules come from
docs/hypothesis_scope_v2_registration_20261105.md, and the claim coding is frozen and hashed
before any verdict. Pool values are read as an evaluator and were never available to the
controller. Effects are % of task pool maximum. Everything here is exploratory.

    python code/analysis/analyze_hypothesis_scope_v2_mechanism.py <v2 run folder> <output folder>
"""
import csv
import hashlib
import json
import math
import random
import re
import statistics as st
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from analyze_hypothesis_scope_method import load_npz, spearman
from analyze_hypothesis_scope_v2_gp_followups import DEV, SEED, metrics, paired, read, rows_of, sign_flip, tops, write_rows

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = ROOT/'archives'
FOLDER = ROOT/'results/hypothesis_scope_v2_mechanism_20261106'
CALIPER, SENSITIVITY = .10, (.05, .20)
EFFECT, MIN_CONTRASTS = 2., 2
UNITS = [(t, s) for t in DEV for s in range(4)] + [('aryl', s) for s in range(8)]
MOLECULAR = re.compile(r'oxidative addition|C[–-]H (?:activation|cleavage)|carboxylate-assisted|transmetal|reductive elimination|speciation|'
                       r'boron activation|boronate activation|chloride activation|coordination|electron-rich|deactivation|decomposition|degradation|'
                       r'catalyst activity|catalyst stability|catalyst behaviou?r|protodeboron|thermal activation|activation or catalyst', re.I)
HEDGE = re.compile(r'confounded|not established|unestablished|not establish|does not isolate|do not isolate|untested|unverified|provisional|conjecture|'
                   r'not an observed|unproven|uncertain|possible explanation|not separately|not a claim|not evidence|cannot establish|unknown', re.I)
LABELS = {
    'O=C([O-])C(C)(C)C.[Cs+]': 'CsOPiv', 'O=C([O-])C(C)(C)C.[K+]': 'KOPiv', 'O=C([O-])C.[Cs+]': 'CsOAc', 'O=C([O-])C.[K+]': 'KOAc',
    'CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)C(OC)=CC=C2OC': 'BrettPhos', 'CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C': 'XPhos',
    'CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C': 'PtBu2Ph', 'C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1': 'CgPPh',
    'P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3': 'PPh3', 'CP(C1=CC=CC=C1)C2=CC=CC=C2': 'PMePh2', 'P(C1=CC=CO1)(C2=CC=CO2)C3=CC=CO3': 'P(2-furyl)3',
    'FC(F)(F)C1=CC(P(C2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C(OC)=CC=C2OC)C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C(F)(F)F)=C1': 'JackiePhos',
    'CC(N(C)C)=O': 'DMAc', 'CCCCOC(C)=O': 'BuOAc', 'CC1=CC=C(C)C=C1': 'p-xylene', 'CCCC#N': 'PrCN',
    'BrC1=CC=C(N=CC=C2)C2=C1': 'Ar-Br', 'ClC1=CC=C(N=CC=C2)C2=C1': 'Ar-Cl', 'IC1=CC=C(C(F)(F)F)C=C1': '4-CF3-ArI', 'ClC1=CC=C(C(F)(F)F)C=C1': '4-CF3-ArCl',
    'IC1=CC=C(OC)C=C1': '4-MeO-ArI', 'ClC1=CC=C(OC)C=C1': '4-MeO-ArCl', 'P1-L1': 'XPhos', 'P1-L2': 'SPhos', 'P1-L3': 'RuPhos', 'P1-L4': 'Xantphos', 'P1-L5': 'PCy3',
    'P1-L6': 'PPh3', 'P1-L7': 'PtBu3', 'P2-L1': 'P2-XPhos'}


class Pool:
    def __init__(self, ids, points, y, top):
        self.ids, self.points, self.y, self.top = ids, points, y, top
        self.ix = {c: i for i, c in enumerate(ids)}
        self.axes = list(points[0])
        self.exact = {a: isinstance(points[0][a], str) or len({p[a] for p in points}) <= 12 for a in self.axes}
        numeric = [a for a in self.axes if not isinstance(points[0][a], str)]
        self.width = {a: max(p[a] for p in points) - min(p[a] for p in points) for a in numeric}


def load_pools(base):
    top, pools = tops(), {}
    for task in DEV:
        rows = rows_of(ROOT/'data/pools'/task/'candidates.csv')
        axes = [k for k in rows[0] if k not in ('candidate_id', 'objective_raw', 'objective_normalized')]

        def cast(value):
            try:
                return float(value)
            except ValueError:
                return value
        pools[task] = Pool([r['candidate_id'] for r in rows], [{a: cast(r[a]) for a in axes} for r in rows], [float(r['objective_raw']) for r in rows], top[task])
    source = read(base/'heldout/input/pool.json')
    oracle = load_npz((base/'heldout/input/oracle.npz').read_bytes())
    pools['aryl'] = Pool(source['ids'], source['points'], oracle['raw_values'], top['aryl'])
    return pools


def member(value, spec):
    if any(isinstance(v, str) for v in spec):
        return value in spec
    low, high = spec
    return (low is None or value >= low - 1e-9) and (high is None or value <= high + 1e-9)


def in_scope(point, scope):
    return all(member(point[a], s['allowed'] if 'allowed' in s else s['bounds']) for a, s in scope.items())


def conditions(claim, contrast):
    return contrast['within'] if 'within' in contrast else {a: s for a, s in claim['scope'].items() if a != contrast['axis']}


def contrast_sets(pool, index, claim, contrast, caliper):
    """Independent matched contrasts: each is a pair of member lists, claimed higher and claimed lower."""
    axis, points = contrast['axis'], pool.points
    keep = [j for j in index if in_scope(points[j], conditions(claim, contrast))]
    high = [j for j in keep if member(points[j][axis], contrast['high'])]
    low = [j for j in keep if member(points[j][axis], contrast['low'])]
    others = [a for a in pool.axes if a != axis]
    fixed = [a for a in others if pool.exact[a]]
    loose = [a for a in others if not pool.exact[a]]
    key = lambda j: tuple(points[j][a] for a in fixed)
    groups = defaultdict(lambda: ([], []))
    for j in high:
        groups[key(j)][0].append(j)
    for j in low:
        groups[key(j)][1].append(j)
    out = []
    for hs, ls in groups.values():
        if not hs or not ls:
            continue
        if not loose:
            if 'pairs' in contrast:
                for a, b in contrast['pairs']:
                    one, two = [j for j in hs if points[j][axis] == a], [j for j in ls if points[j][axis] == b]
                    if one and two:
                        out.append((one, two))
            else:
                out.append((hs, ls))
            continue
        used = set()
        for j in hs:
            best = None
            for i in ls:
                if i in used:
                    continue
                gaps = [abs(points[j][a] - points[i][a])/pool.width[a] for a in loose]
                if max(gaps) <= caliper:
                    distance = sum(g*g for g in gaps)
                    if best is None or distance < best[0]:
                        best = (distance, i)
            if best:
                used.add(best[1])
                out.append(([j], [best[1]]))
    return out


def effect(pool, pair):
    return 100*(st.mean(pool.y[j] for j in pair[0]) - st.mean(pool.y[j] for j in pair[1]))/pool.top


def verdict(effects):
    if len(effects) < MIN_CONTRASTS:
        return 'no_contrast'
    wins, losses, mean = sum(d > 1e-9 for d in effects), sum(d < -1e-9 for d in effects), st.mean(effects)
    if mean >= EFFECT and wins >= 2*losses:
        return 'supported'
    if mean <= -EFFECT and losses >= 2*wins:
        return 'contradicted'
    return 'mixed'


def combine(parts):
    if all(p == 'no_contrast' for p in parts):
        return 'no_contrast'
    if all(p == 'supported' for p in parts):
        return 'supported'
    if 'contradicted' in parts and 'supported' not in parts:
        return 'contradicted'
    return 'mixed'


def level_verdict(pool, claim):
    rule = claim['level']
    values = [y for p, y in zip(pool.points, pool.y) if in_scope(p, claim['scope'])]
    threshold, stat = rule['threshold'], rule['stat']
    above = sum(v > threshold for v in values)
    passed = dict(exists_above=above > 0, none_above=above == 0, typical_above=st.median(values) >= threshold, typical_below=st.median(values) < threshold,
                  mean_above=st.mean(values) >= threshold)[stat]
    return ('supported' if passed else 'contradicted'), len(values), above/len(values), st.median(values)


def histories(base):
    """Commits by run, with each query's selected candidate."""
    seed0 = zipfile.ZipFile(ARCHIVES/'run__hypothesis_scope_loop_20261105.zip')
    rest = zipfile.ZipFile(ARCHIVES/'run__hypothesis_scope_multiseed_20261105.zip')
    runs = {}
    for task in DEV:
        for seed in range(4):
            archive, folder = (seed0, f'experiments/runs/hypothesis_scope_loop_20261105/runs/{task}') if seed == 0 else \
                (rest, f'experiments/runs/hypothesis_scope_multiseed_20261105/runs/s{seed}/{task}_hyp')
            runs[task, seed] = [json.loads(archive.read(f'{folder}/rounds/{q:02d}_commit.json').decode('utf-8-sig')) for q in range(1, 21)]
    for seed in range(8):
        folder = base/'heldout/runs/h1'/f's{seed}'/'aryl/rounds'
        runs['aryl', seed] = [read(folder/f'{q:02d}_commit.json') for q in range(1, 21)]
    return runs


def label(values):
    return '/'.join(LABELS.get(v, str(v)) for v in values) if any(isinstance(v, str) for v in values) else f'{values[0]}-{values[1]}'


def audit(pools, runs, coding):
    claims, contrasts = [], []
    for c in coding:
        pool = pools[c['task']]
        everything = range(len(pool.ids))
        parts, pool_effects = [], []
        for spec in c['contrasts']:
            pairs = contrast_sets(pool, everything, c, spec, CALIPER)
            effects = [effect(pool, p) for p in pairs]
            part = verdict(effects)
            parts.append(part)
            pool_effects += effects
            row = dict(key=c['key'], task=c['task'], group=c['group'], origin='pre-data' if c['installed_query'] == 1 else 'refreshed', direction=c['direction'], axis=spec['axis'],
                       high=label(spec['high']), low=label(spec['low']), contrasts=len(effects), mean_effect=st.mean(effects) if effects else '',
                       wins=sum(d > 1e-9 for d in effects), losses=sum(d < -1e-9 for d in effects), ties=sum(abs(d) <= 1e-9 for d in effects), verdict=part)
            for caliper in SENSITIVITY:
                row[f'verdict_caliper_{int(100*caliper)}'] = verdict([effect(pool, p) for p in contrast_sets(pool, everything, c, spec, caliper)])
            contrasts.append(row)
        pool_verdict = combine(parts) if parts else None
        level = level_verdict(pool, c) if c['level'] else None
        streams = range(4) if c['seed'] == 'all' else [int(c['seed'])]
        for seed in streams:
            commits = runs[c['task'], seed]
            order = [pool.ix[x['decision']['selected_id']] for x in commits]
            query = {j: i + 1 for i, j in enumerate(order)}
            wins = losses = total = 0
            for spec in c['contrasts']:
                for pair in contrast_sets(pool, order, c, spec, CALIPER):
                    if max(query[j] for j in pair[0] + pair[1]) >= c['installed_query']:
                        d = effect(pool, pair)
                        total += 1
                        wins += d > 1e-9
                        losses += d < -1e-9
            run_verdict = 'retained' if wins >= 2 and losses < 2 else 'refuted' if losses >= 2 and wins < 2 else 'unresolved'
            states = [n for x in commits for n in x['working_knowledge']['notes'] if n['semantic_key'] == c['semantic_key']]
            active = sum(c['id'] in x['decision'].get('active_hypothesis_ids', []) for x in commits)
            in_scope_after = sum(in_scope(pool.points[j], c['scope']) and query[j] >= c['installed_query'] for j in order)
            claims.append(dict(key=c['key'], stream=seed, group=c['group'], task=c['task'], origin='pre-data' if c['installed_query'] == 1 else 'refreshed', direction=c['direction'],
                               kind='contrast' if c['contrasts'] else 'level', axes=';'.join(s['axis'] for s in c['contrasts']), pool_verdict=pool_verdict or '',
                               pool_contrasts=len(pool_effects), pool_mean_effect=st.mean(pool_effects) if pool_effects else '',
                               level_stat=c['level']['stat'] if c['level'] else '', level_verdict=level[0] if level else '', level_share_above=level[2] if level else '',
                               prospective_contrasts=total, prospective_wins=wins, prospective_losses=losses, run_verdict=run_verdict if c['contrasts'] else '',
                               ever_paused=any(n['permission'] == 'paused' for n in states), final_permission=states[-1]['permission'] if states else '',
                               queries_active=active, in_scope_queries_after_install=in_scope_after, molecular_language=bool(MOLECULAR.search(c['claim'])), hedged=bool(HEDGE.search(c['claim']))))
    return claims, contrasts


def share(rows, test):
    return dict(n=len(rows), count=sum(map(test, rows)))


def summarize_claims(claims, contrasts, coding):
    distinct = {c['key']: c for c in claims}
    rows = list(distinct.values())
    out = dict(hypotheses=len(rows), runs_evaluated=len(claims), kind=dict(Counter(r['kind'] for r in rows)),
               kind_by_origin={o: dict(Counter(r['kind'] for r in rows if r['origin'] == o)) for o in ('pre-data', 'refreshed')},
               contrasts=len(contrasts), molecular_language=share(rows, lambda r: r['molecular_language']), hedged=share(rows, lambda r: r['hedged']),
               molecular_and_hedged=share([r for r in rows if r['molecular_language']], lambda r: r['hedged']))
    cc = [r for r in rows if r['kind'] == 'contrast']
    out['pool_verdicts'] = dict(Counter(r['pool_verdict'] for r in cc))
    for name, test in (('development', lambda r: r['group'] == 'dev'), ('held_out', lambda r: r['group'] == 'held'), ('pre_data', lambda r: r['origin'] == 'pre-data'), ('refreshed', lambda r: r['origin'] == 'refreshed'),
                       ('emulator', lambda r: r['task'] in ('c1', 'c2', 'c3', 'c4')), ('measured', lambda r: r['task'] in ('perera', 'doyle', 'aryl'))):
        out['pool_verdicts_' + name] = dict(Counter(r['pool_verdict'] for r in cc if test(r)))
    out['pool_verdicts_by_task'] = {t: dict(Counter(r['pool_verdict'] for r in cc if r['task'] == t)) for t in sorted({r['task'] for r in cc})}
    out['contrast_verdicts'] = dict(Counter(r['verdict'] for r in contrasts))
    out['contrast_verdicts_by_axis'] = {a: dict(Counter(r['verdict'] for r in contrasts if r['axis'] == a)) for a in sorted({r['axis'] for r in contrasts})}
    for caliper in SENSITIVITY:
        out[f'contrast_verdicts_caliper_{int(100*caliper)}'] = dict(Counter(r[f'verdict_caliper_{int(100*caliper)}'] for r in contrasts))
    lv = [r for r in rows if r['level_verdict']]
    out['level_verdicts'] = {s: dict(Counter(r['level_verdict'] for r in lv if r['level_stat'] == s)) for s in sorted({r['level_stat'] for r in lv})}
    ran = [c for c in claims if c['kind'] == 'contrast']
    out['in_run'] = dict(claim_runs=len(ran), none=sum(c['prospective_contrasts'] == 0 for c in ran), one=sum(c['prospective_contrasts'] == 1 for c in ran),
                         two_or_more=sum(c['prospective_contrasts'] >= 2 for c in ran), verdicts=dict(Counter(c['run_verdict'] for c in ran)), mean_prospective=st.mean(c['prospective_contrasts'] for c in ran))
    for name, test in (('development', lambda c: c['group'] == 'dev'), ('held_out', lambda c: c['group'] == 'held')):
        sub = [c for c in ran if test(c)]
        out['in_run_' + name] = dict(claim_runs=len(sub), two_or_more=sum(c['prospective_contrasts'] >= 2 for c in sub), verdicts=dict(Counter(c['run_verdict'] for c in sub)))
    decided = [c for c in ran if c['run_verdict'] != 'unresolved']
    agree = [c for c in decided if c['pool_verdict'] in ('supported', 'contradicted')]
    out['in_run_vs_pool'] = dict(decided=len(decided), with_clear_pool_verdict=len(agree),
                                 agree=sum((c['run_verdict'] == 'retained') == (c['pool_verdict'] == 'supported') for c in agree),
                                 table=dict(Counter(f"{c['run_verdict']}|{c['pool_verdict']}" for c in decided)))
    # The gate never reads a claim's direction, so its pauses are compared with pool verdicts.
    out['gate_vs_pool'] = {v: dict(claim_runs=sum(c['pool_verdict'] == v for c in ran), paused=sum(c['pool_verdict'] == v and c['ever_paused'] for c in ran),
                                   mean_queries_active=st.mean([c['queries_active'] for c in ran if c['pool_verdict'] == v] or [0])) for v in ('supported', 'mixed', 'contradicted', 'no_contrast')}
    return out


def translation(pools, coding, claims):
    """Compare two readings of a favour claim: keep only its scope, or remove only its claimed-lower side."""
    verdicts = {c['key']: c['pool_verdict'] for c in claims}
    rows = []
    for c in coding:
        if c['direction'] != 'favour' or not c['contrasts']:
            continue
        pool = pools[c['task']]
        n = len(pool.y)
        cut = sorted(pool.y, reverse=True)[max(1, int(.05*n)) - 1]
        best = max(pool.y)
        keep = [in_scope(p, c['scope']) for p in pool.points]
        drop = [any(member(p[s['axis']], s['low']) and in_scope(p, conditions(c, s)) for s in c['contrasts']) for p in pool.points]
        tops_ = [j for j in range(n) if pool.y[j] >= cut]
        maxima = [j for j in range(n) if pool.y[j] >= best - 1e-9]
        rest = [pool.y[j] for j in range(n) if not drop[j]]
        rows.append(dict(key=c['key'], task=c['task'], origin='pre-data' if c['installed_query'] == 1 else 'refreshed', verdict=verdicts[c['key']], scope_share=sum(keep)/n, dominated_share=sum(drop)/n,
                         scope_ceiling=100*max(y for y, k in zip(pool.y, keep) if k)/pool.top, dominance_ceiling=100*max(rest)/pool.top, scope_keeps_maximum=any(keep[j] for j in maxima),
                         dominance_keeps_maximum=any(not drop[j] for j in maxima), scope_top5_kept=sum(keep[j] for j in tops_)/len(tops_), dominance_top5_kept=sum(not drop[j] for j in tops_)/len(tops_),
                         scope_lift=(sum(keep[j] for j in tops_)/sum(keep))/(len(tops_)/n)))

    def block(sub):
        return dict(n=len(sub), scope_share=st.median(r['scope_share'] for r in sub), dominated_share=st.median(r['dominated_share'] for r in sub), scope_ceiling=st.median(r['scope_ceiling'] for r in sub),
                    dominance_ceiling=st.median(r['dominance_ceiling'] for r in sub), scope_keeps_maximum=sum(r['scope_keeps_maximum'] for r in sub), dominance_keeps_maximum=sum(r['dominance_keeps_maximum'] for r in sub),
                    scope_top5_kept=st.median(r['scope_top5_kept'] for r in sub), dominance_top5_kept=st.median(r['dominance_top5_kept'] for r in sub), scope_lift=st.median(r['scope_lift'] for r in sub))
    return rows, dict(all=block(rows), pre_data=block([r for r in rows if r['origin'] == 'pre-data']), refreshed=block([r for r in rows if r['origin'] == 'refreshed']),
                      supported=block([r for r in rows if r['verdict'] == 'supported']), mixed=block([r for r in rows if r['verdict'] == 'mixed']),
                      supported_pre_data=block([r for r in rows if r['verdict'] == 'supported' and r['origin'] == 'pre-data']))


def forecast_quality(pools, runs, rng):
    rows = defaultdict(list)
    for (task, seed), commits in runs.items():
        pool = pools[task]
        for c in commits:
            d = c['decision']
            y = pool.y[pool.ix[d['selected_id']]]
            installed = {n['semantic_key']: n['installed_query'] for n in c['working_knowledge']['notes']}
            for f in d.get('locked_forecasts', []):
                m, s = f['forecast']['mean_raw'], f['forecast']['sd_raw']
                z = (y - m)/s
                row = dict(z=z, log=-.5*z*z - math.log(s) - .5*math.log(2*math.pi), mean=100*m/pool.top, y=100*y/pool.top, origin='pre-data' if installed.get(f['semantic_key']) == 1 else 'refreshed', query=d['query'])
                if d.get('GP_mean_raw') is not None and d.get('GP_predictive_SD_raw'):
                    g, t = d['GP_mean_raw'], max(d['GP_predictive_SD_raw'], 1e-9)
                    w = (y - g)/t
                    row.update(gp_log=-.5*w*w - math.log(t) - .5*math.log(2*math.pi), gp_mean=100*g/pool.top, gp_z=w)
                if f.get('p_GP') is not None and f.get('incumbent_raw') is not None:
                    hit = float(y > f['incumbent_raw'] + 1e-12)
                    row.update(brier_llm=(f['p_LLM'] - hit)**2, brier_gp=(f['p_GP'] - hit)**2, improved=hit)
                rows[task].append(row)
    table = []
    for task, items in rows.items():
        paired_rows = [r for r in items if 'gp_log' in r]
        bri = [r for r in items if 'brier_llm' in r]
        table.append(dict(task=task, forecasts=len(items), within_1sd=st.mean(abs(r['z']) <= 1 for r in items), within_2sd=st.mean(abs(r['z']) <= 2 for r in items), mean_z=st.mean(r['z'] for r in items),
                          compared=len(paired_rows), llm_log=st.mean(r['log'] for r in paired_rows) if paired_rows else '', gp_log=st.mean(r['gp_log'] for r in paired_rows) if paired_rows else '',
                          log_gain_llm_minus_gp=st.mean(r['log'] - r['gp_log'] for r in paired_rows) if paired_rows else '',
                          gp_within_2sd=st.mean(abs(r['gp_z']) <= 2 for r in paired_rows) if paired_rows else '', scored=len(bri), brier_llm=st.mean(r['brier_llm'] for r in bri) if bri else '',
                          brier_gp=st.mean(r['brier_gp'] for r in bri) if bri else ''))
    every = [r for items in rows.values() for r in items]
    gains = [t['log_gain_llm_minus_gp'] for t in table if t['log_gain_llm_minus_gp'] != '']
    brier = [t['brier_gp'] - t['brier_llm'] for t in table if t['brier_llm'] != '']
    both = [r for r in every if 'gp_mean' in r]
    return table, dict(forecasts=len(every), within_1sd=st.mean(abs(r['z']) <= 1 for r in every), within_2sd=st.mean(abs(r['z']) <= 2 for r in every), mean_z=st.mean(r['z'] for r in every),
                       gp_within_2sd=st.mean(abs(r['gp_z']) <= 2 for r in both), llm_within_2sd_same_points=st.mean(abs(r['z']) <= 2 for r in both),
                       by_origin={o: dict(n=sum(r['origin'] == o for r in every), within_2sd=st.mean(abs(r['z']) <= 2 for r in every if r['origin'] == o), mean_z=st.mean(r['z'] for r in every if r['origin'] == o)) for o in ('pre-data', 'refreshed')},
                       tasks=len(gains), log_gain_mean=st.mean(gains), log_tasks_up=sum(g > 0 for g in gains), log_tasks_down=sum(g < 0 for g in gains), log_sign_flip_p=sign_flip(gains, rng),
                       brier_gain_mean=st.mean(brier), brier_tasks_up=sum(g > 0 for g in brier), brier_tasks_down=sum(g < 0 for g in brier), brier_sign_flip_p=sign_flip(brier, rng),
                       rank_llm_mean_vs_outcome=spearman([r['mean'] for r in both], [r['y'] for r in both]), rank_gp_mean_vs_outcome=spearman([r['gp_mean'] for r in both], [r['y'] for r in both]), rank_pairs=len(both))


def state_of(base, arm, task, seed):
    return read(base/'gp_followup'/arm/task/f's{seed}'/'state.json')


def ceilings(pools, base):
    rows = []
    for task, seed in UNITS:
        pool = pools[task]
        note = read(base/'gp_followup/p0'/task/f's{seed}'/'rounds/01_commit.json')['working_knowledge']['notes'][0]
        inside = [in_scope(p, note['scope']) for p in pool.points]
        keep = [y for y, i in zip(pool.y, inside) if (i if note['direction'] == 'favour' else not i)]
        top_ceiling = 100*max(keep)/pool.top
        on, _, _ = metrics(state_of(base, 'p0', task, seed), pool.top)
        off, off_final, _ = metrics(state_of(base, 'p0g', task, seed), pool.top)
        p0 = state_of(base, 'p0', task, seed)
        bound = sum(r['decision'].get('reason') == 'hypothesis_constrained_native_qLogEI' for r in p0['decisions'])
        rows.append(dict(task=task, stream=seed, direction=note['direction'], region_share=len(keep)/len(pool.y), ceiling_pct=top_ceiling, ceiling_gap=100 - top_ceiling, no_gate_final=off_final,
                         hit_ceiling=off_final >= top_ceiling - 1e-6, above_ceiling=off_final > top_ceiling + 1e-6, gate_gain=on - off, gate_queries_bound=bound))
    dev = [(st.mean(r['ceiling_gap'] for r in rows if r['task'] == t), st.mean(r['gate_gain'] for r in rows if r['task'] == t), st.mean(r['no_gate_final'] for r in rows if r['task'] == t)) for t in DEV]
    held = [r for r in rows if r['task'] == 'aryl']
    return rows, dict(runs=len(rows), hit_ceiling=sum(r['hit_ceiling'] for r in rows), above_ceiling=sum(r['above_ceiling'] for r in rows),
                      by_task={t: dict(ceiling=round(100 - a, 2), gate_gain=round(b, 2), no_gate_final=round(c, 2)) for t, (a, b, c) in zip(DEV, dev)},
                      rank_gap_vs_gain_dev=spearman([a for a, _, _ in dev], [b for _, b, _ in dev]), rank_gap_vs_gain_held=spearman([r['ceiling_gap'] for r in held], [r['gate_gain'] for r in held]),
                      rank_gap_vs_gain_all=spearman([a for a, _, _ in dev] + [r['ceiling_gap'] for r in held], [b for _, b, _ in dev] + [r['gate_gain'] for r in held]),
                      mean_gate_queries_dev=st.mean(r['gate_queries_bound'] for r in rows if r['task'] != 'aryl'), mean_gate_queries_held=st.mean(r['gate_queries_bound'] for r in held),
                      median_gate_queries=st.median(r['gate_queries_bound'] for r in rows))


def variance(pool):
    """Share of pool variance from additive main effects and from main plus two-way terms."""
    def fitted(terms):
        fit = [st.mean(pool.y)]*len(pool.y)
        parts = {t: [0.]*len(pool.y) for t in terms}
        for _ in range(60):
            for t in terms:
                groups = defaultdict(list)
                for j, p in enumerate(pool.points):
                    groups[tuple(p[a] for a in t)].append(pool.y[j] - fit[j] + parts[t][j])
                means = {k: st.mean(v) for k, v in groups.items()}
                for j, p in enumerate(pool.points):
                    new = means[tuple(p[a] for a in t)]
                    fit[j] += new - parts[t][j]
                    parts[t][j] = new
        return fit
    total = sum((v - st.mean(pool.y))**2 for v in pool.y)
    r2 = lambda fit: 1 - sum((a - b)**2 for a, b in zip(pool.y, fit))/total
    axes = pool.axes
    main = r2(fitted([(a,) for a in axes]))
    pairs = r2(fitted([(a, b) for i, a in enumerate(axes) for b in axes[i + 1:]]))
    shares = {}
    for a in axes:
        means = defaultdict(list)
        for p, v in zip(pool.points, pool.y):
            means[p[a]].append(v)
        shares[a] = sum(len(v)*(st.mean(v) - st.mean(pool.y))**2 for v in means.values())/total
    return dict(candidates=len(pool.y), additive_r2=main, pairwise_r2=pairs, axis_share=shares, share_zero=st.mean(v == 0 for v in pool.y))


def scoped_tests(base, pools, rng):
    """Tests C and T. Each contrast is arm a minus arm b, by task on development and by stream on held-out."""
    first_dev = {(r['task'], int(r['seed'])): r for r in rows_of(ROOT/'results/hypothesis_scope_v2_20261105/per_seed_results.csv') if r['arm'] == 'first_env'}

    def score(arm, task, seed):
        top = pools[task].top
        if arm == 'first':
            if task == 'aryl':
                a, f, _ = metrics(read(base/'heldout/runs/first'/f's{seed}'/'aryl/state.json'), top)
                return a, f
            r = first_dev[task, seed]
            return 100*float(r['SearchAUC'])/top, 100*float(r['final_best'])/top
        a, f, _ = metrics(state_of(base, arm, task, seed), top)
        return a, f

    rows = []
    for task, seed in UNITS:
        row = dict(data='held-out' if task == 'aryl' else 'development', task=task, stream=seed)
        for arm in ('first', 'p0', 'p0g', 'p0s', 'p0sg', 'p0t1', 'p0t3', 'p0t5', 'p0t10'):
            row[arm + '_auc'], row[arm + '_final'] = score(arm, task, seed)
        rows.append(row)

    def contrast(fn, data):
        sub = [r for r in rows if r['data'] == data]
        if data == 'development':
            auc = [st.mean(fn(r, '_auc') for r in sub if r['task'] == t) for t in DEV]
            final = [st.mean(fn(r, '_final') for r in sub if r['task'] == t) for t in DEV]
        else:
            auc, final = [fn(r, '_auc') for r in sub], [fn(r, '_final') for r in sub]
        return paired(auc, final, rng)

    gap = lambda a, b: (lambda r, k: r[a + k] - r[b + k])
    out = {}
    for data, name in (('development', 'development'), ('held-out', 'held_out')):
        sub = [r for r in rows if r['data'] == data]
        result = dict(C1_content_with_gate=contrast(gap('p0', 'p0s'), data), C2_content_without_gate=contrast(gap('p0g', 'p0sg'), data), C3_gate_on_placebo=contrast(gap('p0s', 'p0sg'), data),
                      gate_on_real=contrast(gap('p0', 'p0g'), data), interaction=contrast(lambda r, k: (r['p0s' + k] - r['p0sg' + k]) - (r['p0' + k] - r['p0g' + k]), data),
                      T_gate_vs_timer3=contrast(gap('p0', 'p0t3'), data), means={a: st.mean(r[a + '_auc'] for r in sub) for a in ('first', 'p0', 'p0g', 'p0s', 'p0sg', 'p0t1', 'p0t3', 'p0t5', 'p0t10')},
                      finals={a: st.mean(r[a + '_final'] for r in sub) for a in ('first', 'p0', 'p0g', 'p0s', 'p0sg', 'p0t1', 'p0t3', 'p0t5', 'p0t10')})
        if data == 'development':
            result['by_task'] = {t: {a: round(st.mean(r[a + '_auc'] for r in sub if r['task'] == t), 2) for a in ('first', 'p0', 'p0g', 'p0s', 'p0sg', 'p0t3')} for t in DEV}
        out[name] = result
    return rows, out


def second_query(base, pools):
    """Whether query 2 keeps any factor level of the first point, by arm, on categorical tasks."""
    out = {}
    for arm in ('p0', 'p0t1', 'p0s', 'q0g', 'q0'):
        rows = {}
        for task in ('aryl', 'doyle', 'perera'):
            streams = range(8) if task == 'aryl' else range(4)
            shared = []
            for seed in streams:
                path = base/'gp_followup'/arm/task/f's{seed}'/'state.json'
                if not path.exists():
                    continue
                d = read(path)['decisions']
                one, two = d[0]['decision']['conditions'], d[1]['decision']['conditions']
                shared.append(sum(one[a] == two[a] for a in one))
            if shared:
                rows[task] = dict(runs=len(shared), mean_shared_levels=st.mean(shared), runs_sharing_none=sum(v == 0 for v in shared))
        out[arm] = rows
    return out


def faithful_tests(base, pools, coding, rows, rng):
    """Rules Q1-Q4. `rows` already holds the reference arms for every unit."""
    keys = {(c['task'], s): c for c in coding if c['installed_query'] == 1 for s in (range(4) if c['seed'] == 'all' else [int(c['seed'])])}
    for r in rows:
        for arm in ('q0g', 'q0'):
            r[arm + '_auc'], r[arm + '_final'], _ = metrics(state_of(base, arm, r['task'], r['stream']), pools[r['task']].top)

    def contrast(fn, data):
        sub = [r for r in rows if r['data'] == data]
        if data == 'development':
            auc = [st.mean(fn(r, '_auc') for r in sub if r['task'] == t) for t in DEV]
            final = [st.mean(fn(r, '_final') for r in sub if r['task'] == t) for t in DEV]
        else:
            auc, final = [fn(r, '_auc') for r in sub], [fn(r, '_final') for r in sub]
        return paired(auc, final, rng)

    gap = lambda a, b: (lambda r, k: r[a + k] - r[b + k])
    evidence = []
    for (task, seed), claim in keys.items():
        if not claim['contrasts']:
            continue
        pool = pools[task]
        state = state_of(base, 'q0', task, seed)
        order = [pool.ix[h['candidate_id']] for h in state['history']]
        wins = losses = total = 0
        for spec in claim['contrasts']:
            for pair in contrast_sets(pool, order, claim, spec, CALIPER):
                d = effect(pool, pair)
                total += 1
                wins += d > 1e-9
                losses += d < -1e-9
        tests = sum(x['decision'].get('reason') == 'matched_contrast_test' for x in state['decisions'])
        evidence.append(dict(task=task, stream=seed, key=claim['key'], contrasts=total, wins=wins, losses=losses, tests=tests,
                             verdict='retained' if wins >= 2 and losses < 2 else 'refuted' if losses >= 2 and wins < 2 else 'unresolved'))
    out = {}
    for data, name in (('development', 'development'), ('held-out', 'held_out')):
        sub = [r for r in rows if r['data'] == data]
        result = dict(Q1_translation=contrast(gap('q0g', 'p0g'), data), Q2_translation_vs_first=contrast(gap('q0g', 'first'), data), Q4_tests_vs_first=contrast(gap('q0', 'first'), data),
                      q0_vs_p0=contrast(gap('q0', 'p0'), data), q0_vs_q0g=contrast(gap('q0', 'q0g'), data),
                      means={a: st.mean(r[a + '_auc'] for r in sub) for a in ('first', 'p0', 'p0g', 'q0g', 'q0')}, finals={a: st.mean(r[a + '_final'] for r in sub) for a in ('first', 'p0', 'p0g', 'q0g', 'q0')})
        mine = [e for e in evidence if (e['task'] == 'aryl') == (data == 'held-out')]
        result['Q3'] = dict(claim_runs=len(mine), two_or_more=sum(e['contrasts'] >= 2 for e in mine), one=sum(e['contrasts'] == 1 for e in mine), none=sum(e['contrasts'] == 0 for e in mine),
                            verdicts=dict(Counter(e['verdict'] for e in mine)), mean_tests=st.mean(e['tests'] for e in mine))
        if data == 'development':
            result['by_task'] = {t: {a: round(st.mean(r[a + '_auc'] for r in sub if r['task'] == t), 2) for a in ('first', 'p0', 'p0g', 'q0g', 'q0')} for t in DEV}
        else:
            result['by_stream'] = {r['stream']: {a: round(r[a + '_auc'], 2) for a in ('first', 'p0', 'p0g', 'q0g', 'q0')} for r in sub}
        out[name] = result
    write_rows(FOLDER/'faithful_evidence.csv', evidence)
    return out


def side_values(spec):
    return spec['allowed'] if 'allowed' in spec else spec['bounds']


def live_claims(base, pools):
    """Every contrast the live arm held, with its in-run evidence and its pool verdict."""
    rows = []
    for task, seed in UNITS:
        pool = pools[task]
        folder = base/'gp_followup/q1'/task/f's{seed}'
        state = read(folder/'state.json')
        last = read(folder/'rounds/20_commit.json')['working_knowledge']
        seen, first = {}, {}
        for note in last['archive'] + last['notes']:
            if note.get('contrast'):
                seen[note['semantic_key']] = note
                # A kept hypothesis is re-stamped at each refresh, so evidence counts from its first installation.
                first[note['semantic_key']] = min(first.get(note['semantic_key'], 99), note['installed_query'])
        order = [pool.ix[h['candidate_id']] for h in state['history']]
        query = {j: i + 1 for i, j in enumerate(order)}
        for key, note in seen.items():
            installed = first[key]
            record = note['contrast']
            axis = record['axis']
            spec = dict(axis=axis, high=side_values(record['high'][axis]), low=side_values(record['low'][axis]), within={a: s for a, s in record['low'].items() if a != axis})
            if record.get('pairs'):
                spec['pairs'] = record['pairs']
            claim = dict(scope=record['low'])
            effects = [effect(pool, p) for p in contrast_sets(pool, range(len(pool.ids)), claim, spec, CALIPER)]
            measured = [p for p in contrast_sets(pool, order, claim, spec, CALIPER) if max(query[j] for j in p[0] + p[1]) >= installed]
            run = [effect(pool, p) for p in measured]
            wins, losses = sum(d > 1e-9 for d in run), sum(d < -1e-9 for d in run)
            removed = [j for j, p in enumerate(pool.points) if in_scope(p, record['low'])]
            rows.append(dict(task=task, stream=seed, id=note['id'], origin='pre-data' if installed == 1 else 'refreshed', installed_query=installed, axis=axis,
                             prospective_contrasts=len(run), wins=wins, losses=losses, run_verdict='retained' if wins >= 2 and losses < 2 else 'refuted' if losses >= 2 and wins < 2 else 'unresolved',
                             pool_contrasts=len(effects), pool_mean_effect=st.mean(effects) if effects else '', pool_verdict=verdict(effects), paused=record['paused'],
                             removed_share=len(removed)/len(pool.y), removed_holds_maximum=any(pool.y[j] >= max(pool.y) - 1e-9 for j in removed)))
    return rows


def live_test(base, pools, rows, rng):
    """Rule Q5. Other arms are read as saved, and none is rerun."""
    saved = {(r['arm'], r['task'], int(r['seed'])): r for r in rows_of(ROOT/'results/hypothesis_scope_v2_20261105/per_seed_results.csv')}
    for r in rows:
        task, seed, top = r['task'], r['stream'], pools[r['task']].top
        r['q1_auc'], r['q1_final'], _ = metrics(state_of(base, 'q1', task, seed), top)
        state = read(folder := base/'gp_followup/q1'/task/f's{seed}'/'state.json')
        r['q1_calls'] = sum(c['query'] > 1 for c in state['calls'])
        r['q1_tests'] = sum(x['decision'].get('reason') == 'matched_contrast_test' for x in state['decisions'])
        if task == 'aryl':
            for arm in ('h1', 'bo', 'lgbo'):
                r[arm + '_auc'], r[arm + '_final'], _ = metrics(read(base/'heldout/runs'/arm/f's{seed}'/'aryl/state.json'), top)
        else:
            for arm, name in (('h1', 'h1'), ('bo_env', 'bo')):
                x = saved[arm, task, seed]
                r[name + '_auc'], r[name + '_final'] = 100*float(x['SearchAUC'])/top, 100*float(x['final_best'])/top
            r['lgbo_auc'] = r['lgbo_final'] = ''  # LGBO's same-backend run exists on the held-out task only.

    def contrast(a, b, data):
        sub = [r for r in rows if r['data'] == data]
        if data == 'development':
            return paired([st.mean(r[a + '_auc'] - r[b + '_auc'] for r in sub if r['task'] == t) for t in DEV], [st.mean(r[a + '_final'] - r[b + '_final'] for r in sub if r['task'] == t) for t in DEV], rng)
        return paired([r[a + '_auc'] - r[b + '_auc'] for r in sub], [r[a + '_final'] - r[b + '_final'] for r in sub], rng)

    claims = live_claims(base, pools)
    write_rows(FOLDER/'live_claims.csv', claims)
    usage = dict(requests=0, input_tokens=0, output_tokens=0, total_tokens=0)
    stated = Counter()
    for path in (base/'requests/q1').glob('*/*/validated.json'):
        saved_call = read(path)
        usage['requests'] += 1
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            usage[key] += saved_call['usage'].get(key, 0)
        for note in saved_call['raw']['hypotheses']:
            stated['contrast' if note.get('contrast') else 'no_contrast'] += 1
    out = dict(api_usage=usage, stated=dict(stated), claims=len(claims), two_or_more=sum(c['prospective_contrasts'] >= 2 for c in claims), pool_verdicts=dict(Counter(c['pool_verdict'] for c in claims)),
               removed_holds_maximum=sum(c['removed_holds_maximum'] for c in claims),
               by_origin={o: dict(claims=sum(c['origin'] == o for c in claims), two_or_more=sum(c['origin'] == o and c['prospective_contrasts'] >= 2 for c in claims),
                                  none=sum(c['origin'] == o and c['prospective_contrasts'] == 0 for c in claims)) for o in ('pre-data', 'refreshed')})
    for data, name in (('development', 'development'), ('held-out', 'held_out')):
        sub = [r for r in rows if r['data'] == data]
        mine = [c for c in claims if (c['task'] == 'aryl') == (data == 'held-out')]
        result = dict(q1_vs_h1=contrast('q1', 'h1', data), q1_vs_first=contrast('q1', 'first', data), q1_vs_bo=contrast('q1', 'bo', data), q1_vs_q0=contrast('q1', 'q0', data),
                      means={a: st.mean(r[a + '_auc'] for r in sub) for a in ('q1', 'h1', 'first', 'bo', 'q0')}, finals={a: st.mean(r[a + '_final'] for r in sub) for a in ('q1', 'h1', 'first', 'bo', 'q0')},
                      q1_sd=st.stdev(r['q1_auc'] for r in sub), mean_later_calls=st.mean(r['q1_calls'] for r in sub), mean_tests=st.mean(r['q1_tests'] for r in sub), claims=len(mine), two_or_more=sum(c['prospective_contrasts'] >= 2 for c in mine),
                      run_verdicts=dict(Counter(c['run_verdict'] for c in mine)), pool_verdicts=dict(Counter(c['pool_verdict'] for c in mine)),
                      verdict_table=dict(Counter(f"{c['run_verdict']}|{c['pool_verdict']}" for c in mine if c['run_verdict'] != 'unresolved')),
                      removed_holds_maximum=sum(c['removed_holds_maximum'] for c in mine), median_removed_share=st.median(c['removed_share'] for c in mine) if mine else None)
        if data == 'held-out':
            result['q1_vs_lgbo'] = contrast('q1', 'lgbo', data)
            result['means']['lgbo'] = st.mean(r['lgbo_auc'] for r in sub)
            result['by_stream'] = {r['stream']: {a: round(r[a + '_auc'], 2) for a in ('q1', 'h1', 'bo', 'q0')} for r in sub}
        else:
            result['by_task'] = {t: {a: round(st.mean(r[a + '_auc'] for r in sub if r['task'] == t), 2) for a in ('q1', 'h1', 'first', 'bo')} for t in DEV}
        out[name] = result
    return out


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    text = (FOLDER/'claim_coding.json').read_text(encoding='utf-8')
    frozen = (FOLDER/'claim_coding.sha256').read_text(encoding='utf-8').split()[0]
    assert hashlib.sha256(text.encode('utf-8')).hexdigest() == frozen, 'Claim coding changed after it was frozen'
    coding = json.loads(text)['hypotheses']
    rng = random.Random(SEED)
    pools = load_pools(base)
    runs = histories(base)
    claims, contrasts = audit(pools, runs, coding)
    write_rows(out/'claims.csv', claims)
    write_rows(out/'contrasts.csv', contrasts)
    translation_rows, translated = translation(pools, coding, claims)
    write_rows(out/'translation.csv', translation_rows)
    forecast_rows, forecasts = forecast_quality(pools, runs, rng)
    write_rows(out/'forecasts_by_task.csv', forecast_rows)
    ceiling_rows, ceiling = ceilings(pools, base)
    write_rows(out/'ceilings.csv', ceiling_rows)
    plan = read(ROOT/'results/hypothesis_scope_v2_gp_followups_20261106/gp_followups.json')['test_F']
    structure = {t: dict(variance(pools[t]), test_F_gap=plan[t]['gap_outside_minus_inside']) for t in ('aryl', 'doyle', 'perera')}
    result = dict(status='EXPLORATORY_SAVED_EVIDENCE', new_model_calls=0, claim_coding_sha256=frozen, claims=summarize_claims(claims, contrasts, coding), translation=translated, forecasts=forecasts, ceilings=ceiling, variance=structure)
    done = all((base/'gp_followup'/a/t/f's{s}'/'state.json').exists() for a in ('p0s', 'p0sg', 'p0t1', 'p0t3', 'p0t5', 'p0t10') for t, s in UNITS)
    if done:
        rows, tests = scoped_tests(base, pools, rng)
        result['tests'] = tests
        result['second_query'] = second_query(base, pools)
        if all((base/'gp_followup'/a/t/f's{s}'/'state.json').exists() for a in ('q0g', 'q0') for t, s in UNITS):
            result['faithful'] = faithful_tests(base, pools, coding, rows, rng)
            live = [base/'gp_followup/q1'/t/f's{s}'/'state.json' for t, s in UNITS]
            if all(p.exists() and read(p)['status'] == 'COMPLETE' for p in live):
                result['live'] = live_test(base, pools, rows, rng)
        write_rows(out/'mechanism_runs.csv', rows)
    (out/'mechanism.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('tests', 'faithful', 'live')}, indent=1, ensure_ascii=False))
    if 'live' in result:
        print('== LIVE usage', result['live']['api_usage'])
        for name in ('development', 'held_out'):
            r = result['live'][name]
            print('== LIVE', name, {k: round(v, 2) for k, v in r['means'].items()}, '| final', {k: round(v, 1) for k, v in r['finals'].items()}, '| calls', round(r['mean_later_calls'], 2), 'tests', round(r['mean_tests'], 2))
            for key in ('q1_vs_h1', 'q1_vs_first', 'q1_vs_bo', 'q1_vs_q0', 'q1_vs_lgbo'):
                if key in r:
                    c = r[key]
                    print(f"  {key:12s} {c['mean_auc']:+7.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']}/{c['tied']} p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f} rule={c['meets_rule']}")
            print('  claims', r['claims'], 'two+', r['two_or_more'], r['run_verdicts'], '| pool', r['pool_verdicts'], '| table', r['verdict_table'], '| removed holds max', r['removed_holds_maximum'], '| median removed', r['median_removed_share'])
            print('  detail', r.get('by_task') or r.get('by_stream'))
    for name, r in result.get('faithful', {}).items():
        print('== Q', name, {k: round(v, 2) for k, v in r['means'].items()}, '| final', {k: round(v, 1) for k, v in r['finals'].items()})
        for key in ('Q1_translation', 'Q2_translation_vs_first', 'Q4_tests_vs_first', 'q0_vs_p0', 'q0_vs_q0g'):
            c = r[key]
            print(f"  {key:26s} {c['mean_auc']:+7.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']}/{c['tied']} p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f} rule={c['meets_rule']}")
        print('  Q3', r['Q3'])
        print('  detail', r.get('by_task') or r.get('by_stream'))
    if done:
        for name, r in result['tests'].items():
            print('==', name, {k: round(v, 2) for k, v in r['means'].items()}, '| final', {k: round(v, 1) for k, v in r['finals'].items()})
            for key in ('C1_content_with_gate', 'C2_content_without_gate', 'C3_gate_on_placebo', 'gate_on_real', 'interaction', 'T_gate_vs_timer3'):
                c = r[key]
                print(f"  {key:26s} {c['mean_auc']:+7.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']}/{c['tied']} p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f} rule={c['meets_rule']}")


if __name__ == '__main__':
    main()
