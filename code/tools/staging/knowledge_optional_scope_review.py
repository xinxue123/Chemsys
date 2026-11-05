"""Read-only optional-validation scope/rank review on committed saved GPs.

No oracle, API, fit, optimizer, or trajectory replay. Counterfactual rankings
keep each pre-experiment GP, qLogEI array, budget set and tie stream frozen.
"""
from collections import Counter
import importlib.util
import json
from math import log, sqrt
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp
import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('scope_review_audit', ROOT / 'code/tools/audit_knowledge_loop.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def contrast(folder, pool, prefix, commit, fit, arrays, hypothesis, indices):
    """Same saved-state reconstruction and direct covariance check as audit."""
    saved = torch.load(folder / 'cache' / pool['task'] / (commit['GP_fit_key'] + '.pt'),
                       map_location='cpu', weights_only=True)
    audit.native_seed_audit(pool, commit, saved['fit_seed'])
    raw = pool['raw']
    if pool['group'] == 'P0':
        labels = ['P1-L1', 'P2-L1', 'P1-L2', 'P1-L3', 'P1-L4', 'P1-L5', 'P1-L6', 'P1-L7']
        features = np.column_stack((np.asarray(raw['z'])[:, :3].clip(1e-6, 1 - 1e-6),
            [saved['category_map'][labels.index(point['catalyst'])] for point in raw['points']]))
        cats = [3]
    elif pool['group'] == 'P1':
        features = [[saved['mapping'][axis][point[axis]] for axis in pool['axes']] for point in raw['points']]
        cats = list(range(len(pool['axes'])))
    else:
        features, cats = raw['z'], None
    X = torch.tensor(features, dtype=torch.double)
    ordered = sorted(prefix, key=(lambda h: pool['features'][pool['ix'][h['candidate_id']]])
                     if pool['group'] == 'P2' else (lambda h: h['candidate_id']))
    train = X[[pool['ix'][h['candidate_id']] for h in ordered]]
    y = torch.tensor([[h['y']] for h in ordered], dtype=torch.double)
    mean, std = y.mean(), y.std(unbiased=False).clamp_min(1e-12)
    ys = (y - mean) / std
    model = SingleTaskGP(train, ys) if cats is None else MixedSingleTaskGP(train, ys, cat_dims=cats)
    model.load_state_dict(saved['state']); model.eval()
    audit.exact({name: p.detach().cpu().tolist() for name, p in model.named_parameters()}, fit, 'fit receipt')
    anchors = [pool['ix'][hypothesis['plus_id']], pool['ix'][hypothesis['minus_id']]]
    with torch.no_grad():
        K = model.covar_module(train).to_dense() + torch.eye(len(train), dtype=torch.double) * model.likelihood.noise.reshape(())
        cross = model.covar_module(X[indices], train).to_dense()
        anchor_cross = model.covar_module(X[anchors], train).to_dense()
        scale2 = model.outcome_transform.stdvs.reshape(()).square() * std.square()
        direct = (model.covar_module(X[indices], X[anchors]).to_dense()
                  - cross @ torch.linalg.solve(K, anchor_cross.T)) * scale2
        direct_cov = (direct[:, 0] - direct[:, 1]).numpy()
        mean_check = model.mean_module(X[indices]) + cross @ torch.linalg.solve(K, model.train_targets - model.mean_module(train))
        mean_check = ((mean_check * model.outcome_transform.stdvs.reshape(()) + model.outcome_transform.means.reshape(())) * std + mean).numpy()
        np.testing.assert_allclose(mean_check, arrays['mu_y'][indices], atol=1e-9, rtol=1e-7)
        covs = np.full(len(pool['ids']), np.nan)
        variance = None
        for start in range(0, len(indices), 128):
            js = indices[start:start + 128]
            native = model.posterior(torch.cat((X[js], X[anchors]))).mvn.covariance_matrix.numpy() * float(std ** 2)
            n = len(js)
            covs[js] = native[:n, n] - native[:n, n + 1]
            current = max(0., float(native[n, n] + native[n + 1, n + 1] - 2 * native[n, n + 1]))
            if variance is not None:
                audit.close(current, variance, 'contrast variance across chunks', atol=1e-9, rtol=1e-6)
            variance = current
        np.testing.assert_allclose(covs[indices], direct_cov, atol=1e-9, rtol=1e-6)
    return float(arrays['mu_y'][anchors[0]] - arrays['mu_y'][anchors[1]]), variance, covs


def rank(scores, eligible, priority):
    js = np.flatnonzero(eligible)
    maxima = js[scores[js] == scores[js].max()]
    return int(maxima[np.argmin(priority[maxima])])


def mixed_scores(base, offsets, weight, noop):
    if weight == 0:
        return base.copy()
    correction = np.logaddexp(np.log1p(-weight), log(weight) + offsets)
    correction[np.abs(offsets) <= noop] = 0.
    return base + correction


def scoped_winner_exists(base, offsets, eligible, priority, scoped, bo, cap, noop):
    """Exact affine-EI envelopes locate intervals; literal score code verifies witnesses."""
    js = np.flatnonzero(eligible)
    A = np.exp(base[js] - max(base[js]))
    B = A * np.expm1(offsets[js])
    witnesses = []
    for j in js:
        if j == bo or not scoped[j]:
            continue
        local = int(np.flatnonzero(js == j)[0])
        slope = B[local] - B
        rhs = A - A[local]
        parallel = slope == 0
        if np.any(parallel & ((rhs > 0) | ((rhs == 0) & (priority[js] < priority[j])))):
            continue
        lower, upper = 0., cap
        if np.any(slope > 0):
            lower = max(lower, float(np.max(rhs[slope > 0] / slope[slope > 0])))
        if np.any(slope < 0):
            upper = min(upper, float(np.min(rhs[slope < 0] / slope[slope < 0])))
        if upper < lower or upper <= 0 or lower > cap:
            continue
        probes = [(max(0., lower) + upper) / 2, upper, max(0., lower), np.nextafter(max(0., lower), upper)]
        for weight in probes:
            if 0 < weight <= cap and rank(mixed_scores(base, offsets, weight, noop), eligible, priority) == j:
                witnesses.append(dict(index=int(j), weight=float(weight), weight_interval=[max(0., lower), upper]))
                break
    return witnesses


def review_round(folder, protocol, pool, state, run, t):
    commit = audit.read(run / 'rounds' / f'{t:02d}_commit.json')
    decision = state['decisions'][t - 1]
    audit.require(commit['mode'] == 'optional_verification' and audit.digest(commit) == decision['commit_sha256'], 'optional commitment custody')
    prefix = state['history'][:t + 3]
    audit.require(audit.digest(prefix) == commit['history_sha256'], 'committed prefix custody')
    target = commit['verification']
    h = next(h for h in commit['hypotheses_at_commit'] if audit.key(h) == audit.key(target))
    with np.load(run / 'rounds' / f'{t:02d}_GP.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    fit = audit.read(run / 'rounds' / f'{t:02d}_fit.json')
    base, eligible, priority = arrays['base_logei'], arrays['eligible'], arrays['tie_priority']
    bo, selected = pool['ix'][commit['BO_reference_id']], pool['ix'][commit['selected_id']]
    audit.require(rank(base, eligible, priority) == bo, 'base BO identity')
    js = np.flatnonzero(eligible)
    md, vd, covs = contrast(folder, pool, prefix, commit, fit, arrays, h, js)
    pars = protocol['parameters']
    tau, cap, noop = pars['preference_noise_sd'], pars['maximum_deployment_weight'], pars['numerical_noop_atol']
    vz = vd + tau ** 2
    scoped = np.array([audit.in_scope(pool, h, j) for j in range(len(pool['ids']))])
    scores = base.copy()
    nodes, weights = np.polynomial.hermite.hermgauss(pars['GH_nodes'])
    logw, best = np.log(weights / sqrt(np.pi)), max(row['y'] for row in prefix)
    for j in js[scoped[js]]:
        vx = max(float(arrays['sd_y'][j]) ** 2, 1e-18)
        draws = arrays['mu_y'][j] + sqrt(2 * vx) * nodes
        improvement = np.maximum(draws - best, 0.)
        positive = improvement > 0
        if not positive.any():
            continue
        common = logw[positive] + np.log(improvement[positive])
        rem = max(vz - covs[j] ** 2 / vx, 1e-15 * vz)
        conditional = (md + covs[j] / vx * (draws[positive] - arrays['mu_y'][j])) / sqrt(rem)
        offset = logsumexp(common + log_ndtr(conditional)) - logsumexp(common) - log_ndtr(md / sqrt(vz))
        if abs(offset) <= noop:
            offset = 0.
        scores[j] += float(np.clip(offset, -pars['maximum_logEI_shift'], pars['maximum_logEI_shift']))
    offsets = scores - base
    capped = mixed_scores(base, offsets, cap, noop)
    winner = rank(capped, eligible, priority)
    row = next(r for r in commit['coverage'] if audit.key(r) == audit.key(h))
    audit.require(row['can_change_rank_at_cap'] == (winner != bo), 'cap ranking disagrees with committed coverage')
    def info(j):
        vy = float(arrays['sd_predictive_y'][j]) ** 2
        rem = max(vz - covs[j] ** 2 / max(vy, 1e-18), 1e-15 * vz)
        return .5 * log(vz / rem)
    def point(j):
        amount = info(j)
        return dict(id=pool['ids'][j], in_scope=bool(scoped[j]), nonBO=bool(j != bo),
                    information_nats=amount, nonzero_information=bool(amount > 0.),
                    scoreable_by_scope_and_information=bool(scoped[j] and amount >= pars['minimum_incremental_information']),
                    base_logEI=float(base[j]), capped_logEI=float(capped[j]))
    remaining = eligible.copy(); remaining[selected] = False
    after = rank(capped, remaining, priority)
    witnesses = scoped_winner_exists(base, offsets, remaining, priority, scoped, bo, cap, noop)
    for witness in witnesses:
        j = witness.pop('index')
        witness.update(point(j))
    return dict(variant=protocol['variant'], trial_seed=protocol['seed'], task=pool['task'], query=t,
        hypothesis_id=h['id'], version=h['version'], created_round=h['created_round'], maximum_weight=cap,
        committed_weight=audit.weight(h), eligible_candidates=int(eligible.sum()), eligible_scope_candidates=int((eligible & scoped).sum()),
        BO_reference=point(bo), validation_point=point(selected), at_cap_winner=point(winner),
        remaining_scope_after_validation=int((remaining & scoped).sum()), after_removal_at_cap_winner=point(after),
        any_remaining_scope_nonBO_winner_for_admissible_weight=bool(witnesses),
        any_remaining_scope_nonBO_scoreable_winner=any(w['scoreable_by_scope_and_information'] for w in witnesses),
        remaining_scope_winner_witnesses=witnesses,
        commit_sha256=decision['commit_sha256'])


def main():
    torch.set_num_threads(1)
    rows = []
    for variant in ('v3', 'v3_seed1_confirm'):
        folder = ROOT / 'experiments/runs/knowledge_loop_20261101' / variant
        p = audit.read(folder / 'protocol.json')
        for item in p['inputs']:
            pool = dict(audit.public_pool(ROOT, item), task=item['task'])
            run = folder / 'runs' / f"{item['task']}_full_agent_initial_terra"
            state = audit.read(run / 'state.json')
            for d in state['decisions']:
                if d['mode'] == 'optional_verification':
                    row = review_round(folder, p, pool, state, run, d['query'])
                    rows.append(row)
                    print(json.dumps(dict(variant=variant, task=pool['task'], query=d['query'],
                        cap_winner_in_scope=row['at_cap_winner']['in_scope'],
                        remaining_scope_can_win=row['any_remaining_scope_nonBO_winner_for_admissible_weight'])), flush=True)
    counts = []
    for variant in ('v3', 'v3_seed1_confirm'):
        part = [r for r in rows if r['variant'] == variant]
        counts.append(dict(variant=variant, optional_rounds=len(part),
            cap_winners_in_scope=sum(r['at_cap_winner']['in_scope'] for r in part),
            cap_winners_scoreable=sum(r['at_cap_winner']['scoreable_by_scope_and_information'] for r in part),
            remaining_scope_can_win_at_any_allowed_weight=sum(r['any_remaining_scope_nonBO_winner_for_admissible_weight'] for r in part),
            remaining_scope_can_win_and_be_scored=sum(r['any_remaining_scope_nonBO_scoreable_winner'] for r in part)))
    report = dict(assessment='READ_ONLY_DEVELOPMENT_DIAGNOSTIC_NOT_NEW_EXPERIMENT',
        script_sha256=audit.filehash(__file__), GP_fits=0, LLM_calls=0, oracle_values_read=False,
        definitions='Remove only the proposed validation point from the existing budget set; retain the same pre-query GP and tie priorities. Test the target hypothesis alone at the maximum allowed weight and via affine-EI weight-interval witnesses.',
        limitation='This examines pre-query deployability, not post-feedback GP refitting or chemical truth. Nonzero covariance outside scope does not authorize deployment or evidence credit.',
        counts=counts, rows=rows)
    out = ROOT / 'experiments/reports/knowledge_loop_20261101/optional_scope_closed_loop_review.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(report=str(out), counts=counts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
