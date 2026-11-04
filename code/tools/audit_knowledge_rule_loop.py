"""Independent read-only V4 rule/probe audit; never fit, reveal, or call LLM.

Production rule math/policy/engine are deliberately not imported. The legacy
auditor supplies only input decoding, custody, seed and source-order helpers.
Full candidate reranking is reconstructed independently; selected/BO expert EI
also receives a second, adaptive-quadrature check over F, not GH over Z.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from math import exp, log, pi, sqrt, tanh
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, logsumexp, ndtri_exp

import audit_knowledge_loop as common

ROOT = common.ROOT
read, digest, filehash = common.read, common.digest, common.filehash
require, close, exact = common.require, common.close, common.exact
GH_X, GH_W = np.polynomial.hermite.hermgauss(64)
GH_X, GH_LW = GH_X*sqrt(2.), np.log(GH_W/sqrt(pi))
ATOL = 1e-12


def hkey(h):
    return h['id'], h['version']


def contract(h):
    return {k: deepcopy(h[k]) for k in ('focus', 'scope', 'categorical_preference')}


def scope_mask(pool, h, js):
    js = np.asarray(js, dtype=int)
    rows = np.asarray(pool['features'], dtype=object)[js]
    keep = np.ones(len(js), bool)
    for k, axis in enumerate(pool['axes']):
        s = h['scope']['axes'][axis]
        require(s['kind'] == pool['kinds'][k], 'rule category/continuous kind changed')
        if s['kind'] == 'continuous':
            keep &= (np.asarray(rows[:, k], float) >= s['bounds'][0]-ATOL)
            keep &= (np.asarray(rows[:, k], float) <= s['bounds'][1]+ATOL)
        else:
            keep &= np.isin(rows[:, k], s['allowed'])
    return keep


def compatible(pool, h, candidates, reference):
    answer = np.ones(len(candidates), bool)
    for k, (axis, kind) in enumerate(zip(pool['axes'], pool['kinds'])):
        if axis in h['focus']:
            if kind == 'continuous':
                sign = 1 if h['focus'][axis]['direction'] == 'increase' else -1
                answer &= sign*(np.asarray(candidates[:, k], float)-float(reference[k])) > 0
            else:
                pref = h['categorical_preference'][axis]
                answer &= (reference[k] == pref['minus']) & (candidates[:, k] == pref['plus'])
        elif kind == 'continuous':
            answer &= np.abs(np.asarray(candidates[:, k], float)-float(reference[k])) <= .25+ATOL
        else:
            answer &= candidates[:, k] == reference[k]
    return answer


def probes(pool, h, prefix, eligible):
    require(h['semantic_key'] == digest(contract(h)), 'rule semantic SHA')
    if h['status'] == 'retracted':
        return {}
    known = {x['candidate_id']: x for x in prefix}
    js = np.flatnonzero(eligible)
    js = js[scope_mask(pool, h, js)]
    require(not any(pool['ids'][j] in known for j in js), 'executable plus already observed')
    if not len(js):
        return {}
    features = np.asarray(pool['features'], dtype=object)
    remaining = np.ones(len(js), bool)
    result = {}
    for row in sorted(prefix, key=lambda r: (-r['y'], pool['features'][pool['ix'][r['candidate_id']]])):
        ix = pool['ix'][row['candidate_id']]
        if not scope_mask(pool, h, [ix])[0]:
            continue
        hit = remaining & compatible(pool, h, features[js], features[ix])
        for j in js[hit]:
            result[int(j)] = dict(reference_id=row['candidate_id'], reference_y=float(row['y']))
        remaining[hit] = False
    return result


def evidence(h):
    l = h['ledger']
    own, inherited = l['events'], [e for e in h.get('inherited_negative_evidence', []) if e['applicable']]
    identity = lambda e: (e['query_round'], e['point_id'], e['reference_id'])
    identities = list(dict.fromkeys(identity(e) for e in own+inherited))
    score = local_score = 0.
    s = f = ls = lf = 0
    for key in identities:
        a, b = [e for e in own if identity(e) == key], [e for e in inherited if identity(e) == key]
        own_scores = [e['delta_log_score'] for e in a]
        local_score += min(own_scores, default=0.)
        score += min(own_scores+[e['delta_log_score'] for e in b if e['delta_log_score'] < 0], default=0.)
        own_failure, failure = any(not e['fact_success'] for e in a), any(not e['fact_success'] for e in a+b)
        lf += int(own_failure)
        ls += int(bool(a) and not own_failure)
        f += int(failure)
        s += int(bool(a) and not failure)
    cap = max(0., (s-f)/(s+f+2))
    formula = min(.5, max(0., tanh(score/2)), cap)
    trusted = min(formula, max(0., float(l['trusted_weight'])))
    return dict(effective_prediction_score=score, successes=s, failures=f,
        trusted_weight=trusted, formula_weight=formula, reliability_cap=cap,
        local_prediction_score=local_score, local_successes=ls,
        local_failures=lf, inherited_failure_count=f-lf,
        inherited_prediction_penalty=score-local_score)


def weight(h):
    return evidence(h)['trusted_weight'] if h['status'] == 'active' else 0.


def normal_log_ei(mu, variance, best):
    """Independent analytic Gaussian EI with asymptotic negative-tail series."""
    mu, variance, best = np.broadcast_arrays(mu, variance, best)
    out = np.full(mu.shape, -np.inf)
    fixed = variance == 0
    pos = fixed & (mu > best)
    out[pos] = np.log(mu[pos]-best[pos])
    random = ~fixed
    sd = np.sqrt(variance[random])
    z = (mu[random]-best[random])/sd
    lp = -.5*z*z-.5*log(2*pi)
    val = np.empty(z.shape)
    pos, far = z >= 0, z < -10
    middle = ~(pos | far)
    with np.errstate(divide='ignore'):
        val[pos] = np.logaddexp(lp[pos], np.log(z[pos])+log_ndtr(z[pos]))
    val[middle] = lp[middle]+np.log(-np.expm1(np.log(-z[middle])+log_ndtr(z[middle])-lp[middle]))
    if np.any(far):
        zz = z[far]*z[far]
        total, term = np.ones_like(zz), np.ones_like(zz)
        for j in range(1, 13):
            term = -term*(2*j+1)/zz
            total += term
        val[far] = lp[far]-2*np.log(-z[far])+np.log(total)
    out[random] = np.log(sd)+val
    return out


def binary_entropy(z):
    a, b = log_ndtr(z), log_ndtr(-z)
    return -np.exp(a)*a-np.exp(b)*b


def math_fields(mu, vf, vy, ref, best, tau=.06):
    mu, vf, vy, ref = np.broadcast_arrays(mu, vf, vy, ref)
    require(np.all(vf >= 0) and np.all(vy >= vf), 'invalid latent/observed variance')
    vz = vy+tau*tau
    alpha = (mu-ref)/np.sqrt(vz)
    lp = log_ndtr(alpha)
    z = -ndtri_exp(lp[..., None]+log_ndtr(-GH_X))
    cm = mu[..., None]+(vf/np.sqrt(vz))[..., None]*z
    cv = np.maximum(0., vf*(1-vf/vz))
    log_ei = logsumexp(normal_log_ei(cm, cv[..., None], best)+GH_LW, axis=-1)
    base = normal_log_ei(mu, vf, best)
    lr = np.zeros(mu.shape)
    positive = vf > 0
    lr[positive] = log_ei[positive]-base[positive]
    lr = np.clip(lr, 0, -lp)
    noop = (~positive) | (lr <= ATOL)
    lr = np.where(noop, 0., lr)
    lt = np.zeros(mu.shape)
    capped = lr > log(4)
    lt[capped] = log(3)-lr[capped]-np.log1p(-np.exp(-lr[capped]))
    lt[noop] = -np.inf
    yy = mu[..., None]+np.sqrt(vy)[..., None]*GH_X
    eh = np.sum(np.exp(GH_LW)*binary_entropy((yy-ref[..., None])/tau), axis=-1)
    narrow = vy > tau*tau
    if np.any(narrow):
        sd = np.sqrt(vy[narrow])
        standardized = ((ref[narrow]-mu[narrow])[..., None]+tau*GH_X)/sd[..., None]
        terms = GH_LW+np.log(tau/sd)[..., None]-.5*standardized**2+.5*GH_X**2+np.log(binary_entropy(GH_X))
        eh = np.array(eh, copy=True)
        eh[narrow] = np.exp(logsumexp(terms, axis=-1))
    info = np.clip(binary_entropy(alpha)-eh, 0, log(2))
    return dict(offset=np.minimum(lr, log(4)), log_tempering=lt, tempering=np.exp(lt),
                pure_log_ei_ratio=lr, sign_information=np.where(vy == 0, 0., info),
                event_log_probability=lp, base_normal_log_ei=base)


def score(pred, y):
    lt = pred['log_tempering']
    if lt is None:
        return 0.
    # The denominator is independently recomputed/checked by shadow_feedback;
    # retain its exact committed representation when checking event hashes.
    pure = float(log_ndtr((y-pred['reference_y'])/pred['preference_noise_sd'])
                 -pred['event_log_probability'])
    if pred['tempering'] == 1:
        return pure
    return float(np.logaddexp(np.log1p(-pred['tempering']), lt+pure))


def adaptive_ei_check(mu, vf, vy, reference, best, expected_log_ratio, tau=.06):
    """Independent adaptive integral over F, stably scaled by base EI.

    The expectation is bounded by 1/P(event). For vanishing latent variance
    the declared zero-effect policy is checked without quadrature.
    """
    if vf == 0:
        close(expected_log_ratio, 0., 'quad zero latent effect', atol=0, rtol=0)
        return 0.
    sd = sqrt(vf)
    cutoff = (best-mu)/sd
    lb = float(normal_log_ei(np.asarray(mu), np.asarray(vf), np.asarray(best)))
    lp = float(log_ndtr((mu-reference)/sqrt(vy+tau*tau)))
    # Integrate distance from the improvement threshold. In the far tail,
    # rescale z-cutoff by its typical exponential-tail width 1/cutoff.
    scale = 1./max(1., cutoff)
    def integrand(u, conditional):
        dz = scale*u
        z = cutoff+dz
        if u == 0:
            return 0.
        value = log(sd)+log(dz)-.5*z*z-.5*log(2*pi)-lb+log(scale)
        if conditional:
            value += float(log_ndtr((best+sd*dz-reference)/sqrt(vy-vf+tau*tau)))-lp
        # Scale the conditional integral by its predicted log ratio to
        # keep extreme-tail ratios representable, then undo it in log.
        return exp(value-(expected_log_ratio if conditional else 0.))
    def integrate(conditional):
        # Infinite-interval quad can miss a narrow peak thousands of
        # standard deviations above a very low improvement threshold.
        # Center its finite window on the known Gaussian/conditioned mean;
        # conditioning a positive Gaussian contrast cannot increase its
        # variance. Sixteen original SDs leave negligible omitted tails.
        center_z = 0.
        if conditional:
            aa = (mu-reference)/sqrt(vy+tau*tau)
            mills = exp(-.5*aa*aa-.5*log(2*pi)-lp)
            center_z = sd/sqrt(vy+tau*tau)*mills
        center = max(0., (center_z-cutoff)/scale)
        lo, hi = (max(0., center-16/scale), center+16/scale) if center > 20 else (0., np.inf)
        return quad(lambda u: integrand(u, conditional), lo, hi, epsabs=1e-9, epsrel=2e-9, limit=200)[0]
    baseline, corrected = integrate(False), integrate(True)
    close(baseline, 1., 'independent normal EI integral', atol=2e-7, rtol=2e-7)
    require(corrected > 0 and np.isfinite(corrected), 'quad conditional EI failed')
    error = abs(log(corrected))
    require(error <= 2e-5, f'GH64/independent F-integral log-ratio error {error}')
    return error


def select(scores, eligible, ties):
    js = np.flatnonzero(eligible & (scores == scores[eligible].max()))
    return int(js[np.argmin(ties[js])])


def calculations(pool, hyps, prefix, arrays):
    eligible, base, ties = arrays['eligible'], arrays['base_logei'], arrays['tie_priority']
    bo = select(base, eligible, ties)
    result, rows = {}, []
    for h in hyps:
        if h['status'] == 'retracted':
            continue
        ps = probes(pool, h, prefix, eligible)
        js = np.array(sorted(ps), dtype=int)
        row = dict(id=h['id'], version=h['version'], weight=weight(h),
            eligible_inside_scope=int(scope_mask(pool, h, np.flatnonzero(eligible)).sum()),
            executable_probes=len(js), can_change_rank_at_cap=False,
            eligibility_state='no_known_comparable_reference' if not len(js) else 'executable_rule')
        if len(js):
            vals = math_fields(arrays['mu_y'][js], arrays['sd_y'][js]**2,
                arrays['sd_predictive_y'][js]**2, np.array([ps[int(j)]['reference_y'] for j in js]),
                max(x['y'] for x in prefix))
            offsets = np.zeros_like(base)
            offsets[js] = vals['offset']
            correction = np.logaddexp(log(.5), log(.5)+offsets)
            correction[offsets <= ATOL] = 0.
            row['can_change_rank_at_cap'] = select(base+correction, eligible, ties) != bo
            row['positive_effect_probes'] = int(np.count_nonzero(vals['offset'] > ATOL))
            row['eligibility_state'] = 'executable_rank_effect' if row['can_change_rank_at_cap'] else 'executable_without_rank_effect'
            result[hkey(h)] = dict(probes=ps, indices=js, positions={int(j): k for k, j in enumerate(js)},
                                  values=vals, offsets=offsets)
        rows.append(row)
    return rows, result


def refresh(pool, hyps, prefix, arrays, t):
    hs = deepcopy(hyps)
    rows, _ = calculations(pool, hs, prefix, arrays)
    by_key = {hkey(r): r for r in rows}
    events = []
    for h in hs:
        if h['status'] == 'retracted':
            continue
        row = by_key[hkey(h)]
        status = 'active' if row['executable_probes'] else 'dormant'
        if status != h['status']:
            events.append(dict(round=t, id=h['id'], version=h['version'], from_status=h['status'],
                               to_status=status, reason=row['eligibility_state']))
        h['status'] = status
        h['coverage'] = dict(round=t, executable_probes=row['executable_probes'], eligible_inside_scope=row['eligible_inside_scope'])
        h['ledger']['weight'] = weight(h)
        row['weight'] = weight(h)
    return hs, events, rows


def selection_audit(pool, prefix, c, arrays, arm, verification_used, parameters):
    base, eligible, ties = arrays['base_logei'], arrays['eligible'], arrays['tie_priority']
    bo = select(base, eligible, ties)
    rows, calc = calculations(pool, c['hypotheses_at_commit'], prefix, arrays)
    exact(c['coverage'], rows, 'independent complete rule coverage')
    deployed = []
    if arm in ('full', 'deployment'):
        raw = [r for r in rows if r['weight'] > 0 and hkey(r) in calc]
        total = sum(r['weight'] for r in raw)
        factor = min(1., .5/total) if total else 1.
        deployed = [dict(id=r['id'], version=r['version'], weight=r['weight'],
                         effective_mixture_weight=r['weight']*factor) for r in raw]
    exact(c['deployed_versions'], deployed, 'actual mixture weights/cap')
    combined = base.copy()
    if deployed:
        ws = [x['effective_mixture_weight'] for x in deployed]
        offsets = np.stack([calc[hkey(x)]['offsets'] for x in deployed])
        correction = logsumexp(np.vstack([np.full_like(base, log(1-sum(ws)))] +
            [log(w)+offset for w, offset in zip(ws, offsets)]), axis=0)
        correction[np.all(offsets <= ATOL, axis=0)] = 0.
        combined += correction
    selected = select(combined, eligible, ties)
    mode = 'knowledge_guided' if selected != bo else 'base_bo'
    ver = None
    if arm == 'full' and selected == bo and c['query'] <= parameters['last_verification_round'] and verification_used < parameters['maximum_optional_verifications']:
        candidates = []
        for row in rows:
            k = hkey(row)
            if row['weight'] > 0 or not row['can_change_rank_at_cap'] or k not in calc:
                continue
            h = next(h for h in c['hypotheses_at_commit'] if hkey(h) == k)
            e = evidence(h)
            if e['effective_prediction_score'] < 0 or e['failures'] > e['successes']:
                continue
            info = calc[k]['values']['sign_information']
            pos = calc[k]['positions'].get(bo)
            ref_info = float(info[pos]) if pos is not None else 0.
            for pos, j in enumerate(calc[k]['indices']):
                gap = float(base[bo]-base[j])
                if j != bo and gap <= min(c['epsilon'], parameters['verification_maximum_gap']) and info[pos]-ref_info >= parameters['minimum_incremental_information'] and calc[k]['values']['offset'][pos] > ATOL:
                    candidates.append((float(info[pos]-ref_info), -gap, -ties[j], int(j), k, float(info[pos])))
        if candidates:
            ver = max(candidates, key=lambda x: x[:3])
            selected, mode = ver[3], 'optional_verification'
    require(pool['ids'][selected] == c['selected_id'] and mode == c['mode'], 'independent full-candidate/optional argmax differs')
    close(c['combined_logei'], combined[selected], 'independent selected mixture score', atol=1e-10, rtol=1e-10)
    if mode == 'knowledge_guided':
        require(combined[selected] > base[selected]+ATOL, 'unsupported outside point selected by suppression')
    if ver:
        r = calc[ver[4]]['probes'][selected]
        expected = dict(id=ver[4][0], version=ver[4][1], incremental_information=ver[0], information=ver[5],
            reference_id=r['reference_id'], reference_y=r['reference_y'],
            remaining_executable_probes=len(calc[ver[4]]['indices'])-1,
            information_kind='binary_soft_sign', minimum_unqueried_endpoints=1)
        exact(c['verification'], expected, 'optional verification receipt/global information argmax')
    else:
        require(c['verification'] is None, 'unexpected verification receipt')
    errors = []
    for data in calc.values():
        for j in {selected, bo}:
            pos = data['positions'].get(j)
            if pos is None:
                continue
            errors.append(adaptive_ei_check(float(arrays['mu_y'][j]), float(arrays['sd_y'][j]**2),
                float(arrays['sd_predictive_y'][j]**2), data['probes'][j]['reference_y'], max(x['y'] for x in prefix),
                float(data['values']['pure_log_ei_ratio'][pos])))
    return calc, max(errors, default=0.)


def saved_gp_audit(folder, pool, prefix, c, arrays, fit):
    """Load cached weights only; condition algebraically and replay MC512."""
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from botorch.models.model import Model
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.distributions import MultivariateNormal
    torch.set_num_threads(1)
    saved = torch.load(folder/'cache'/pool['task']/(c['GP_fit_key']+'.pt'), map_location='cpu', weights_only=True)
    require(saved['key'] == c['GP_fit_key'], 'cache fit key')
    common.native_seed_audit(pool, c, saved['fit_seed'])
    raw = pool['raw']
    if pool['group'] == 'P0':
        labels = ['P1-L1', 'P2-L1', 'P1-L2', 'P1-L3', 'P1-L4', 'P1-L5', 'P1-L6', 'P1-L7']
        features = np.column_stack((np.asarray(raw['z'])[:, :3].clip(1e-6, 1-1e-6),
                        [saved['category_map'][labels.index(p['catalyst'])] for p in raw['points']]))
        cats = [3]
    elif pool['group'] == 'P1':
        features = np.asarray([[saved['mapping'][a][p[a]] for a in pool['axes']] for p in raw['points']], float)
        cats = list(range(len(pool['axes'])))
    else:
        features, cats = np.asarray(raw['z'], float), None
    X = torch.tensor(features, dtype=torch.double)
    ordered = sorted(prefix, key=(lambda h: pool['features'][pool['ix'][h['candidate_id']]])
                     if pool['group'] == 'P2' else (lambda h: h['candidate_id']))
    train = X[[pool['ix'][h['candidate_id']] for h in ordered]]
    y = torch.tensor([[h['y']] for h in ordered], dtype=torch.double)
    outer_mean, outer_std = y.mean(), y.std(unbiased=False).clamp_min(1e-12)
    ys = (y-outer_mean)/outer_std
    if 'train_candidate_ids' in saved:
        require(saved['train_candidate_ids'] == [h['candidate_id'] for h in ordered], 'GP prefix ID order')
        require(torch.equal(torch.as_tensor(saved['train_X'], dtype=torch.double), train), 'GP prefix X')
    else:
        exact(saved['history'], [{'candidate_id': h['candidate_id'], 'y': h['y']} for h in ordered], 'GP prefix outcomes/order')
    model = MixedSingleTaskGP(train, ys, cat_dims=cats) if cats is not None else SingleTaskGP(train, ys)
    model.load_state_dict(saved['state'])
    model.eval()
    exact({name: p.detach().cpu().tolist() for name, p in model.named_parameters()}, fit, 'saved fit receipt')
    with torch.no_grad():
        K = model.covar_module(train).to_dense()+torch.eye(len(train), dtype=torch.double)*model.likelihood.noise.reshape(())
        cross = model.covar_module(X, train).to_dense()
        solve = torch.linalg.solve(K, cross.T)
        scale2 = model.outcome_transform.stdvs.reshape(()).square()*outer_std.square()
        mu = model.mean_module(X)+cross@torch.linalg.solve(K, model.train_targets-model.mean_module(train))
        mu = (mu*model.outcome_transform.stdvs.reshape(())+model.outcome_transform.means.reshape(()))*outer_std+outer_mean
        diagonal = torch.cat([model.covar_module(part, diag=True) for part in X.split(256)])
        var = (diagonal-(cross*solve.T).sum(-1))*scale2
        obsvar = var+model.likelihood.noise.reshape(())*scale2
        np.testing.assert_allclose(arrays['mu_y'], mu.numpy(), atol=1e-9, rtol=1e-7, err_msg='dense GP mean')
        np.testing.assert_allclose(arrays['sd_y']**2, var.numpy(), atol=1e-9, rtol=1e-6, err_msg='dense GP latent variance')
        np.testing.assert_allclose(arrays['sd_predictive_y']**2, obsvar.numpy(), atol=1e-9, rtol=1e-6, err_msg='dense GP observation variance')
        # Reproduce native marginal arithmetic before sampler evaluation;
        # direct dense conditioning above is an independent second check.
        mus, variances = [], []
        for part in X.split(256):
            p = model.posterior(part.unsqueeze(1))
            mus.append(p.mean.reshape(-1))
            variances.append(p.variance.reshape(-1).clamp_min(1e-18))
        native_mu, native_var = torch.cat(mus), torch.cat(variances)

    class MarginalModel(Model):
        @property
        def num_outputs(self): return 1
        @property
        def batch_shape(self): return torch.Size()
        def posterior(self, X, **unused):
            ix = X[..., 0].long()
            return GPyTorchPosterior(MultivariateNormal(native_mu[ix], torch.diag_embed(native_var[ix]+1e-6)))
    acq = qLogExpectedImprovement(MarginalModel(), best_f=float(ys.max()),
            sampler=SobolQMCNormalSampler(torch.Size([512]), seed=c['GP_acquisition_meta']['mc_seed']))
    lookup = torch.arange(len(X), dtype=torch.double).reshape(-1, 1, 1)
    with torch.no_grad():
        mc = torch.cat([acq(part).reshape(-1) for part in lookup.split(256)]).numpy()
    np.testing.assert_allclose(arrays['base_logei'], mc, atol=1e-9, rtol=1e-8, err_msg='native MC512 qLogEI from cached GP')


def applicable_negative(pool, new, row):
    def relation(h):
        return dict(directions={a: v['direction'] for a, v in h['focus'].items()},
                    categorical_preference=h['categorical_preference'])
    if digest(relation(new)) != digest(relation(row['source_rule'])):
        return False
    js = [pool['ix'][row['point_id']], pool['ix'][row['reference_id']]]
    if not scope_mask(pool, new, js).all():
        return False
    features = np.asarray(pool['features'], dtype=object)
    return bool(compatible(pool, new, features[js[:1]], features[js[1]])[0])


def rule_geometry_audit(pool, h):
    focus = h['focus']
    require(1 <= len(focus) <= 2 and set(focus) <= set(pool['axes']), 'invalid focus axes')
    require(h['scope']['background_radius'] == .25, 'background radius drift')
    a, b = h['plus_features'], h['minus_features']
    expected_scope, categorical = {}, {}
    for k, (axis, kind) in enumerate(zip(pool['axes'], pool['kinds'])):
        if kind == 'continuous':
            require(0 <= a[k] <= 1 and 0 <= b[k] <= 1, 'non-normalized continuous anchor')
            if axis in focus:
                spec = focus[axis]
                require(set(spec) == {'direction', 'bounds'} and spec['direction'] in ('increase', 'decrease'), 'continuous focus contract')
                sign = 1 if spec['direction'] == 'increase' else -1
                lo, hi = spec['bounds']
                require(0 <= lo < hi <= 1 and lo <= min(a[k], b[k]) <= max(a[k], b[k]) <= hi
                        and sign*(a[k]-b[k]) > 0, 'continuous focus direction/anchor bounds')
            else:
                require(abs(a[k]-b[k]) <= .25+ATOL, 'nonfocus continuous confounding distance')
                lo, hi = max(0., a[k]-.25, b[k]-.25), min(1., a[k]+.25, b[k]+.25)
            expected_scope[axis] = dict(kind=kind, bounds=[float(lo), float(hi)])
        else:
            if axis in focus:
                spec = focus[axis]
                require(set(spec) == {'direction', 'allowed'} and spec['direction'] == 'prefer_plus', 'categorical focus schema')
                allowed = spec['allowed']
                real = {r[k] for r in pool['features']}
                require(a[k] != b[k] and a[k] in allowed and b[k] in allowed
                        and len(set(allowed)) == len(allowed) and set(allowed) <= real, 'categorical preference real labels')
                require(allowed == sorted(allowed, key=digest), 'categorical canonical set order')
                categorical[axis] = dict(plus=a[k], minus=b[k])
            else:
                require(a[k] == b[k], 'nonfocus categorical confounding')
                allowed = [a[k]]
            expected_scope[axis] = dict(kind=kind, allowed=allowed)
    exact(h['scope']['axes'], expected_scope, 'independently reconstructed rule scope')
    exact(h['categorical_preference'], categorical, 'categorical preference orientation')


def inherited_records(pool, hyps, target):
    records = {}
    for source in hyps:
        if hkey(source) == hkey(target): continue
        for row in source['inherited_negative_evidence']:
            records[row['event_id']] = deepcopy(row)
        for event in source['ledger']['events']:
            if event['delta_log_score'] < 0 or not event['fact_success']:
                records[event['event_id']] = dict(event_id=event['event_id'], source_id=source['id'],
                    source_version=source['version'], source_rule=contract(source),
                    **{k: deepcopy(event[k]) for k in ('point_id', 'reference_id', 'reference_y', 'observed_y',
                        'query_round', 'delta_log_score', 'fact_success', 'prediction')})
    for row in records.values():
        row['applicable'] = applicable_negative(pool, target, row)
        row['interpretation'] = 'conservative inherited constraint; not a new likelihood or positive fact'
    return sorted(records.values(), key=lambda r: r['event_id'])


def validate_hypotheses(pool, hyps, previous, prefix, t, menu_ids):
    mapping = {hkey(h): h for h in hyps}
    require(len(mapping) == len(hyps), 'duplicate rule/version')
    old = {hkey(h): h for h in previous}
    require(old.keys() <= mapping.keys(), 'historical rule version disappeared')
    live = [h for h in hyps if h['status'] != 'retracted']
    require(len(live) <= 4 and len({h['id'] for h in live}) == len(live), 'live-rule or version count')
    for h in hyps:
        require(h['semantic_key'] == digest(contract(h)), 'rule semantic fingerprint')
        require(h['tau_y'] == .06 and set(h['scope']['axes']) == set(pool['axes']), 'rule geometry/noise schema')
        for side in ('plus', 'minus'):
            j = pool['ix'][h[side+'_id']]
            exact(h[side+'_features'], list(pool['features'][j]), 'origin anchor real physical features')
            require(scope_mask(pool, h, [j])[0], 'origin example outside rule scope')
        rule_geometry_audit(pool, h)
        if hkey(h) in old:
            prior = old[hkey(h)]
            exact(contract(h), contract(prior), 'same-version semantic rewrite')
            for field in ('events', 'seen', 'prediction_score', 'score', 'successes', 'failures', 'trusted_weight'):
                exact(h['ledger'][field], prior['ledger'][field], 'prequery credit rewrite: '+field)
            exact(h['inherited_negative_evidence'], prior['inherited_negative_evidence'], 'inherited negatives rewritten')
            require(prior['status'] != 'retracted' or h['status'] == 'retracted', 'old version silently reactivated')
        else:
            require(h['created_round'] == t, 'new rule creation round')
            l = h['ledger']
            require(not l['events'] and not l['seen'] and l['prediction_score'] == l['score'] == l['successes'] == l['failures'] == l['trusted_weight'] == 0, 'new version has retrospective positive credit')
            require(h['excluded_observed_ids'] == sorted(x['candidate_id'] for x in prefix), 'new rule excludes wrong old outcomes')
            require(h['minus_id'] in h['excluded_observed_ids'] and h['plus_id'] not in h['excluded_observed_ids'], 'origin new-plus/known-reference roles')
            require(h['plus_id'] in menu_ids, 'new rule plus not proposed from visible input menu')
            exact(h['inherited_negative_evidence'], inherited_records(pool, hyps, h), 'negative prediction/fact inheritance completeness')
            if h['parent'] is not None:
                parent = mapping[hkey(h['parent'])]
                require(h['id'] == parent['id'] and h['version'] == parent['version']+1
                        and parent['status'] == 'retracted' and parent['retired_round'] == t, 'revision parent/version retirement')
        seen = set()
        for row in h['inherited_negative_evidence']:
            require(row['event_id'] not in seen, 'duplicate inherited event')
            seen.add(row['event_id'])
            source = mapping[(row['source_id'], row['source_version'])]
            ev = next(e for e in source['ledger']['events'] if e['event_id'] == row['event_id'])
            require(ev['delta_log_score'] < 0 or not ev['fact_success'], 'inherited positive-only evidence')
            for field in ('point_id', 'reference_id', 'reference_y', 'query_round', 'observed_y', 'delta_log_score', 'fact_success', 'prediction'):
                exact(row[field], ev[field], 'inherited event changed: '+field)
            exact(row['source_rule'], contract(source), 'inherited source contract')
            require(row['applicable'] == applicable_negative(pool, h, row), 'inherited scope/relation applicability')
        l = h['ledger']
        close(l['score'], l['prediction_score'], 'ledger score alias')
        close(l['weight'], weight(h), 'ledger deployment weight')


def shadow_feedback(pool, h, sh, receipt, observation, prefix, arrays, data, t):
    p = sh['prediction']
    require(digest({k: v for k, v in p.items() if k != 'prediction_sha256'}) == p['prediction_sha256'], 'prediction self hash')
    require((p['hypothesis_id'], p['version']) == hkey(h) and p['semantic_key'] == h['semantic_key'], 'prediction rule/version identity')
    require(p['point_id'] == observation['candidate_id'] and p['query_round'] == t, 'prediction point/query')
    j = pool['ix'][p['point_id']]
    require(j in data['probes'], 'forecast lacks a compatible known reference')
    ref = data['probes'][j]
    require(p['reference_id'] == ref['reference_id'] and p['reference_y'] == ref['reference_y'], 'not highest compatible already-observed reference')
    pos = data['positions'][j]
    fields = data['values']
    for name, expected in [('gp_mean_y', arrays['mu_y'][j]), ('gp_var_latent_y', arrays['sd_y'][j]**2),
                           ('gp_var_y', arrays['sd_predictive_y'][j]**2), ('tempering', fields['tempering'][pos]),
                           ('event_log_probability', fields['event_log_probability'][pos])]:
        close(p[name], expected, 'frozen predictor '+name)
    lt = float(fields['log_tempering'][pos])
    require(p['log_tempering'] is None if not np.isfinite(lt) else p['log_tempering'] is not None, 'no-effect/log-tempering contract')
    if np.isfinite(lt): close(p['log_tempering'], lt, 'frozen log tempering')
    require(p['zero_effect_policy'] == (not np.isfinite(lt)) and sh['informative'] == np.isfinite(lt), 'scoreability/no-op flag')
    close(sh['information_nats'], fields['sign_information'][pos], 'binary sign information')
    mu, vy, r, tau = p['gp_mean_y'], p['gp_var_y'], p['reference_y'], p['preference_noise_sd']
    require(tau == .06, 'forecast preference noise')
    alpha = (mu-r)/sqrt(vy+tau*tau)
    lam = exp(-.5*alpha*alpha-.5*log(2*pi)-float(log_ndtr(alpha)))
    cm = mu+vy/sqrt(vy+tau*tau)*lam
    cv = max(0., vy-vy*vy/(vy+tau*tau)*np.clip(lam*(lam+alpha), 0, 1))
    temper = p['tempering']
    for name, value in [('conditional_mean_y', cm), ('conditional_var_y', cv),
                        ('knowledge_mean_y', mu+temper*(cm-mu)),
                        ('knowledge_var_y', (1-temper)*vy+temper*cv+temper*(1-temper)*(cm-mu)**2)]:
        close(p[name], value, 'frozen predictor moment '+name)
    l = h['ledger']
    require(p['point_id'] not in l['seen'] and p['point_id'] not in h['excluded_observed_ids'], 'feedback repeated/retrospective')
    require(all(e['query_round'] < t for e in l['events']), 'feedback order')
    before, before_w = evidence(h), weight(h)
    gain, success = score(p, observation['y']), bool(observation['y'] > ref['reference_y'])
    event = dict(hypothesis_id=h['id'], version=h['version'], point_id=p['point_id'], reference_id=ref['reference_id'],
        reference_y=ref['reference_y'], query_round=t, prediction=deepcopy(p), observed_y=float(observation['y']),
        fact_success=success, delta_prediction_score=gain, delta_log_score=gain)
    event['event_id'] = digest(event)
    l['prediction_score'] += gain
    l['score'] = l['prediction_score']
    l['successes'] += int(success)
    l['failures'] += int(not success)
    l['seen'] = sorted(set(l['seen']) | {p['point_id']})
    l['events'].append(event)
    formula = evidence(h)['formula_weight']
    recover = success and gain > 0
    l['trusted_weight'] = formula if recover else min(before['trusted_weight'], formula)
    l['weight'] = weight(h)
    after = evidence(h)
    expected = dict(event_id=event['event_id'], hypothesis_id=h['id'], version=h['version'], point_id=p['point_id'],
        reference_id=ref['reference_id'], reference_y=ref['reference_y'], query_round=t, delta_log_score=gain,
        delta_prediction_score=gain, fact_success=success, weight_before=before_w, weight_after=l['weight'],
        trusted_weight_before=before['trusted_weight'], trusted_weight_after=after['trusted_weight'],
        score_after=l['prediction_score'], effective_prediction_score=after['effective_prediction_score'],
        successes=after['successes'], failures=after['failures'], reliability_cap=after['reliability_cap'],
        formula_weight=after['formula_weight'], weight_transition='positive_fact_and_positive_forecast' if recover
        else 'nonincreasing_after_failure_or_nonpositive_forecast')
    exact(receipt, expected, 'independent dual-ledger feedback/weight receipt')
    require(recover or after['trusted_weight'] <= before['trusted_weight'], 'failure/nonpositive prediction increased trust')
    return gain


def candidate_menu(pool, prefix, arrays, t, maximum=24):
    eligible = [int(j) for j in np.flatnonzero(arrays['eligible'])]
    if len(eligible) <= 1:
        return [pool['ids'][j] for j in eligible]
    priorities, base = arrays['tie_priority'], arrays['base_logei']
    bo = select(base, arrays['eligible'], priorities)
    features = pool['features']
    def distance(i, j):
        return sum(float(features[i][k] != features[j][k]) if kind == 'categorical'
                   else (float(features[i][k])-float(features[j][k]))**2
                   for k, kind in enumerate(pool['kinds']))
    chosen = [bo]
    ranked = sorted(eligible, key=lambda j: (-base[j], priorities[j]))
    for i in ranked[:min(6, len(ranked))]:
        if i not in chosen: chosen.append(i)
        others = sorted((j for j in eligible if j != i and j not in chosen), key=lambda j: (distance(i, j), priorities[j]))
        for j in others[:2]:
            if len(chosen) < maximum: chosen.append(j)
    while len(chosen) < min(maximum, len(eligible)):
        rest = [j for j in eligible if j not in chosen]
        chosen.append(max(rest, key=lambda j: (min(distance(j, i) for i in chosen), -priorities[j])))
    np.random.default_rng(713000+t).shuffle(chosen)
    protected = [bo]
    refs = sorted(prefix, key=lambda r: (-r['y'], features[pool['ix'][r['candidate_id']]]))[:4]
    for row in refs:
        ref = pool['ix'][row['candidate_id']]
        legal = []
        for j in eligible:
            necessary = sum(features[j][k] != features[ref][k] if kind == 'categorical'
                            else abs(float(features[j][k])-float(features[ref][k])) > .25+ATOL
                            for k, kind in enumerate(pool['kinds']))
            if necessary <= 2 and features[j] != features[ref] and j not in protected:
                legal.append(j)
        if legal and len(protected) < maximum:
            protected.append(min(legal, key=lambda j: (distance(j, ref), priorities[j])))
    final = (protected+[j for j in chosen if j not in protected])[:maximum]
    np.random.default_rng(713000+t).shuffle(final)
    return [pool['ids'][j] for j in final]


def should_call(calls, prior_decisions, rows, t, params):
    if len(calls) >= params['call_limit']: return False
    if not calls: return True
    if t-calls[-1]['query'] < params['minimum_call_gap']: return False
    if t < params['late_call_round'] and len(calls) >= 3: return False
    if t >= params['late_call_round'] and not any(x['query'] >= params['late_call_round'] for x in calls): return True
    no_support = not any(r['can_change_rank_at_cap'] for r in rows)
    recent = prior_decisions[-3:]
    stalled = len(recent) == 3 and recent[-1]['incumbent_after_y'] <= recent[0]['incumbent_before_y']+ATOL
    negative = any(e['delta_log_score'] < -.1 for d in prior_decisions[-2:] for e in d['evidence'])
    return no_support or stalled or negative


def prompt_audit(folder, protocol, pool, call, prefix, c, prepared, rows):
    context = read(folder/'requests'/call['context_sha256']/'context.json')
    require(digest(context) == call['context_sha256'], 'LLM context hash')
    require(context['model'] == protocol['model'] and context['reasoning_effort'] == protocol['reasoning_effort'], 'LLM identity/effort')
    require(digest(context['system']) == protocol['prompt_sha256'], 'LLM frozen system prompt')
    public = context['public']
    require(set(public) == {'task', 'objective_unit', 'query', 'remaining_queries', 'domain_context', 'menu_policy', 'axes',
        'own_observations', 'candidate_menu', 'knowledge_ledger', 'actionable_coverage', 'feedback_note'}, 'LLM field whitelist')
    common.domain_context_audit(public['domain_context'], pool)
    require(public['task'] == pool['task'] and public['query'] == c['query'] and public['remaining_queries'] == 17-c['query'], 'LLM task/round')
    require(public['objective_unit'] == pool['unit'], 'LLM engineering units')
    exact(public['axes'], [dict(name=a, type=k) for a, k in zip(pool['axes'], pool['kinds'])], 'LLM true axes/categories')
    def point(key):
        j = pool['ix'][key]
        return dict(id=key, point=pool['points'][j], normalized_continuous={
            a: float(pool['features'][j][k]) for k, (a, kind) in enumerate(zip(pool['axes'], pool['kinds'])) if kind == 'continuous'})
    exact(public['candidate_menu'], [point(k) for k in c['menu_ids']], 'LLM input-only candidate menu')
    exact(public['own_observations'], [dict(point(h['candidate_id']), objective=h['raw_value']) for h in prefix], 'LLM own exact prefix only')
    ledger = []
    for h in prepared:
        row = {k: deepcopy(h[k]) for k in ('id', 'version', 'parent', 'focus', 'scope', 'categorical_preference', 'status', 'reason') if k in h}
        row['origin_example'] = dict(plus_id=h['plus_id'], minus_id=h['minus_id'])
        row['evidence'], row['deployment_weight'] = evidence(h), weight(h)
        row['tested_pairs'] = [dict(plus_id=e['point_id'], reference_id=e['reference_id'], query_round=e['query_round'],
            plus_objective=e['observed_y']*pool['scale'], reference_objective=e['reference_y']*pool['scale'],
            fact_success=e['fact_success'], delta_prediction_logscore=e['delta_log_score']) for e in h['ledger']['events']]
        row['inherited_counterevidence'] = [dict(event_id=e['event_id'], plus_id=e['point_id'], reference_id=e['reference_id'],
            plus_objective=e['observed_y']*pool['scale'], reference_objective=e['reference_y']*pool['scale'],
            fact_success=e['fact_success'], delta_prediction_logscore=e['delta_log_score'], applicable=e['applicable'])
            for e in h['inherited_negative_evidence']]
        ledger.append(row)
    exact(public['knowledge_ledger'], ledger, 'LLM ledger must exclude all GP/shadow fields and include only past feedback')
    safe = ('id', 'version', 'weight', 'eligible_inside_scope', 'executable_probes', 'remaining_probe_count', 'can_change_rank_at_cap', 'eligibility_state')
    exact(public['actionable_coverage'], [{k: r[k] for k in safe if k in r} for r in rows], 'LLM current executable coverage')
    receipt = read(folder/'requests'/call['context_sha256']/'validated.json')
    exact(call['receipt'], receipt, 'request cache receipt')
    response = read(folder/'requests'/call['context_sha256']/f"response_{receipt['attempt']:02d}.json")
    require(digest(response) == receipt['response_sha256'], 'LLM raw response hash')
    import json
    exact(json.loads(response['output_text']), receipt['proposal'], 'validated response text')
    exact(c['proposal'], receipt['proposal'], 'committed proposal changed')


def audit_run(folder, path, protocol, pool, manifest, include_incomplete=False, joint=False):
    state = read(path)
    params = protocol['parameters']
    trial, tie_seed = common.trial_seeds(protocol)
    require(state['protocol_sha256'] == protocol['protocol_sha256'], 'state protocol')
    require(state['condition'] == 'agent_initial_terra' and state['mode'] in protocol['supported_modes'], 'unsupported V4 arm/condition')
    history, decisions = state['history'], state['decisions']
    require(state['seed'] == trial and len(history) == 4+len(decisions), 'trial/budget/history')
    require(len(decisions) == 16 or include_incomplete and 0 < len(decisions) < 16 and state['status'] != 'COMPLETE', 'query budget')
    require([r['candidate_id'] for r in history[:4]] == manifest['initial_ids'], 'initial design changed')
    common.initial_commit_audit(path, state, protocol, pool, manifest)
    require(len({r['candidate_id'] for r in history}) == len(history), 'duplicate query')
    for row in history:
        require(row['candidate_id'] in pool['ix'] and np.isfinite(row['y']), 'invalid observed point')
        close(row['raw_value'], row['y']*pool['scale'], 'observed engineering unit')
    previous, calls, modes, forecasts = [], [], Counter(), 0
    max_quad_error, roles = 0., []
    seen_predictions = set()
    for t, decision in enumerate(decisions, 1):
        prefix, observation = history[:t+3], history[t+3]
        cp = path.parent/'rounds'/f'{t:02d}_commit.json'
        c = read(cp)
        require(c['query'] == decision['query'] == t and c['mechanism'] == 'rule_probe_v4', 'commit sequence/mechanism')
        require(c['history_sha256'] == digest(prefix), 'commit observed prefix hash')
        require(c['protocol_sha256'] == protocol['protocol_sha256'] and digest(c) == decision['commit_sha256'], 'commit content hash')
        require(c['selected_id'] == decision['selected_id'] == observation['candidate_id'], 'selected outcome identity')
        require(c['mode'] == decision['mode'] and c['BO_reference_id'] == decision['BO_reference_id'], 'selected mode/reference')
        close(decision['y'], observation['y'], 'decision y')
        close(decision['raw_value'], observation['raw_value'], 'decision objective')
        close(decision['incumbent_before_y'], max(r['y'] for r in prefix), 'incumbent before')
        close(decision['incumbent_after_y'], max(r['y'] for r in history[:t+4]), 'incumbent after')
        with np.load(cp.with_name(f'{t:02d}_GP.npz'), allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        require(set(arrays) == {'mu_y', 'sd_y', 'sd_predictive_y', 'base_logei', 'available', 'eligible', 'tie_priority'}, 'GP artifact fields')
        n = len(pool['ids'])
        require(all(x.shape == (n,) and np.isfinite(x).all() for x in arrays.values()), 'invalid stored GP fields')
        require(np.all(arrays['sd_y'] >= 0) and np.all(arrays['sd_predictive_y'] >= arrays['sd_y']), 'stored variance contract')
        available = ~np.isin(pool['ids'], [r['candidate_id'] for r in prefix])
        order = sorted(range(n), key=lambda j: pool['features'][j])
        ties = np.empty(n, np.int64)
        ties[order] = np.random.default_rng(np.random.SeedSequence([tie_seed, t])).permutation(n)
        require(np.array_equal(ties, arrays['tie_priority']), 'physical tie RNG/order')
        require(np.array_equal(available, arrays['available']), 'available mask')
        base = arrays['base_logei']
        bo = select(base, available, ties)
        epsilon = log(4)*(16-t)/15
        eligible = available & (base[bo]-base <= epsilon)
        require(np.array_equal(eligible, arrays['eligible']), 'qLogEI opportunity set')
        selected = pool['ix'][c['selected_id']]
        require(c['BO_reference_id'] == pool['ids'][bo] and eligible[selected], 'BO argmax or selected budget')
        require(c['eligible_count'] == int(eligible.sum()), 'eligible count')
        for field, value in [('epsilon', epsilon), ('base_logei', base[selected]), ('reference_logei', base[bo]), ('opportunity_gap', base[bo]-base[selected])]:
            close(c[field], value, 'commit '+field)
        common.native_seed_audit(pool, c)
        require(c['GP_fit_key'] == c['GP_acquisition_meta']['fit_key'], 'GP acquisition fit key')
        fit = read(cp.with_name(f'{t:02d}_fit.json'))
        require(fit and all(np.isfinite(np.asarray(x)).all() for x in fit.values()), 'nonfinite fit receipt')
        if joint: saved_gp_audit(folder, pool, prefix, c, arrays, fit)
        prepared, events, pre_rows = refresh(pool, previous, prefix, arrays, t)
        exact(c['pre_call_coverage_events'], events, 'pre-call coverage transition')
        want_call = state['mode'] != 'bo' and should_call(calls, decisions[:t-1], pre_rows, t, params)
        expected_menu = candidate_menu(pool, prefix, arrays, t) if want_call else []
        exact(c['menu_ids'], expected_menu, 'semantic candidate menu')
        require(c['calls_at_commit'][:len(calls)] == calls and len(c['calls_at_commit']) == len(calls)+int(want_call and bool(expected_menu)), 'LLM event schedule/call prefix')
        if len(c['calls_at_commit']) > len(calls):
            call = c['calls_at_commit'][-1]
            require(call['query'] == t, 'LLM call current round')
            prompt_audit(folder, protocol, pool, call, prefix, c, prepared, pre_rows)
        else:
            require(c['proposal'] is None, 'proposal without a scheduled call')
        calls = c['calls_at_commit']
        hyps = deepcopy(c['hypotheses_at_commit'])
        validate_hypotheses(pool, hyps, prepared, prefix, t, c['menu_ids'])
        for h in hyps:
            if h['status'] == 'retracted': continue
            ps = probes(pool, h, prefix, eligible)
            require((h['status'] == 'active') == bool(ps), 'rule dormancy/executable probes')
        calc, err = selection_audit(pool, prefix, c, arrays, state['mode'], modes['optional_verification'], params)
        max_quad_error = max(max_quad_error, err)
        expected_shadows = [hkey(h) for h in hyps if hkey(h) in calc and selected in calc[hkey(h)]['probes']]
        require([hkey(s) for s in c['shadows']] == expected_shadows, 'missing/extraneous new-plus forecasts')
        require(len(c['shadows']) == len(decision['evidence']), 'forecast/receipt count')
        weights = {hkey(x): x['effective_mixture_weight'] for x in c['deployed_versions']}
        expected_mixture = [dict(id=s['id'], version=s['version'], weight=weights[hkey(s)], prediction_sha256=s['prediction']['prediction_sha256'])
                            for s in c['shadows'] if weights.get(hkey(s), 0) > 0]
        exact(c['selected_mixture'], expected_mixture, 'selected actual deployed mixture')
        scores = {}
        for sh, receipt in zip(c['shadows'], decision['evidence']):
            pred_sha = sh['prediction']['prediction_sha256']
            require(pred_sha not in seen_predictions, 'prediction reused')
            seen_predictions.add(pred_sha)
            h = next(h for h in hyps if hkey(h) == hkey(sh))
            scores[pred_sha] = shadow_feedback(pool, h, sh, receipt, observation, prefix, arrays, calc[hkey(sh)], t)
            forecasts += 1
        mixture = c['selected_mixture']
        total = sum(r['weight'] for r in mixture)
        actual_gain = float(logsumexp([log(1-total)]+[log(r['weight'])+scores[r['prediction_sha256']] for r in mixture])) if mixture else 0.
        close(decision['deployed_mixture_log_score_gain'], actual_gain, 'actual deployed mixture logscore')
        require(actual_gain >= -log(2)-ATOL, 'half-GP mixture predictive logloss bound')
        modes[c['mode']] += 1
        roles.append(dict(query=t, mode=c['mode'], selected_id=c['selected_id'], BO_reference_id=c['BO_reference_id'],
            deployed_experts=len(c['selected_mixture']), forecasts=len(c['shadows']), mixture_gain=actual_gain,
            positive_prediction_failed_fact=sum(e['delta_log_score'] > 0 and not e['fact_success'] for e in decision['evidence'])))
        previous = hyps
    exact(state['hypotheses'], previous, 'state final rule ledgers differ from independent replay')
    exact(state['calls'], calls, 'state final call history')
    require(state['verification_used'] == modes['optional_verification'] <= params['maximum_optional_verifications'], 'optional query budget')
    require(state.get('cache_counters', {}).get('operator_compiles_in_loop', 0) == 0, 'operators compiled during experiment loop')
    raw = np.array([h['raw_value'] for h in history])
    curve = np.maximum.accumulate(raw)
    return dict(task=state['task'], group=pool['group'], mode=state['mode'], condition=state['condition'], status=state['status'],
        technical_checks='PASS_LISTED_CHECKS', queries=len(decisions), forecasts=forecasts, selection_modes=dict(modes),
        initial_history_sha256=digest(history[:4]), SearchAUC=float(curve[4:].mean()), final_best=float(curve[-1]),
        first_best_total=int(np.flatnonzero(raw == curve[-1])[0])+1,
        argmax_scope='FULL_CANDIDATE_RULE_AND_OPTIONAL_SELECTION', saved_GP_and_MC512_recomputed=bool(joint),
        max_selected_BO_quad_logratio_error=max_quad_error, roles=roles, run=path.relative_to(ROOT).as_posix())


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='v4')
    parser.add_argument('--joint', action='store_true', help='Also reconstruct saved GP and MC512; never fit')
    parser.add_argument('--allow-source-drift', action='store_true')
    parser.add_argument('--include-incomplete', action='store_true')
    parser.add_argument('--task', action='append')
    parser.add_argument('--run', action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(args.variant == 'v4', 'This independent auditor supports only the registered V4 mechanism')
    folder = ROOT/'experiments/runs/knowledge_loop_20261101'/args.variant
    p = read(folder/'protocol.json')
    require(p['protocol_sha256'] == digest({k: v for k, v in p.items() if k != 'protocol_sha256'}), 'protocol self hash')
    require(p['variant'] == 'v4' and p['mechanism'] == 'rule_probe_v4' and p['initial'] == 4 and p['queries'] == 16, 'V4 protocol identity/budget')
    common.trial_seeds(p)
    require(p['seed'] == 0, 'V4 seed1 was already inspected; no new confirmation implied')
    drift = [name for name, expected in p['source_sha256'].items() if not (ROOT/name).exists() or filehash(ROOT/name) != expected]
    require(not drift or args.allow_source_drift, 'Frozen source changed; historical internals require --allow-source-drift')
    goal_path = p.get('goal_config_path', 'code/resources/knowledge_loop_goal_20261101_majority.json')
    require(filehash(ROOT/goal_path) == p['goal_config_sha256'], 'goal-config custody')
    uninspected_input_receipts = []
    for item in p['inputs']:
        for path, expected in item['files_sha256'].items():
            if path.endswith('/pool.json'):
                require(filehash(ROOT/path) == expected, 'input pool content SHA: '+path)
            else:
                # The inherited manifest also records full oracle/advice
                # files. Their protocol hashes are retained without opening
                # them: this audit is restricted to candidate-only pools and
                # already observed histories.
                uninspected_input_receipts.append(dict(path=path, declared_sha256=expected))
    source = common.source_order_audit(ROOT)
    require(source['initial_commit_before_initial_oracle'], 'initial commitment must precede initial outcome read')
    manifests = {x['task']: x for x in p['inputs']}
    pools = {task: dict(common.public_pool(ROOT, item), task=task) for task, item in manifests.items()}
    rows, failures, pending = [], [], []
    for path in sorted((folder/'runs').glob('*/state.json')):
        state = read(path)
        if args.run and path.parent.name not in args.run: continue
        if args.task and state['task'] not in args.task: continue
        if state['status'] != 'COMPLETE':
            pending.append(dict(run=path.parent.name, status=state['status'], queries=len(state['decisions'])))
            if not args.include_incomplete or not state['decisions']: continue
        try:
            rows.append(audit_run(folder, path, p, pools[state['task']], manifests[state['task']], args.include_incomplete, args.joint))
            print(json.dumps(dict(audited=path.parent.name, queries=rows[-1]['queries'], result='PASS_LISTED_CHECKS'), ensure_ascii=False), flush=True)
        except Exception as error:
            failures.append(dict(run=path.parent.name, error_type=type(error).__name__, error=str(error)))
            print(json.dumps(dict(audited=path.parent.name, result='FAIL', error=str(error)), ensure_ascii=False), flush=True)
    baselines = {r['task']: r for r in rows if r['mode'] == 'bo' and r['status'] == 'COMPLETE'}
    for row in rows:
        if row['mode'] == 'bo' or row['status'] != 'COMPLETE' or row['task'] not in baselines: continue
        bo = baselines[row['task']]
        require(row['initial_history_sha256'] == bo['initial_history_sha256'], 'matched baseline initial outcomes differ')
        row['AUC_delta_vs_matched_BO'] = row['SearchAUC']-bo['SearchAUC']
        row['final_delta_vs_matched_BO'] = row['final_best']-bo['final_best']
    comparisons = []
    for arm in sorted({r['mode'] for r in rows if r['mode'] != 'bo'}):
        same = [r for r in rows if r['mode'] == arm and 'AUC_delta_vs_matched_BO' in r]
        comparisons.append(dict(mode=arm, matched_tasks=len(same),
            strict_SearchAUC_wins=sum(r['AUC_delta_vs_matched_BO'] > 1e-9 for r in same),
            SearchAUC_ties=sum(abs(r['AUC_delta_vs_matched_BO']) <= 1e-9 for r in same),
            SearchAUC_losses=sum(r['AUC_delta_vs_matched_BO'] < -1e-9 for r in same)))
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), variant=args.variant,
        auditor_sha256=filehash(Path(__file__)), protocol_sha256=p['protocol_sha256'], source_drift=drift,
        technical_result='PASS_LISTED_CHECKS' if rows and not failures else 'FAIL_OR_NO_AUDITED_RUNS',
        performance_goal_achieved=None, performance_interpretation='Development comparisons only; technical PASS is not scientific superiority.',
        read_only=True, fits=0, LLM_calls=0, unqueried_targets_read=0,
        source_order=source, joint=bool(args.joint), runs=rows, failures=failures, pending=pending,
        uninspected_input_receipts=uninspected_input_receipts,
        totals=dict(runs=len(rows), rounds=sum(r['queries'] for r in rows), forecasts=sum(r['forecasts'] for r in rows)),
        comparisons=comparisons,
        limitations=['Commit hashes prove internal causal ordering; they are not an external timestamp attestation.',
            'Source/schema inspection is not an OS-level I/O trace.',
            'Only candidate-only pool inputs were rehashed; oracle/advice custody remains the frozen protocol receipt and was not reopened.',
            'GP is reconstructed from saved fitted parameters, never independently optimized.',
            'GH64 is approximate; selected and executable BO forecasts have a separate adaptive F-integral check.',
            'Native MC512 qLogEI times conditional-normal-EI ratio remains the preregistered acquisition approximation.',
            *([] if args.joint else ['Stored GP/MC arrays were not reconstructed; full rule argmax uses the saved arrays.'])])
    output = args.output or ROOT/'experiments/reports/knowledge_loop_20261101/v4_independent_audit.json'
    output = output.resolve()
    approved = (ROOT/'experiments/reports/knowledge_loop_20261101').resolve()
    require(output.is_relative_to(approved), 'Audit output must stay in dedicated reports directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(result=report['technical_result'], totals=report['totals'], report=str(output)), ensure_ascii=False), flush=True)
    return int(bool(failures) or not rows)


if __name__ == '__main__':
    raise SystemExit(main())
