"""Read-only saved-state GP blocks; no fitting, oracle access, or cache writes.

Reuses the reconstruction equations reviewed in audit_knowledge_loop.py,
joint_selection_audit. Only training points and the already-selected point are
evaluated. Covariance blocks are returned in chronological prefix order.
"""
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from knowledge_shadow import digest


class SavedJointGP:
    def __init__(self, cache_path, pool, prefix, commit, fit_parameters):
        if digest(prefix) != commit['history_sha256']:
            raise ValueError('Saved GP reconstruction requires the exact committed prefix')
        self.history_sha256 = commit['history_sha256']
        self.gp_fit_key = commit['GP_fit_key']
        self.selected_id = commit['selected_id']
        torch.set_num_threads(1)
        saved = torch.load(cache_path, map_location='cpu', weights_only=True)
        if saved['key'] != commit['GP_fit_key']:
            raise ValueError('Cached fit key differs from the committed GP')
        ix, raw, group = pool['ix'], pool['raw'], pool['group']
        def vector(candidate_id):
            j = ix[candidate_id]
            if group == 'P0':
                labels = ['P1-L1', 'P2-L1', 'P1-L2', 'P1-L3', 'P1-L4', 'P1-L5', 'P1-L6', 'P1-L7']
                return list(np.clip(raw['z'][j][:3], 1e-6, 1 - 1e-6)) + [
                    saved['category_map'][labels.index(raw['points'][j]['catalyst'])]]
            if group == 'P1':
                return [saved['mapping'][a][raw['points'][j][a]] for a in pool['axes']]
            return raw['z'][j]
        ordered = sorted(prefix, key=(lambda h: pool['features'][ix[h['candidate_id']]])
                         if group == 'P2' else (lambda h: h['candidate_id']))
        train = torch.tensor([vector(h['candidate_id']) for h in ordered], dtype=torch.double)
        selected = torch.tensor([vector(commit['selected_id'])], dtype=torch.double)
        y = torch.tensor([[h['y']] for h in ordered], dtype=torch.double)
        outer_mean, outer_std = y.mean(), y.std(unbiased=False).clamp_min(1e-12)
        ys = (y - outer_mean) / outer_std
        if 'train_candidate_ids' in saved:
            if saved['train_candidate_ids'] != [h['candidate_id'] for h in ordered]:
                raise ValueError('Cached GP uses a different training prefix')
            if not torch.equal(torch.as_tensor(saved['train_X'], dtype=torch.double), train):
                raise ValueError('Cached GP inputs differ from the committed public inputs')
        elif saved.get('history') != [{'candidate_id': h['candidate_id'], 'y': float(h['y'])}
                                      for h in ordered]:
            raise ValueError('Cached GP history differs from this observed prefix')
        if group == 'P2':
            model = SingleTaskGP(train, ys)
        else:
            cats = [3] if group == 'P0' else list(range(len(pool['axes'])))
            model = MixedSingleTaskGP(train, ys, cat_dims=cats)
        model.load_state_dict(saved['state'])
        model.eval()
        actual_fit = {name: p.detach().cpu().tolist() for name, p in model.named_parameters()}
        if actual_fit != fit_parameters:
            raise ValueError('Cached weights differ from the per-round fit receipt')
        with torch.no_grad():
            noise = model.likelihood.noise.reshape(())
            K = model.covar_module(train).to_dense() + torch.eye(len(train), dtype=torch.double) * noise
            cross = model.covar_module(selected, train).to_dense().reshape(-1)
            scale2 = model.outcome_transform.stdvs.reshape(()).square() * outer_std.square()
            m = model.mean_module(selected).reshape(()) + cross @ torch.linalg.solve(
                K, model.train_targets - model.mean_module(train))
            self.mean = float((m * model.outcome_transform.stdvs.reshape(()) +
                               model.outcome_transform.means.reshape(())) * outer_std + outer_mean)
            latent = model.covar_module(selected, diag=True).reshape(()) - cross @ torch.linalg.solve(K, cross)
            self.predictive_variance = float((latent + noise) * scale2)
            index = {h['candidate_id']: i for i, h in enumerate(ordered)}
            order = [index[h['candidate_id']] for h in prefix]
            # Scalar scale cancels in projection but common units are audited.
            self.k_noise = (K * scale2).numpy()[np.ix_(order, order)]
            self.k_x = (cross * scale2).numpy()[order]
        self.training_ids = [h['candidate_id'] for h in prefix]
        self.training_points = [dict(zip(pool['axes'], pool['features'][ix[i]])) for i in self.training_ids]

    def conditioning(self, commit):
        if (commit['history_sha256'] != self.history_sha256 or
                commit['GP_fit_key'] != self.gp_fit_key or
                commit['selected_id'] != self.selected_id):
            raise ValueError('Cannot relabel saved covariance blocks with a different commit')
        return dict(history_sha256=commit['history_sha256'], gp_fit_key=commit['GP_fit_key'],
                    training_ids=self.training_ids, training_points=self.training_points,
                    k_noise=self.k_noise, k_x=self.k_x)
