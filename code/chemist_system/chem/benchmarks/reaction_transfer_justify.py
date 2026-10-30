"""Run upstream LLINBO Justify verbatim, with named finite nominal I/O hooks."""
from contextlib import ExitStack
from hashlib import sha256
import json
from unittest.mock import patch

import torch

from .official_suzuki_adapter import modules
from .llinbo_guarded_engine import MatchedThreshold
from . import reaction_transfer_engine as e
from . import reaction_transfer_llm as llm
from .suzuki_domain import digest


def execute(pool, state, cache, propose, objective):
    upstream = modules('LLINBO')
    ids = pool['ids']
    by_vector = {tuple(x): i for i, x in enumerate(cache.X.tolist())}
    index = {key: i for i, key in enumerate(ids)}
    warm = [(tuple(cache.X[index[h['candidate_id']]].tolist()), h['y']) for h in state['history'][:4]]
    active = {}
    gates = []

    def convert(history):
        return [{'candidate_id': ids[by_vector[tuple(x)]], 'y': float(y)} for x, y in history]

    def warmstart(self, num_warmstart=1, objective_function=None):
        assert num_warmstart == 4
        return list(warm)

    def fit(history):
        prefix = convert(history)
        scores, _, meta = cache.score(prefix)
        active['current'] = {'history': prefix, 'scores': scores, 'metadata': meta}
        return active['current']

    def maximum(model, bounds, **kwargs):
        return MatchedThreshold(4)

    def argmax(model, bounds, beta):
        available = ~e.np.isin(ids, [h['candidate_id'] for h in model['history']])
        j = e.argmax(model['scores'], available, ids)
        active.update(iteration=len(model['history'])-3, reference=j, source_beta=float(beta))
        # Native prompt history is shuffled; keep that audit text deterministic.
        e.seed_all(40102026+cache.task_seed*100+active['iteration'])
        return cache.X[j:j+1]

    def query(self, prompt, model='gpt-3.5-turbo', max_tokens=2000):
        prefix = convert(self.history)
        public = {'task': state['task'], 'phase': 'point_proposal',
            'reaction_class': pool['reaction_class'], 'fixed_substrates': pool['fixed_substrates'],
            'objective_unit': pool['objective_unit'], 'domain': llm.domain(pool),
            'iteration': active['iteration'], 'remaining_queries': 17-active['iteration'],
            'observations': llm.observations(pool, {'history': prefix, 'decisions': []}, include_gp=False),
            'uncertain_preferences': llm.public_knowledge(pool, state['initial_knowledge'])}
        projection, response = propose(prefix, public, prompt, active['iteration'])
        active['proposal'] = projection
        active['response'] = response
        return json.dumps(cache.X[index[projection['candidate_id']]].tolist())

    def acquisition(model, beta):
        gate = {'iteration': active['iteration'], 'source_beta_computed_but_unused': float(beta),
                'budget_logei': float(e.np.log(4)*(16-active['iteration'])/15), 'values': []}
        gates.append(gate)
        class Lookup:
            def __call__(self, X):
                j = by_vector[tuple(X.detach().reshape(-1).tolist())]
                value = float(model['scores'][j])
                gate['values'].append(value)
                return torch.tensor(value, dtype=torch.double)
        return Lookup()

    def evaluate(vector):
        j = by_vector[tuple(vector)]
        row = gates[-1]
        best, proposed = row['values']
        reject = best > proposed + row['budget_logei']
        expected = ids[active['reference']] if reject else active['proposal']['candidate_id']
        assert ids[j] == expected
        loss = best-float(active['current']['scores'][j])
        assert loss <= row['budget_logei']
        event = {'iteration': active['iteration'], 'selected_id': ids[j], 'reference_id': ids[active['reference']],
            'proposed_id': active['proposal']['candidate_id'], 'accepted': not reject,
            'projection': active['proposal'], 'proposal_logei_loss': best-proposed,
            'base_logei_loss': loss, 'budget_logei': row['budget_logei'],
            'base_acquisition': active['current']['metadata'],
            'response_context_sha256': active['response']['context_sha256'],
            'source_beta_computed_but_unused': row['source_beta_computed_but_unused']}
        return objective(ids[j], event, active['current']['history'])

    description = ('A fixed-substrate '+pool['reaction_class']+' yield optimization. '
        'All varying axes are NOMINAL identifiers. Actual provider I/O uses named category IDs and their source names/SMILES. '
        'The original numeric [0,1]^d range and distance phrasing do not apply to category codes. '
        'No ordered/Euclidean metric on codes is used by the GP or projection.')
    with ExitStack() as stack:
        for obj, name, replacement in [(upstream.LLAMAGENT_L, 'llm_warmstarting', warmstart),
            (upstream.LLAMAGENT_L, 'query_llm', query), (upstream, 'train_gp', fit),
            (upstream, 'find_max_variance_bound', maximum), (upstream, 'optimize_acqf_ucb', argmax),
            (upstream, 'UpperConfidenceBound', acquisition)]:
            stack.enter_context(patch.object(obj, name, replacement))
        agent = upstream.LLMIBO_BFO('justify', evaluate, dim=len(pool['axes']), desc=description,
                                  T=16, T_ini=4, T_rep=1, verbose=False)
        histories, regrets = agent.run()
    return {'native_gate_calls': len(gates), 'source_entrypoint': 'LLMIBO_BFO.run -> unchanged _run_justify',
        'source_function_sha256': sha256(upstream.LLMIBO_BFO._run_justify.__code__.co_code).hexdigest()}
