import numpy as np
import torch
from pathlib import Path
from chem.benchmarks import stable_additive_fit as a
from chem.benchmarks import reasoning_bo_data as data, pointwise_hte_study as common


def test_kernel_is_PSD_and_category_label_invariant():
    x=torch.tensor([[0,0,1],[0,1,2],[2,0,2],[2,1,1],[2,1,2]],dtype=torch.double)
    k=a.AdditiveInteractionKernel().double();m=k(x).to_dense().detach()
    assert torch.linalg.eigvalsh(m).min()>=-1e-12
    torch.testing.assert_close(k(31-7*x).to_dense(),m,atol=0,rtol=0)
    torch.testing.assert_close(m.diag(),torch.ones(5,dtype=torch.double))
    assert m[0,1]>m[0,4]


def test_collapse_switch_is_cached_and_healthy_prefix_stays_native(tmp_path):
    common.warm()
    for task,expected in [('Perera_Suzuki',True),('Doyle_Buchwald_Hartwig',False)]:
        root=Path('data/runs/hte_r2_tuning_20261028/focused_countercheck_v3_shuffle_s0')
        hist=data.read(root/'runs'/(task+'_s0_focused_countercheck_v3_shuffle_s0')/'state.json')['history'][:6]
        c=a.Cache(data.load_pool(task),tmp_path/task);c.sources=[root/'cache'/task]
        key,val=c.model(hist);record=c.last_health
        assert c.latched==expected
        if expected:
            assert record['model_kind']=='additive_interaction'
            assert isinstance(val['model'].covar_module.base_kernel,a.AdditiveInteractionKernel)
            assert np.isfinite(c.stats(hist)['mu_y']).all()
        else:assert not record['used_robust']
        c.release();assert c.model(hist)[0]==key
