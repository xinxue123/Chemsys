from pathlib import Path
import numpy as np
import torch
from chem.benchmarks import stable_structure_fit as a
from chem.benchmarks import reasoning_bo_data as data, pointwise_hte_study as common


def test_kernel_is_PSD_diagonal_and_category_recoding_invariant():
    pool=data.load_pool('Perera_Suzuki')
    c=a.Cache(pool,Path('outputs/structure_kernel_test'))
    x=torch.tensor([[0,0,1,0,0],[0,1,2,1,1],[2,0,2,2,2],[2,1,1,3,3],[2,1,2,4,0]],dtype=torch.double)
    k=a.MolecularCategoricalKernel(c.matrices)
    m=k(x).to_dense().detach()
    assert torch.linalg.eigvalsh(m).min()>=-1e-12
    torch.testing.assert_close(m.diag(),torch.ones(5,dtype=torch.double))
    torch.testing.assert_close(k(x,diag=True),m.diag())
    recoded=x.clone();matrices=[];rng=np.random.default_rng(17)
    for j,original in enumerate(c.matrices):
        original=np.asarray(original);p=rng.permutation(len(original));inverse=np.argsort(p)
        matrices.append(original[np.ix_(inverse,inverse)])
        recoded[:,j]=torch.as_tensor(p[x[:,j].long().numpy()])
    other=a.MolecularCategoricalKernel(matrices)
    torch.testing.assert_close(other(recoded).to_dense(),m,atol=0,rtol=0)
    batched=x[:,None,:]
    torch.testing.assert_close(k(batched).to_dense().flatten(),m.diag())


def test_trigger_prefix_finite_predictions_and_healthy_path(tmp_path):
    common.warm()
    root=Path('data/runs/hte_r2_tuning_20261028/focused_countercheck_v3_shuffle_s0')
    for task,expected in [('Perera_Suzuki',True),('Doyle_Buchwald_Hartwig',False)]:
        hist=data.read(root/'runs'/(task+'_s0_focused_countercheck_v3_shuffle_s0')/'state.json')['history'][:4]
        c=a.Cache(data.load_pool(task),tmp_path/task);c.sources=[root/'cache'/task]
        key,val=c.model(hist);record=c.last_health
        assert c.latched==expected,(task,record)
        if expected:
            assert record['model_kind']=='molecular_categorical'
            assert record['identification']['trigger'] and record['identification']['isolated_pairs'][record['identification']['critical_axis']]==0
            assert isinstance(val['model'].covar_module.base_kernel,a.MolecularCategoricalKernel)
            assert np.isfinite(c.stats(hist)['mu_y']).all()
        else:assert not record['used_robust']
        c.release();assert c.model(hist)[0]==key
