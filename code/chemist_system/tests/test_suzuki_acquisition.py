import numpy as np
import pytest
import torch
from botorch.acquisition.analytic import ExpectedImprovement,LogExpectedImprovement
from botorch.models import SingleTaskGP

from chem.benchmarks.acquisition_suzuki_engine import beta_at,equal_capacity_mask
from chem.benchmarks.acquisition_suzuki_study import state_path


def test_official_schedule_and_beta_convention():
    for t in range(1,17):assert beta_at('official',t)==2*np.log(t*4*np.pi**2/.1*6)
    for k in (.5,1.,2.):assert np.sqrt(beta_at('fixed',1,k))==k
    assert 3.9<np.sqrt(beta_at('official',1))<4.0
    assert 4.5<np.sqrt(beta_at('official',16))<4.7


@pytest.mark.parametrize('t',[0,17])
def test_invalid_iteration(t):
    with pytest.raises(ValueError):beta_at('official',t)


def test_equal_capacity_lexical_ties_and_unavailable():
    scores=np.array([4.,9.,4.,2.]);available=np.array([1,0,1,1],dtype=bool)
    mask=equal_capacity_mask(scores,available,1,['b','a','a','c'])
    assert mask.tolist()==[False,False,True,False]
    assert equal_capacity_mask(scores,available,3,['b','a','a','c']).sum()==3
    assert not equal_capacity_mask(scores,available,0,['b','a','a','c']).any()


def test_exact_analytic_log_is_monotonic_EI():
    model=SingleTaskGP(torch.tensor([[0.],[.5],[1.]],dtype=torch.double),torch.tensor([[0.],[1.],[.4]],dtype=torch.double)).eval()
    x=torch.linspace(0,1,20,dtype=torch.double).reshape(-1,1,1)
    with torch.no_grad():
        ei=ExpectedImprovement(model,1.)(x);logei=LogExpectedImprovement(model,1.)(x)
    assert torch.equal(ei.argsort(),logei.argsort())
    assert torch.allclose(ei.log(),logei,atol=1e-10,rtol=1e-10)


def test_selected_development_trajectory_is_reference():
    assert 'development' in state_path(1,0,'none','fixed',1.).parts
    assert 'main' in state_path(2,0,'none','fixed',1.).parts
    assert 'main' in state_path(1,0,'correct','fixed',1.).parts
