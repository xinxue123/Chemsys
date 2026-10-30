"""Mechanism and no-future-data tests for uncertainty use."""
import numpy as np
import pytest
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from chem.benchmarks import uncertain_suzuki_engine as e


def fixture():
    return np.array([0.,-.1,-.2,-2.]),np.array([1.,.9,-1.,-1.]),np.array([.1,.2,.1,.1]),np.array([[False,True,True,True],[False,True,False,True]]),np.zeros(4,dtype=bool),['a','b','c','d'],['H1','H2']


def test_credible_guard_revokes_jointly_and_respects_score_budget():
    base,mu,sd,h,q,ids,hids=fixture()
    selected,event=e.select('credible_B4',1.,base,mu,sd,h,.3,q,ids,hids,lambda *a:base,1)
    assert selected==1 and event['retained_ids']==['H1','H2']
    assert event['base_logei_loss']<=.3
    # A seemingly feasible but confidently dominated clause must be revoked.
    h[:,1]=False
    selected,event=e.select('credible_B4',1.,base,mu,sd,h,.3,q,ids,hids,lambda *a:base,1)
    assert selected==0 and event['retained_ids']==[]


def test_high_uncertainty_preserves_an_otherwise_dominated_region():
    base,mu,sd,h,q,ids,hids=fixture();h[:,1]=False;sd[2]=3.
    selected,event=e.select('credible_B4',1.,base,mu,sd,h,.3,q,ids,hids,lambda *a:base,1)
    assert selected==2 and event['retained_ids']==['H1']


def test_boundary_probe_is_not_unconditional_maximum_variance():
    base,mu,sd,h,q,ids,hids=fixture();sd[:]=[.1,10.,2.,20.]
    target,_=e.select('boundary_probe',3,base,mu,sd,h,.3,q,ids,hids,lambda *a:base,4)
    control,_=e.select('sigma_probe',3,base,mu,sd,h,.3,q,ids,hids,lambda *a:base,4)
    assert target==2 and control==1


def test_residual_scale_uses_only_completed_prefix_and_is_bounded():
    assert e.residual_scale([])==1.
    assert e.residual_scale([{'predictive_z':100.}]*2)==1.
    assert e.residual_scale([{'predictive_z':2.}]*3)==2.
    assert e.residual_scale([{'predictive_z':100.}]*3)==3.
    assert e.residual_scale([{'predictive_z':.1}]*3)==1.


def test_cap_preserves_both_shift_signs_and_covariance():
    class Fake(Model):
        def __init__(self,means):super().__init__();self.means=torch.tensor(means,dtype=torch.double)
        @property
        def num_outputs(self):return 1
        def posterior(self,X,**kwargs):return GPyTorchPosterior(MultivariateNormal(self.means,torch.eye(2,dtype=torch.double)))
    base=Fake([0.,0.]);guided=Fake([-5.,5.]);model=e.SigmaCappedModel(base,guided,.5)
    p=model.posterior(torch.zeros(2,4,dtype=torch.double))
    assert torch.equal(p.mean.squeeze(-1),torch.tensor([-.5,.5],dtype=torch.double))
    assert torch.equal(p.mvn.covariance_matrix,torch.eye(2,dtype=torch.double))


def test_final_budget_keeps_only_base_optimal_points():
    base,mu,sd,h,q,ids,hids=fixture()
    for m,p in [('credible_B4',2),('residual_B4',2),('boundary_probe',4),('sigma_probe',4),('sigma_BO',4),('sigma_cap',1)]:
        selected,event=e.select(m,p,base,mu,sd,h,0.,q,ids,hids,lambda *a:np.array([0.,20.,10.,50.]),16)
        assert selected==0 and event['base_logei_loss']==0.


def test_cache_copy_publishes_only_a_complete_file(tmp_path,monkeypatch):
    source=tmp_path/'source';target=tmp_path/'target';source.mkdir();target.mkdir()
    (source/'a.pt').write_bytes(b'complete-state')
    cache=object.__new__(e.UncertaintyCache);cache.directory=target;cache.sources=[source];cache.counters={'copied':0}
    original=e.shutil.copyfile
    def checked_copy(src,dst):
        assert not (target/'a.pt').exists()
        assert str(dst).endswith('.copytmp')
        return original(src,dst)
    monkeypatch.setattr(e.shutil,'copyfile',checked_copy);cache.ensure('a.pt')
    assert (target/'a.pt').read_bytes()==b'complete-state'
    assert not list(target.glob('*.copytmp')) and cache.counters['copied']==1
