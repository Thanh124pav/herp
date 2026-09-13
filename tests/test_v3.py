import math
from types import SimpleNamespace
import numpy as np
import torch
from herp.config import HERPV3Config
from herp.archive import RegionArchive
from herp.chain_features import RunningFeatureNormalizer,symmetric_gaussian_kl,diagonal_gaussian_kl,chain_entry_distance
from herp.chain_partition import BoundaryDetector,ChainRegionizer,Chain
from herp.sigma import direct_q_estimate,full_window_q,root_total_variance
from herp.sigma_predictor import LinearVariancePredictor,combined_q
from herp.allocator import v3_priority_distribution,allocate_fragments
from herp.acquisition import AcquisitionScheduler,RolloutFragment
from herp.region_graph import RegionGraph


def fragment(x,mask=None):
    return SimpleNamespace(traj_features=x,valid_mask=torch.ones(len(x),dtype=torch.bool) if mask is None else mask)


def test_gaussian_kl_matches_torch():
    torch.manual_seed(9)
    a,b=torch.randn(7,3),torch.randn(7,3)
    la,lb=torch.randn(7,3),torch.randn(7,3)
    expected=torch.distributions.kl_divergence(torch.distributions.Normal(a,la.exp()),torch.distributions.Normal(b,lb.exp())).sum(-1)
    torch.testing.assert_close(diagonal_gaussian_kl(a,la,b,lb),expected)
    torch.testing.assert_close(symmetric_gaussian_kl(a,la,a,la),torch.zeros(7))


def test_constant_policy_scale_invariance_and_switch():
    cfg=HERPV3Config()
    for scale in [1,10,100,1000]:
        detector=BoundaryDetector(cfg)
        boundaries=[detector.observe(0.,float(scale),t)[0] for t in range(200)]
        assert sum(boundaries)==0
        assert detector.observe(30.,float(scale),200)[0]


def test_normalizer_scale_equivariance():
    torch.manual_seed(12)
    x=torch.randn(500,4);scaled=x*torch.tensor([1.,1000.,.1,10.])
    a,b=RunningFeatureNormalizer(),RunningFeatureNormalizer();a.update(x);b.update(scaled)
    torch.testing.assert_close(a.normalize(x),b.normalize(scaled),atol=2e-6,rtol=2e-6)


def test_chain_clustering_ignores_interiors():
    ar=RegionArchive();cfg=HERPV3Config();reg=ChainRegionizer(ar,cfg)
    for n in [2,1000]:
        c=Chain(0,0,n-1,torch.arange(n).float()[:,None],torch.zeros(n,1),torch.zeros(n,1),torch.zeros(n,1),torch.arange(n),entry_feature=torch.zeros(2))
        assert reg.assign(c)==1
    assert len(ar)==2 and ar.regions[0].is_root


def test_sigma_identical_insufficient_mask_and_permutation():
    fs=[fragment(torch.zeros(32,3)) for _ in range(4)]
    assert direct_q_estimate(fs)==(0.,6)
    assert math.isnan(direct_q_estimate(fs[:1])[0])
    fs[0].traj_features[0]=1.
    assert math.isclose(direct_q_estimate(fs)[0],direct_q_estimate(fs[::-1])[0],rel_tol=1e-12)
    fs=[fragment(torch.ones(32,3),torch.arange(32)<2),fragment(torch.zeros(32,3))]
    assert math.isnan(direct_q_estimate(fs,8)[0])


def test_sigma_unbiased_known_distribution_and_fast_equivalence():
    torch.manual_seed(33)
    x=torch.randn(500,8,16,3)*torch.tensor([1.,2.,3.])
    estimates=x.double().var(1,unbiased=True).sum(-1).mean(-1)
    assert abs(float(estimates.mean())-14.)<.25
    fs=[fragment(f) for f in x[0]]
    assert abs(direct_q_estimate(fs)[0]-full_window_q(x[0]))<1e-5


def test_predictor_realizable_intercept_clipping_and_weights():
    torch.manual_seed(42)
    p=LinearVariancePredictor(3,ridge=1e-8)
    X=torch.cat([torch.ones(1000,1),torch.randn(1000,2)],-1).double()
    y=X@torch.tensor([2.,.5,-.25],dtype=torch.float64)+.02*torch.randn(1000)
    for x,q in zip(X,y):p.add_label(x,float(q),1)
    p.fit();torch.testing.assert_close(p.raw_coefficients(),torch.tensor([2.,.5,-.25],dtype=torch.float64),atol=.01,rtol=0)
    assert p.predict(torch.tensor([1.,-100.,0.]))==0
    p=LinearVariancePredictor(1,ridge=1000.)
    p.add_label([1],2,1);p.add_label([1],10,3);p.fit()
    assert abs(p.predict([1])-8)<1e-8
    assert combined_q(float('nan'),4,0)==4
    assert combined_q(2,4,8)==3


def test_root_graph_and_total_variance():
    graph=RegionGraph();graph.observe_path([0,1,1,2,1]);graph.observe_path([0,2])
    assert graph.child_probabilities(0)=={1:.5,2:.5}
    assert graph.edge_counts[1,2]==1
    assert root_total_variance({1:.5,2:.5},{1:torch.tensor([0.]),2:torch.tensor([2.])},{1:2.,2:4.})==4.
    assert math.isnan(root_total_variance({1:1.},{},{1:0.}))


def test_allocator_analytic_uniform_and_root_zero_allowed():
    # Analytic p*sigma requires the un-normalized path.
    cfg=HERPV3Config(relevance_floor=0.,score_normalize='none')
    for p,s,expect in [([1,1],[1,1],[.5,.5]),([2,1],[1,1],[2/3,1/3]),([1,1],[3,1],[.75,.25]),([2,1],[3,1],[6/7,1/7])]:
        regs=[SimpleNamespace(p_ema=a,sigma_raw=b) for a,b in zip(p,s)]
        torch.testing.assert_close(v3_priority_distribution(regs,cfg),torch.tensor(expect,dtype=torch.float64))
    regs=[SimpleNamespace(p_ema=0.,sigma_raw=.001) for _ in range(4)]
    torch.testing.assert_close(v3_priority_distribution(regs,HERPV3Config()),torch.full((4,),.25,dtype=torch.float64))
    # Default (rank-normalized): the top-p top-sigma region should dominate;
    # a region high in one but bottom in the other is de-weighted, so
    # neither factor can swamp the other regardless of raw scale.
    cfg_rank=HERPV3Config(relevance_floor=1e-3)
    regs=[SimpleNamespace(p_ema=p,sigma_raw=s)
          for p,s in [(0.01,50.),(0.9,0.001),(0.5,1.),(0.7,10.)]]
    dist=v3_priority_distribution(regs,cfg_rank)
    assert dist.argmax().item()==3
    assert dist.min().item()<1e-6  # (0.9 p, tiny sigma) collapses under rank
    counts=allocate_fragments(torch.tensor([0.,1.]),20)
    assert counts.tolist()==[0,20]
    ar=RegionArchive();ar.ensure_root()
    assert AcquisitionScheduler(ar).make_jobs({0:1})[0].snapshot is None


def test_fragment_gae_bootstraps_but_does_not_cross_jobs():
    # Independent one-step windows: bootstrapped M-end vs true termination.
    for terminal,expected in [(False,2.),(True,-1.)]:
        f=object.__new__(RolloutFragment)
        f.rewards=torch.tensor([1.]);f.values=torch.tensor([2.]);f.next_values=torch.tensor([3.])
        f.terminated=torch.tensor([terminal]);f.truncated=torch.tensor([False])
        adv,returns=f.gae(1.,1.)
        assert float(adv)==expected


def test_vector_collector_counts_refills_and_keeps_sources():
    from herp.vector_acquisition import VectorFragmentCollector
    from herp.archive import Snapshot
    class Env:
        num_envs=4;action_dim=2;device=torch.device('cpu')
        def __init__(self):self.x=torch.zeros(4,2);self.t=torch.zeros(4,dtype=torch.long)
        def reset(self):self.x.zero_();self.t.zero_();return self.x.clone(),{}
        def reset_indices(self,ids):self.x[ids]=0;self.t[ids]=0;return self.x.clone(),{}
        def restore_state(self,ids,snapshots):
            for i,s in zip(ids,snapshots):self.x[i]=s;self.t[i]=0
            return self.x[ids].clone()
        def action_low(self):return torch.full((4,2),-1.)
        def action_high(self):return torch.ones(4,2)
        def step(self,action):
            self.x+=1;self.t+=1;done=self.t==2;last=self.x.clone()
            self.x[done]=0;self.t[done]=0
            return self.x.clone(),torch.ones(4),torch.zeros(4,dtype=torch.bool),done,{'final_observation':last}
    class Policy:
        logstd=torch.zeros(1,2)
        def get_distribution(self,o):return torch.distributions.Normal(torch.zeros_like(o),torch.ones_like(o))
        def get_action_and_value(self,o):return torch.zeros_like(o),torch.zeros(len(o)),torch.zeros(len(o)),self.get_value(o)
        def get_value(self,o):return o[:,0]
    archive=RegionArchive();archive.ensure_root();archive.add_region(torch.zeros(4),0)
    archive.add_snapshot(1,Snapshot(torch.ones(2)*7,torch.ones(2)*7,0))
    collector=VectorFragmentCollector(Env(),Policy(),RunningFeatureNormalizer(),4)
    fs=collector.collect_round(archive,torch.tensor([0.,1.]),0,torch.Generator().manual_seed(1),reference_slots=1)
    assert collector.total_steps==16
    assert collector.counters=={'REFERENCE':4,'ROOT_ACQUISITION':0,'REGION_ACQUISITION':12}
    assert len(fs)==8 and all(len(f.rewards)==2 for f in fs)
    for f in fs:
        assert f.source_region_id==(0 if f.slot_id==0 else 1)
        adv,_=f.gae(1.,1.)
        torch.testing.assert_close(adv,torch.tensor([4.,2.]))


def test_restore_save_preserves_contact_observation_memory():
    from herp.envs.maniskill import ManiSkillAdapter
    class Base:
        def __init__(self):self.pose=torch.tensor([[2.]]);self._elapsed_steps=torch.zeros(1,dtype=torch.long)
        def get_state_dict(self):return {'actors':{'pose':self.pose.clone()}}
        def set_state_dict(self,s):self.pose=s['actors']['pose'].clone()
        def get_info(self):return {'is_grasped':torch.tensor([False])}
        def get_obs(self,info=None):return torch.cat([self.pose,info['is_grasped'].float()[:,None]],dim=1)
    adapter=ManiSkillAdapter('unused',device='cpu')
    adapter.env=SimpleNamespace(unwrapped=Base());adapter.num_envs=1;adapter._counter=torch.tensor([7])
    adapter._observation_info={'is_grasped':torch.tensor([True])}
    first=adapter.save_state(torch.tensor([0]));obs=adapter.restore_state(torch.tensor([0]),first)
    second=adapter.save_state(torch.tensor([0]));obs2=adapter.restore_state(torch.tensor([0]),second)
    torch.testing.assert_close(obs,torch.tensor([[2.,1.]]));torch.testing.assert_close(obs2,obs)
    assert bool(second[0].observation_info['is_grasped'][0])
