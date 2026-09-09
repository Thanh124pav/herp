"""Regression checks for likelihoods, episode boundaries, snapshots, and scoring."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from train import Args, Agent, collect_fragment, signature, parse_args, update_region_scores
from herp.archive import RegionArchive, Snapshot
from herp.regions import OnlineRegionizer
from herp.allocator import robust_ranks
from herp.gradient_signature import empirical_fisher_diagonal, signature_parameters
from herp.baselines import RND, Disagreement


class TinyEnv:
    def __init__(self, terminated=False, truncated=False):
        self.action_space=Box(-.1,.1,(1,));self.terminated=terminated;self.truncated=truncated
    def reset(self,seed=None,options=None):return np.array([0.,1.],dtype=np.float32),{}
    def step(self,action):
        assert np.max(np.abs(action))<=.10001
        return np.array([1.,2.],dtype=np.float32),1.,self.terminated,self.truncated,{}


def test_stored_gaussian_likelihood_matches_even_when_applied_action_clips():
    torch.manual_seed(3);agent=Agent(2,1,8)
    batch=collect_fragment(TinyEnv(),agent,Args(device='cpu'),16,0)
    assert (batch['actions'].abs()>.1).any()
    current=agent.get_distribution(batch['obs']).log_prob(batch['actions']).sum(-1)
    torch.testing.assert_close(current,batch['logprobs'])


def test_time_limit_bootstraps_final_state_but_terminal_does_not():
    agent=Agent(2,1,8)
    for p in agent.critic.parameters():p.data.zero_()
    agent.critic[-1].bias.data.fill_(2.)
    args=Args(device='cpu',gamma=.9)
    trunc=collect_fragment(TinyEnv(truncated=True),agent,args,1,0)
    term=collect_fragment(TinyEnv(terminated=True),agent,args,1,0)
    assert trunc['returns'].item()==pytest.approx(2.8)
    assert term['returns'].item()==pytest.approx(1.)


def test_real_signature_is_nonzero_and_changes_with_advantage_sign():
    torch.manual_seed(7);agent=Agent(2,1,8)
    batch=collect_fragment(TinyEnv(),agent,Args(device='cpu'),24,0)
    g=signature(agent,batch,'cpu')
    batch['advantages']=-batch['advantages']
    assert g.norm()>1e-5
    torch.testing.assert_close(signature(agent,batch,'cpu'),-g)


def test_snapshot_is_owned_not_a_mutable_simulator_view():
    archive=RegionArchive();r=archive.add_region(torch.zeros(2),0)
    state={'position':torch.tensor([1.])}
    archive.add_snapshot(r.region_id,Snapshot(state,torch.ones(2),0,0,0.))
    state['position'].zero_()
    assert r.snapshots[0].env_state['position'].item()==1.


def test_rank_normalization_is_scale_invariant_and_respects_ties():
    x=torch.tensor([1.,1.,10.,1000.])
    torch.testing.assert_close(robust_ranks(x),robust_ranks(x*1e-9))
    assert robust_ranks(x)[0]==robust_ranks(x)[1]
    torch.testing.assert_close(robust_ranks(torch.ones(5)),torch.full((5,),.5))


def test_centroids_follow_normalizer_coordinate_changes():
    archive=RegionArchive();rz=OnlineRegionizer(2,archive)
    rz.assign(torch.tensor([1.,2.]),0)
    raw=archive.regions[0].centroid*rz.normalizer.std+rz.normalizer.mean
    rz.assign(torch.tensor([100.,200.]),1)
    restored=archive.regions[0].centroid*rz.normalizer.std+rz.normalizer.mean
    torch.testing.assert_close(raw,restored)


def test_fisher_uses_per_sample_score_squares():
    torch.manual_seed(2);agent=Agent(2,1,8);obs=torch.randn(5,2)
    actions=agent.act(obs)
    actual=empirical_fisher_diagonal(agent,obs,actions)
    rows=[]
    for i in range(5):
        grads=torch.autograd.grad(agent.get_distribution(obs[i:i+1]).log_prob(actions[i:i+1]).sum(),signature_parameters(agent))
        rows.append(torch.cat([g.flatten() for g in grads]).square())
    torch.testing.assert_close(actual,torch.stack(rows).mean(0))


@pytest.mark.parametrize('kind',[RND,Disagreement])
def test_intrinsic_predictors_train_with_frozen_rnd_target(kind):
    torch.manual_seed(1);model=kind(2,1)
    x,y,a=torch.randn(32,2),torch.randn(32,2),torch.randn(32,1)
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith('target.')}
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.003)
    before=float(model.loss(x,a,y).detach())
    for _ in range(30):
        opt.zero_grad();model.loss(x,a,y).backward();opt.step()
    assert float(model.loss(x,a,y).detach())<before
    assert torch.isfinite(model.bonus(x,a,y)).all()
    for k,v in frozen.items():torch.testing.assert_close(model.state_dict()[k],v)


def test_yaml_then_cli_precedence_and_rejection(tmp_path):
    path=tmp_path/'config.yaml';path.write_text('hidden: 32\nmethod: rnd\n')
    args=parse_args(['--config',str(path),'--hidden','64'])
    assert args.hidden==64 and args.method=='rnd'
    path.write_text('unknown_setting: 1\n')
    with pytest.raises(SystemExit):parse_args(['--config',str(path)])


def test_alignment_predicts_first_order_change_on_the_same_reference_loss():
    """Checks the sign and parameter ordering separately from noisy return prediction."""
    import copy
    torch.manual_seed(17)
    agent=Agent(2,1,8)
    ref=dict(obs=torch.randn(128,2),advantages=torch.randn(128))
    reg=dict(obs=torch.randn(96,2),advantages=torch.randn(96))
    for batch in (ref,reg):
        batch['actions']=agent.act(batch['obs'])
    gr=signature(agent,ref,'cpu');gv=signature(agent,reg,'cpu')
    with torch.no_grad():
        old=agent.get_distribution(ref['obs']).log_prob(ref['actions']).sum(-1)
    adv=ref['advantages'];adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    eta=.001;clone=copy.deepcopy(agent)
    with torch.no_grad():
        offset=0
        for param in signature_parameters(clone):
            param.add_(gv[offset:offset+param.numel()].view_as(param),alpha=-eta)
            offset+=param.numel()
        new=clone.get_distribution(ref['obs']).log_prob(ref['actions']).sum(-1)
        change=float(((new-old).exp()*adv).mean()-adv.mean())
    predicted=eta*float(gr@gv)
    assert change==pytest.approx(predicted,abs=1e-6,rel=.05)


@pytest.mark.parametrize('limit',[0,1,2,5,20])
def test_candidate_selection_never_exceeds_probe_budget_capacity(limit):
    archive=RegionArchive(seed=0)
    for i in range(10):archive.add_region(torch.tensor([float(i)]),i)
    chosen=archive.candidates([7,8,9],100,max_candidates=limit)
    assert len(chosen)==min(limit,10)
    assert len({r.region_id for r in chosen})==len(chosen)
    if limit==1:assert chosen[0].region_id==9


@pytest.mark.parametrize('argv',[
    ['--max-snapshots-per-region','0'],['--uniform-mix','1.1'],
])
def test_invalid_archive_and_probability_config_is_rejected(argv):
    with pytest.raises(SystemExit):parse_args(argv)
