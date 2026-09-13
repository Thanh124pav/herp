import pytest
from herp.experiment_protocol import EvaluationRecord, balanced_order, saturation_candidate
from scripts.merge_cross_backbone_results import read_run
import json


def test_success_validation_and_missing():
    args=dict(method='sac',family='SAC',task='PushCube-v1',seed=0,env_steps=100,
              success_once=None,success_at_end=None,eval_return=None,source='test')
    assert EvaluationRecord(**args).success_once is None
    with pytest.raises(ValueError):
        EvaluationRecord(**{**args,'success_once':float('nan')})


def test_round_robin_before_next_seed():
    jobs=[dict(family=f,method=m,task='easy',seed=s,year=y) for s in (1,0)
          for f,m,y in [('SAC','sac',2018),('PPO','ppo',2017),('SAC','maxinfo',2025),('MBRL','tdmpc2',2024)]]
    out=balanced_order(jobs)
    assert [j['family'] for j in out[:4]]==['PPO','SAC','MBRL','SAC']
    assert out[1]['method']=='maxinfo'
    assert [j['seed'] for j in out]==[0]*4+[1]*4


def test_zero_success_is_not_saturation():
    rows=[dict(env_steps=50000+i*10000,success_once=0.,eval_return=0.) for i in range(8)]
    assert saturation_candidate(rows) is None
    for r in rows:r.update(success_once=.5,eval_return=10.)
    assert saturation_candidate(rows)==50000


def test_merger_does_not_impute_terminal_success(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(training_freq=64,env_id='PushCube-v1',seed=0)))
    path=tmp_path/'metrics.jsonl'
    path.write_text(json.dumps(dict(tag='eval/success_once',value=.5,env_steps=64))+'\n')
    row=list(read_run(path))[0]
    assert row.success_once==.5 and row.success_at_end is None


def test_main_table_keeps_families_and_missing_metrics_separate():
    from herp.result_tables import render_main_table
    r=dict(method='sac',family='SAC',task='PushCube-v1',seed=0,env_steps=100,
           success_once=.5,success_at_end=None,phase='performance',protocol='p')
    html=render_main_table([r],100)
    assert 'rowspan="1">SAC' in html and '0.500 (n=1)' in html and 'N/A' in html
    assert 'No completed evaluations' in render_main_table([r],200)
    with pytest.raises(ValueError):render_main_table([r,r],100)


def test_evaluation_does_not_attribute_reset_success_to_ended_episode():
    import torch
    from scripts.train_v3 import evaluate
    class Env:
        num_envs=1;device='cpu';max_episode_steps=1
        def reset(self,seed=None):return torch.zeros(1,1),{}
        def action_low(self):return torch.tensor([-1.])
        def action_high(self):return torch.tensor([1.])
        def step(self,a):
            return torch.ones(1,1),torch.zeros(1),torch.tensor([False]),torch.tensor([True]),{'success':torch.ones(1),'final_info':{'success':torch.zeros(1)}}
        def success_from_info(self,info):return info['success']
    class Agent:
        def act(self,obs,deterministic=False):return torch.zeros_like(obs)
    row=evaluate(Env(),Agent(),1,0)
    assert row['success_once']==row['success_at_end']==0.
