"""Actual ManiSkill snapshot/replay gate for the batched adapter API."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from herp.envs.maniskill import ManiSkillAdapter
p=argparse.ArgumentParser();p.add_argument('--env-id',default='PickCube-v1');p.add_argument('--output',required=True)
a=p.parse_args();torch.set_num_threads(1)
env=ManiSkillAdapter(a.env_id,sim_backend='physx_cpu',render_backend='cpu',device='cpu').make(1,0)
obs,_=env.reset(seed=43)
torch.manual_seed(4)
for _ in range(7):obs,*_=env.step(torch.randn(1,env.action_dim)*.1)
snap=env.save_state(torch.tensor([0]));clock=int(env.elapsed_steps()[0]);action=torch.randn(1,env.action_dim)*.1
first=env.step(action)
restored=env.restore_state(torch.tensor([0]),snap);restored_clock=int(env.elapsed_steps()[0])
second=env.step(action)
r=dict(env_id=a.env_id,observation_error=float((restored-obs).abs().max()),
 next_observation_error=float((first[0]-second[0]).abs().max()),reward_error=float((first[1]-second[1]).abs().max()),
 terminated_match=torch.equal(first[2],second[2]),truncated_match=torch.equal(first[3],second[3]),clock=clock,restored_clock=restored_clock)
r['passed']=r['observation_error']<1e-4 and r['next_observation_error']<1e-4 and r['reward_error']<1e-4 and r['terminated_match'] and r['truncated_match'] and clock==restored_clock
Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r));env.close()
if not r['passed']:raise SystemExit(1)
