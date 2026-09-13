"""Cross-reset contact replay diagnostic using reached archived states."""
import argparse,json,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import torch
from train import Agent
from herp.envs.maniskill import ManiSkillAdapter
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
a=p.parse_args();torch.set_num_threads(1)
s=torch.load(a.checkpoint,weights_only=False,map_location='cpu')
env=ManiSkillAdapter(s['args']['env_id'],sim_backend='physx_cpu',render_backend='cpu',device='cpu').make(1,0)
env.reset(seed=42);agent=Agent(env.obs_dim,env.action_dim);agent.load_state_dict(s['policy'])
rows=[]
# PickCube privileged grasp indicator follows qpos/qvel (9+9 dimensions).
candidates=[snap for r in s['archive'] for snap in r.snapshots if float(snap.obs[18])>.5]
for old in candidates[:10]:
    env.restore_state(torch.tensor([0]),[old.env_state])
    with torch.no_grad():action=agent.act(old.obs[None],True).clamp(env.action_low(),env.action_high())
    obs,*_=env.step(action)
    snap=env.save_state(torch.tensor([0]));clock=int(env.elapsed_steps()[0])
    first=env.step(action)
    env.reset()
    restored=env.restore_state(torch.tensor([0]),snap)
    second=env.step(action)
    rows.append(dict(grasped=float(obs[0,18]),observation_error=float((restored-obs).abs().max()),
      next_observation_error=float((first[0]-second[0]).abs().max()),reward_error=float((first[1]-second[1]).abs().max()),
      termination_equal=torch.equal(first[2],second[2]),clock=clock))
result=dict(checkpoint=a.checkpoint,contact_candidates=len(candidates),cases=rows,
            observation_gate=bool(rows) and max(x['observation_error'] for x in rows)<1e-4,
            one_step_gate=bool(rows) and max(x['next_observation_error'] for x in rows)<1e-4)
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result));env.close()
