"""Audit restoration of reached states from an actual training archive."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from train import Args,make_env
from herp.probe import reset_to_snapshot,default_obs_tensor,elapsed_steps

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True);p.add_argument('--states',type=int,default=20)
p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(1)
cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
args=Args(**cp['config']);args.device='cpu';env=make_env(args)
regions=cp['archive'].regions
indices=torch.linspace(0,len(regions)-1,min(a.states,len(regions))).long().tolist()
action=torch.linspace(-.2,.2,env.action_space.shape[-1]).numpy()
rows=[]
for idx in indices:
    snap=regions[idx].snapshots[0]
    restored,_=reset_to_snapshot(env,snap)
    obs_error=float((default_obs_tensor(restored)-snap.obs).abs().max())
    clock=elapsed_steps(env)
    first=default_obs_tensor(env.step(action)[0])
    reset_to_snapshot(env,snap)
    second=default_obs_tensor(env.step(action)[0])
    next_error=float((first-second).abs().max())
    rows.append(dict(region_id=idx,elapsed_steps=clock,expected_elapsed_steps=snap.elapsed_steps,
                     observation_error=obs_error,next_observation_error=next_error))
result=dict(checkpoint=a.checkpoint,env_id=args.env_id,states=rows,
            max_observation_error=max(r['observation_error'] for r in rows),
            max_next_observation_error=max(r['next_observation_error'] for r in rows))
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='states'}))
assert all(r['elapsed_steps']==r['expected_elapsed_steps'] for r in rows)
assert result['max_observation_error']<1e-4,result
assert result['max_next_observation_error']<1e-4,result
env.close()
