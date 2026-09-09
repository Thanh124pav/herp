"""Physical/controller/clock replay check on the installed ManiSkill backend."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import torch
from train import Args,make_env,get_env_state
from herp.archive import Snapshot
from herp.probe import default_obs_tensor,elapsed_steps,reset_to_snapshot

torch.set_num_threads(1)
args=Args();env=make_env(args);obs,_=env.reset(seed=83)
action=torch.zeros(env.action_space.shape).numpy()
for _ in range(7):obs,*_=env.step(action)
snapshot=Snapshot(get_env_state(env),default_obs_tensor(obs),7,0,0.,elapsed_steps(env))
restored,_=reset_to_snapshot(env,snapshot)
observation_error=float((default_obs_tensor(restored)-snapshot.obs).abs().max())
clock=elapsed_steps(env)
first=default_obs_tensor(env.step(action)[0])
reset_to_snapshot(env,snapshot)
second=default_obs_tensor(env.step(action)[0])
next_error=float((first-second).abs().max())
result=dict(observation_max_error=observation_error,next_observation_max_error=next_error,restored_clock=clock)
print(json.dumps(result))
assert observation_error<1e-4 and next_error<1e-4 and clock==7,result
env.close()
