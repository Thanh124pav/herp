"""Controlled full-PPO-delta diagnostic at a frozen GPU checkpoint."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import torch
from scipy.stats import spearmanr,pearsonr
from train import Agent,Args,ppo_update
from scripts.train_v3 import evaluate
from herp.envs.maniskill import ManiSkillAdapter
from herp.vector_acquisition import VectorFragmentCollector
from herp.acquisition import concatenate_batches
from herp.gradient_signature import policy_gradient_signature
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--regions',type=int,default=30)
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(923);np.random.seed(923)
s=torch.load(a.checkpoint,weights_only=False,map_location='cpu');cfg=s['cfg'];out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
env=ManiSkillAdapter(s['args']['env_id'],sim_backend='physx_cuda',render_backend='cpu',device='cuda').make(16,923);env.reset(seed=923)
eval_env=ManiSkillAdapter(s['args']['env_id'],sim_backend='physx_cuda',render_backend='cpu',device='cuda',ignore_terminations=True).make(16,1923)
agent=Agent(env.obs_dim,env.action_dim).cuda();agent.load_state_dict(s['policy']);archive=s['archive']
collector=VectorFragmentCollector(env,agent,s['normalizer'],cfg.future_horizon,cfg.action_feature_weight)
gen=torch.Generator().manual_seed(923);args=Args(num_minibatches=8)
def collect(rid):
 probs=torch.zeros(len(archive),dtype=torch.float64);probs[rid]=1.
 fs=collector.collect_round(archive,probs,s['policy_version'],gen)
 for f in fs:f.gae(args.gamma,args.gae_lambda)
 return {k:v.cuda() for k,v in concatenate_batches(fs).items()}
base=collect(0);scale=float(base['advantages'].std().clamp_min(1e-8))
gref=policy_gradient_signature(agent,base['obs'],base['actions'],base['advantages'],scale)
def clone_update(batch):
 clone=copy.deepcopy(agent);opt=torch.optim.Adam(clone.parameters(),lr=args.learning_rate,eps=1e-5)
 opt.load_state_dict(copy.deepcopy(s['optimizer']))
 np.random.seed(7159);torch.manual_seed(7159)
 stats=ppo_update(clone,opt,batch,args)
 return clone,stats
A,statsA=clone_update(base);evA=evaluate(eval_env,A,16,1923);del A
ids=[r.region_id for r in archive.regions[1:] if r.snapshots];ids=np.random.default_rng(923).choice(ids,min(a.regions,len(ids)),replace=False)
rows=[];started=time.time();eval_steps=evA['eval_steps']
for rid in ids:
 regional=collect(int(rid));g=policy_gradient_signature(agent,regional['obs'],regional['actions'],regional['advantages'],scale)
 cosine=float(torch.dot(g,gref)/(g.norm()*gref.norm()+1e-8))
 combined={k:torch.cat([base[k],regional[k]]) for k in base}
 B,statsB=clone_update(combined);evB=evaluate(eval_env,B,16,1923);del B
 eval_steps+=evB['eval_steps']
 row=dict(region_id=int(rid),cosine=cosine,dot=float(torch.dot(g,gref)),
          delta_return=evB['eval_return']-evA['eval_return'],delta_success=evB['eval_success']-evA['eval_success'],
          base_return=evA['eval_return'],augmented_return=evB['eval_return'],ppo_steps_A=statsA['ppo_minibatches'],ppo_steps_B=statsB['ppo_minibatches'])
 rows.append(row);(out/'regions.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
x=[r['cosine'] for r in rows];y=[r['delta_return'] for r in rows]
summary=dict(regions=len(rows),spearman=float(spearmanr(x,y).statistic),pearson=float(pearsonr(x,y).statistic),
 sign_accuracy=float(np.mean(np.sign(x)==np.sign(y))),diagnostic_acquisition_steps=collector.total_steps,diagnostic_eval_steps=eval_steps,
 wall_seconds=time.time()-started,protocol='A: PPO on shared base; B: PPO on base+region. Same optimizer state, seed, epochs and paired evaluation resets. Full PPO updates; KL early stopping can differ.',
 checkpoint=a.checkpoint)
(out/'summary.json').write_text(json.dumps(summary,indent=2));env.close();eval_env.close()
