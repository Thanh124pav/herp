"""Launch pinned ManiSkill TD-MPC2 in an isolated Python environment.

Only protocol fixes are applied to a run-local source snapshot; native model,
planner and update rules are unchanged. Use .venvs/tdmpc2/bin/python.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SHA='62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3'


def replace_once(text, old, new):
    if text.count(old)!=1:
        raise RuntimeError(f'Upstream protocol patch mismatch: {old[:60]}')
    return text.replace(old,new,1)


def prepare(destination):
    upstream=ROOT/'third_party/ManiSkill'
    sha=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if sha!=SHA:
        raise RuntimeError(f'Unexpected ManiSkill revision {sha}; review patches before updating lock')
    source=upstream/'examples/baselines/tdmpc2'
    if destination.exists():
        raise FileExistsError(f'Run source already exists: {destination}')
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__'))
    train_path=destination/'train.py'
    train_text=train_path.read_text()
    train_text=replace_once(train_text, 'video_path=video_path, is_eval=True', 'video_path=video_path if cfg.save_video_local else None, is_eval=True')
    train_path.write_text(train_text)
    env_path=destination/'envs/maniskill.py'
    env_text=env_path.read_text()
    env_text=replace_once(env_text, 'render_mode=cfg.render_mode,',
        'render_mode=cfg.render_mode,\n\t\tsim_backend=cfg.env_type, render_backend="cpu" if cfg.env_type=="cpu" else "gpu",')
    env_text=env_text.replace('dummy_env.control_mode','dummy_env.unwrapped.control_mode')
    env_text=env_text.replace('control_mode = env.control_mode','control_mode = env.unwrapped.control_mode')
    env_text=env_text.replace('vector_env_cls = SyncVectorEnv', 'vector_env_cls = partial(SyncVectorEnv, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)')
    env_text=env_text.replace('partial(AsyncVectorEnv, context="forkserver")','partial(AsyncVectorEnv, context="forkserver", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)')
    env_path.write_text(env_text)
    tensor_path=destination/'envs/wrappers/tensor.py'
    tensor_text=tensor_path.read_text().replace('class TensorWrapper(gym.Wrapper):','class TensorWrapper(gym.vector.VectorWrapper):')
    tensor_text=tensor_text.replace('        tensor_dict = {}', '''        if "final_obs" in info:
            info["final_observation"] = np.stack(info.pop("final_obs"))
        if isinstance(info.get("final_info"), dict):
            episode = info["final_info"]["episode"]
            info["final_info"] = [{"episode": {k:v[i] for k,v in episode.items() if not k.startswith("_")}} for i in range(self.num_envs)]
        tensor_dict = {}''')
    tensor_path.write_text(tensor_text)

    path=destination/'trainer/online_trainer.py'
    text=path.read_text()
    text=replace_once(text,'while self._step <= self.cfg.steps:', 'while self._step < self.cfg.steps:')
    text=replace_once(text,'\t\tself.logger.finish(self.agent)', '\t\teval_metrics = self.eval()\n\t\teval_metrics.update(self.common_metrics())\n\t\tself.logger.log(eval_metrics, "eval")\n\t\tself.logger.finish(self.agent)')
    text=replace_once(text,'\t\tfor i in range(self.cfg.eval_episodes_per_env):', '\t\tall_metrics = defaultdict(list)\n\t\tfor i in range(self.cfg.eval_episodes_per_env):')
    text=replace_once(text,'\t\t# Update logger', '\t\t\tfor key, value in self.final_info_metrics(info).items():\n\t\t\t\tall_metrics[key].append(value)\n\t\t# Update logger')
    text=replace_once(text,'eval_metrics.update(self.final_info_metrics(info))','eval_metrics.update({k: float(np.mean(v)) for k,v in all_metrics.items()})')
    path.write_text(text)
    path=destination/'common/logger.py';text=path.read_text()
    marker='\tdef log(self, '
    # Inspect signature rather than depending on CRLF formatting.
    start=text.index(marker); body=text.index('\n',start)+1
    insertion='''\t\tif category == 'eval':
\t\t\timport json
\t\t\trow = dict(method='tdmpc2', family='MBRL', task=self.cfg.env_id,
\t\t\t\tseed=int(self.cfg.seed), env_steps=int(d['step']),
\t\t\t\tsuccess_once=d.get('success_once'), success_at_end=d.get('success_at_end'),
\t\t\t\teval_return=d.get('return'), source='maniskill_tdmpc2', phase=os.environ.get('HERP_PHASE','pilot'),
\t\t\t\teval_episodes=int(self.cfg.num_eval_envs*self.cfg.eval_episodes_per_env),
\t\t\t\tprotocol='native-normalized_dense')
\t\t\twith (self._log_dir/'metrics.jsonl').open('a') as f:
\t\t\t\tf.write(json.dumps(row)+'\\n')
'''
    text=text[:body]+insertion+text[body:];path.write_text(text)
    (destination/'patch_manifest.json').write_text(json.dumps(dict(upstream_sha=sha,
       files={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in destination.rglob('*.py')}),indent=2))
    return destination/'train.py'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--env-id',default='PushCube-v1')
    p.add_argument('--phase',choices=['pilot','ablation','performance','smoke'],default='pilot')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--total-timesteps',type=int,default=100000)
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--eval-interval',type=int,default=10000)
    p.add_argument('--wandb-project',default='herp-framework')
    p.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.total_timesteps<=0 or args.num_envs<=0 or args.total_timesteps%args.num_envs:
        p.error('Positive budget must be divisible by num_envs')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    script=prepare(out/'native_source')
    cmd=[sys.executable,str(script),f'env_id={args.env_id}',f'seed={args.seed}',
         f'steps={args.total_timesteps}',f'num_envs={args.num_envs}','num_eval_envs=1',
         'env_type=cpu',f'eval_episodes_per_env={args.eval_episodes}',f'eval_freq={args.eval_interval}',
         'control_mode=pd_ee_delta_pose',f'wandb={str(args.wandb_mode!="disabled").lower()}',
         f'wandb_project={args.wandb_project}', 'wandb_group=framework-pilot', 'save_video_local=false']
    (out/'command.json').write_text(json.dumps(cmd,indent=2))
    if args.prepare_only:return
    env=os.environ.copy();env['HERP_PHASE']=args.phase;env['WANDB_MODE']=args.wandb_mode if args.wandb_mode!='disabled' else 'offline'
    env.setdefault('VK_ICD_FILENAMES','/usr/share/vulkan/icd.d/lvp_icd.json')
    subprocess.run(cmd,cwd=out,env=env,check=True)

if __name__=='__main__':main()
