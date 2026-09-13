"""Launch existing locked BRO/MaxInfoRL checkouts; never substitute algorithms.

Use the method's own Python environment. DMC success is undefined (N/A); native
returns must not be mislabeled as robotics success probabilities.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def verify(name):
    entries=json.loads((ROOT/'docs/v3/external_baselines.lock.json').read_text())
    entry=next(x for x in entries if x['name']==name)
    repo=ROOT/entry['path']
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if sha!=entry['commit']:raise RuntimeError(f'{name}: revision differs from lock')
    return repo,entry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['BRO','MaxInfoRL'],required=True)
    p.add_argument('--task',default='cartpole-swingup_sparse')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--env-steps',type=int,default=100000)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();repo,entry=verify(a.method);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.method=='BRO':
        source=(repo/'train_parallel.py').read_text()
        source=source.replace("entity='naumix',",'entity=None,').replace("project='BRO',","project='herp-framework',")
        source=source.replace("save_dir = f'./results/{FLAGS.env_name}_RR{str(FLAGS.updates_per_step)}/'",'save_dir = FLAGS.save_dir')
        script=out/'native_train.py';script.write_text(source)
        cmd=[sys.executable,str(script),'--benchmark=dmc',f'--env_name={a.task}',f'--seed={a.seed}',
             '--num_seeds=1',f'--max_steps={a.env_steps}',f'--eval_interval={min(25000,a.env_steps)}',f'--save_dir={out}/',f'--config={repo}/configs/bro_default.py']
    else:
        # Original experiment function accepts budget directly; native CLI's
        # YAML hides it. Action repeat=2 is retained, so raw budget must be even.
        if a.env_steps%2:raise ValueError('Native MaxInfoRL action_repeat=2 requires an even budget')
        script=out/'native_train.py'
        native=(repo/'examples/dmc/experiment.py').read_text()
        native=native.replace('from maxinforl_torch.envs.dm2gym import DMCGym', '''from maxinforl_torch.envs.dm2gym import DMCGym as OriginalDMCGym
    class DMCGym(OriginalDMCGym):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.raw_steps = 0
        def step(self, action):
            self.raw_steps += 1
            return super().step(action)''')
        native=native.replace('    algorithm.learn(', '''    algorithm.learn(''',1)
        marker='\n\ndef main(args):'
        extra='''
    from stable_baselines3.common.evaluation import evaluate_policy
    import json
    returns, lengths = evaluate_policy(algorithm, eval_env, n_eval_episodes=5, deterministic=True, return_episode_rewards=True)
    raw_steps = sum(vec_env.get_attr('raw_steps'))
    row = dict(method='MaxInfoRL', family='SAC', task=domain_name, seed=seed,
        env_steps=int(raw_steps), success_once=None, success_at_end=None,
        eval_return=float(np.mean(returns)), source='locked_MaxInfoRL', phase='pilot',
        eval_episodes=5, protocol='native-dmc-action-repeat-2-action-cost-0')
    with open(logs_dir+'metrics.jsonl','a') as f: f.write(json.dumps(row)+'\\n')
    algorithm.save(logs_dir+'final_model')
    algorithm.save_replay_buffer(logs_dir+'final_replay.pkl')
    run.log({'env_steps':raw_steps,'eval/return':row['eval_return']})
    run.finish()
    vec_env.close();eval_env.close()
'''
        if marker not in native:raise RuntimeError('MaxInfoRL patch anchor missing')
        native=native.replace(marker,extra+marker,1)
        snapshot=out/'experiment_native.py';snapshot.write_text(native)
        script.write_text('import sys\n'+f'sys.path.insert(0,{str(out)!r})\n'+
          'from experiment_native import experiment\n'+f'experiment(alg="maxinfosac",domain_name={a.task!r},logs_dir={str(out)+"/"!r},project_name="herp-framework",total_steps={a.env_steps//2},seed={a.seed},action_cost=0.0)\n')
        cmd=[sys.executable,str(script)]
    (out/'provenance.json').write_text(json.dumps(dict(**entry,requested_raw_env_steps=a.env_steps,
        note='BRO DMC uses one raw DMC step per action per seed; MaxInfoRL counts inner DMC steps explicitly',command=cmd),indent=2))
    if a.prepare_only:return
    env=os.environ.copy();env['PYTHONPATH']=str(repo)+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('WANDB_MODE','online');env.setdefault('MUJOCO_GL','egl')
    subprocess.run(cmd,cwd=repo,env=env,check=True)

if __name__=='__main__':main()
