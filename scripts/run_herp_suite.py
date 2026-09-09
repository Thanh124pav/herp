"""Bounded, restartable, equal-budget six-method ManiSkill experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',default=['PushCube-v1','PickCube-v1'])
    p.add_argument('--methods',nargs='+',default=['ppo','rnd','disagreement','herp_sigma','herp_p','herp'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--steps',type=int,default=32768)
    p.add_argument('--eval-interval',type=int,default=8192)
    p.add_argument('--eval-episodes',type=int,default=50)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--output-dir',default='outputs/herp_suite')
    opt=p.parse_args();root=Path(opt.output_dir);root.mkdir(parents=True,exist_ok=True)
    protocol=vars(opt).copy();protocol.pop('workers')
    path=root/'suite.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Use a new output directory for a changed protocol')
    path.write_text(json.dumps(protocol,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def run(task,method,seed):
        cell=root/f'{task}_{method}_s{seed}';cell.mkdir(exist_ok=True)
        completed=list(cell.glob('*/complete.json'))
        if completed:return dict(task=task,method=method,seed=seed,status='already_complete')
        command=[sys.executable,'train.py','--env-id',task,'--method',method,'--seed',str(seed),
                 '--total-timesteps',str(opt.steps),'--eval-interval',str(opt.eval_interval),
                 '--eval-episodes',str(opt.eval_episodes),'--output-dir',str(cell)]
        with (cell/'console.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        return dict(task=task,method=method,seed=seed,status='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
    jobs=[(task,method,seed) for seed in opt.seeds for task in opt.tasks for method in opt.methods]
    failed=[]
    with ThreadPoolExecutor(max_workers=opt.workers) as pool:
        futures=[pool.submit(run,*job) for job in jobs]
        for future in as_completed(futures):
            status=future.result();print(json.dumps(status),flush=True)
            if status['status']=='failed':failed.append(status)
            subprocess.run([sys.executable,'analysis/aggregate.py','--root',str(root),'--output-dir',str(root/'analysis')],env=env,check=True)
    (root/'suite_complete.json').write_text(json.dumps(dict(failures=failed,finished_at=time.time()),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
