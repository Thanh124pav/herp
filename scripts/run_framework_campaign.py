"""Restartable, serial campaign with explicit pilot/ablation/performance stages.

Planning is the default. --execute runs only the selected stage. Phase 1 requires
an explicitly selected pilot T; inconclusive pilots never unlock comparisons.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from herp.experiment_protocol import balanced_order
ROOT=Path(__file__).resolve().parents[1]


def make_jobs(phase,tasks,seeds,budget,python,td_python,out,num_envs_ppo=512,num_envs_sac_native=16):
    methods=[('PPO','ppo',2017),('SAC','sac',2018),('MBRL','tdmpc2',2024)] if phase=='pilot' else [
        ('PPO','herp',2026),('SAC','sac_herp',2026),('MBRL','tdmpc2',2024),('PPO','ppo',2017),('SAC','sac',2018)]
    variants=[{}]
    if phase=='ablation':
        methods=[('PPO','herp',2026)]
        variants=[dict(sigma_mode=mode,future_horizon=m,sigma_kappa=8.) for mode in ('direct','predictor','shrinkage') for m in (16,32,64)]
        variants += [dict(sigma_mode='shrinkage',future_horizon=32,sigma_kappa=k) for k in (0.,2.,32.)]
    jobs=[]
    for seed in seeds:
        for difficulty,task in enumerate(tasks):
            for family,method,year in methods:
                for variant in variants:
                    suffix=''.join(f'-{k}-{v}' for k,v in variant.items())
                    run=out/phase/family/task/f'{method}{suffix}-seed{seed}'
                    if family=='PPO':
                        # ALL PPO methods (vanilla + HERP + intrinsic + revisit)
                        # run vectorized on GPU. HERP-PPO's VectorPartitionObserver
                        # + VectorFragmentCollector handle parallel restore/observe;
                        # forcing num_envs=1 (as SAC does) would make HERP-PPO
                        # ~500x slower AND change its batch statistics vs vanilla
                        # PPO — breaking the same-backbone causal comparison.
                        cmd=[python,str(ROOT/'scripts/train_v3.py'),'--method',method,'--phase',phase,
                             '--num-envs',str(num_envs_ppo),'--num-eval-envs','16',
                             '--wandb-mode','online','--wandb-project','herp-framework',
                             '--eval-episodes','10','--eval-interval','50000']
                    elif family=='SAC':
                        # Pilot vanilla SAC uses the ManiSkill upstream SAC
                        # (native, num_envs_sac_native=16) purely for a fast
                        # saturation curve. All performance-phase runs go
                        # through the vectorized HERP-SAC runner so vanilla
                        # SAC and HERP-SAC share identical num_envs/replay/
                        # SAC update mechanics — only the acquisition
                        # allocator differs. scripts/train_herp_sac.py (serial
                        # num_envs=1) is kept as a debugging backup.
                        if method=='sac' and phase=='pilot':
                            cmd=[python,str(ROOT/'scripts/sac_official.py'),
                                 '--phase',phase,'--num-envs',str(num_envs_sac_native),
                                 '--num-eval-envs','8','--track','--wandb-project-name','herp-framework',
                                 '--eval-freq','25','--log-freq','10000']
                        else:
                            cmd=[python,str(ROOT/'scripts/train_herp_sac_vector.py'),
                                 '--method',method,'--phase',phase,
                                 '--num-envs',str(num_envs_sac_native),'--num-eval-envs','8',
                                 '--wandb-mode','online','--wandb-project-name','herp-framework',
                                 '--eval-episodes','10','--eval-freq','50000']
                    else:
                        cmd=[td_python,str(ROOT/'scripts/tdmpc2_official.py'),'--phase',phase,'--eval-episodes','10','--eval-interval','50000']
                    # Vector envs require budget divisible by num_envs;
                    # round up to preserve headroom rather than truncate.
                    per_run_budget=budget
                    if family=='PPO':
                        per_run_budget=-(-budget//num_envs_ppo)*num_envs_ppo
                    elif family=='SAC':
                        per_run_budget=-(-budget//num_envs_sac_native)*num_envs_sac_native
                    cmd+=['--env-id',task,'--seed',str(seed),'--total-timesteps',str(per_run_budget),'--output-dir',str(run)]
                    for k,v in variant.items():cmd += ['--'+k.replace('_','-'),str(v)]
                    jobs.append(dict(family=family,method=method,year=year,difficulty=difficulty,task=task,seed=seed,
                                     phase=phase,variant=variant,output_dir=str(run),command=cmd,status='pending'))
    if phase=='ablation':
        # Mechanism controls at the chosen horizon accompany sigma ablations.
        for method in ('uniform','herp_p','herp_sigma','state_radius_uniform','state_radius_psigma','plr_region','sacl_style'):
            for seed in seeds:
                task=tasks[0];run=out/phase/'PPO'/task/f'{method}-seed{seed}'
                cmd=[python,str(ROOT/'scripts/train_v3.py'),'--method',method,'--phase',phase,'--env-id',task,
                     '--seed',str(seed),'--total-timesteps',str(budget),'--output-dir',str(run),
                     '--wandb-mode','online','--wandb-project','herp-framework','--eval-episodes','10']
                jobs.append(dict(family='PPO',method=method,year=2026,difficulty=0,task=task,seed=seed,phase=phase,
                                 output_dir=str(run),command=cmd,status='pending'))
    return balanced_order(jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['pilot','ablation','performance'],required=True)
    p.add_argument('--tasks',nargs='+',default=['PushCube-v1','PickCube-v1','StackCube-v1'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--budget',type=int,default=1000000)
    p.add_argument('--task-budgets',type=Path,help='JSON mapping task names to raw interaction budgets')
    p.add_argument('--selected-t',type=int,help='T chosen after inspecting pilot curves; required beyond pilot')
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--td-python',default=str(ROOT/'.venvs/tdmpc2/bin/python'))
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/framework_campaign')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.budget<=0:p.error('budget must be positive')
    if a.phase!='pilot' and a.selected_t is None and a.execute:p.error('Inspect pilot plateau before selecting T')
    budget=a.selected_t or a.budget
    task_budgets=json.loads(a.task_budgets.read_text()) if a.task_budgets else {}
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    jobs=make_jobs(a.phase,a.tasks,a.seeds,budget,a.python,a.td_python,out)
    for job in jobs:
        if job['task'] in task_budgets:
            amount=int(task_budgets[job['task']])
            if amount<1:p.error('Task budgets must be positive')
            job['command'][job['command'].index('--total-timesteps')+1]=str(amount)
        if job['family']=='SAC':job['command'].append('--cuda')
    manifest=out/f'{a.phase}-jobs.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        old={tuple(j['command']):j for j in previous}
        jobs=[old.get(tuple(j['command']),j) for j in jobs]
    def save():
        temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(jobs,indent=2));temp.replace(manifest)
    save()
    if not a.execute:
        print(f'{len(jobs)} jobs: {manifest}');return
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.json')
    for job in jobs:
        if job['status']=='completed':continue
        run=Path(job['output_dir']);run.mkdir(parents=True,exist_ok=True)
        job.update(status='running',started=time.time());save()
        with (run/'console.log').open('a') as log:
            result=subprocess.run(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        job.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,finished=time.time());save()
        if result.returncode:
            raise SystemExit(f'Failed job: {run}; inspect before continuing')

if __name__=='__main__':main()
