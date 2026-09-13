"""Restartable seed-first local campaign, with a wall-clock cutoff."""
import argparse,json,os,subprocess,sys,time,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--steps',type=int,default=327680);p.add_argument('--deadline',type=float,required=True);p.add_argument('--future-horizon',type=int,default=32);p.add_argument('--predictor-min-labels',type=int,default=16)
a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
methods=['herp','ppo','rnd','disagreement','uniform','state_radius_uniform','state_radius_psigma','plr_region','sacl_style','herp_sigma','herp_p']
manifest=[]
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/usr/lib/wsl/lib',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.json',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with tarfile.open(out/'source.tar.gz','w:gz') as tar:
 for path in ['src/herp','scripts/train_v3.py','scripts/run_v3_campaign.py','train.py','THEORY.md','IMPLEMENTATION.md','EXPERIMENTS.md']:
  tar.add(ROOT/path,arcname=path,filter=lambda x:None if '__pycache__' in x.name else x)
for method in methods:
 run=out/method
 if (run/'summary.json').exists():
  manifest.append(dict(method=method,status='completed',reused=True));continue
 if time.time()>=a.deadline:break
 cmd=[sys.executable,'-u',str(ROOT/'scripts/train_v3.py'),'--method',method,'--future-horizon',str(a.future_horizon),'--predictor-min-labels',str(a.predictor_min_labels),'--seed','0','--num-envs','512','--num-eval-envs','32','--total-timesteps',str(a.steps),'--eval-episodes','64','--eval-interval','65536','--checkpoint-interval','65536','--output-dir',str(run)]
 record=dict(method=method,status='running',started=time.time(),command=cmd);manifest.append(record)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 with open(out/f'{method}.log','w') as log:
  process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  try:code=process.wait(timeout=max(1,a.deadline-time.time()))
  except subprocess.TimeoutExpired:
   process.terminate()
   try:process.wait(timeout=15)
   except subprocess.TimeoutExpired:process.kill();process.wait()
   code=-15
 record.update(status='completed' if code==0 else 'failed' if code!=-15 else 'deadline',exit_code=code,finished=time.time())
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(record),flush=True)
