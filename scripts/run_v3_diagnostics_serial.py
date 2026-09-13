"""Temporarily reserve the GPU for diagnostics between two campaign jobs."""
import os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs/v3_campaign_20260913'
runner_pid=32991;training_pid=32992
os.kill(runner_pid,signal.SIGSTOP)
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/usr/lib/wsl/lib',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.json',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
try:
 while Path(f'/proc/{training_pid}/stat').exists():
  state=Path(f'/proc/{training_pid}/stat').read_text().split()[2]
  if state=='Z':break
  time.sleep(2)
 checkpoint=out/'seed0_pilot/herp/checkpoint_final.pt'
 if checkpoint.exists():
  jobs=[['scripts/collect_sigma_oracle.py','--checkpoint',str(checkpoint),'--regions','30','--oracle-samples','128','--low-pool','32','--output-dir',str(out/'oracle_validated')],
        ['analysis/sigma_diagnostics.py','--pool',str(out/'oracle_validated/oracle_pool.npz'),'--output-dir',str(out/'oracle_validated/analysis'),'--resamples','100'],
        ['scripts/relevance_diagnostic_v3.py','--checkpoint',str(checkpoint),'--regions','30','--output-dir',str(out/'relevance_validated')]]
  for i,job in enumerate(jobs):
   with open(out/f'diagnostic_serial_{i}.log','w') as log:
    try:result=subprocess.run([sys.executable,'-u']+job,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=420)
    except subprocess.TimeoutExpired:print('Diagnostic timeout',job[0],flush=True);continue
    print(job[0],result.returncode,flush=True)
finally:
 try:os.kill(runner_pid,signal.SIGCONT)
 except ProcessLookupError:pass
