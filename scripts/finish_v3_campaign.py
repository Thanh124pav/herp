"""Finish corrected HERP and regenerate report after the seed-first runner."""
import os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs/v3_campaign_20260913';pid=32991
while Path(f'/proc/{pid}/stat').exists():
 if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
 time.sleep(3)
env=os.environ.copy();env.update(LD_LIBRARY_PATH='/usr/lib/wsl/lib',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.json',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
try:
 remaining=1789274550-time.time()
 if remaining>120:
  with open(out/'seed0_pilot/herp_corrected.log','w') as log:
   subprocess.run([sys.executable,'-u','scripts/train_v3.py','--method','herp','--seed','0','--num-envs','512','--num-eval-envs','32','--total-timesteps','327680','--eval-episodes','64','--eval-interval','65536','--checkpoint-interval','65536','--output-dir',str(out/'seed0_pilot/herp_corrected')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
finally:
 subprocess.run([sys.executable,'analysis/report_v3.py','--campaign',str(out),'--output-dir','docs/v3/report'],cwd=root,env=env)
