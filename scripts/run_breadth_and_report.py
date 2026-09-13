"""Complete seed-zero breadth pilot after the active longer PPO run."""
import os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs/v3_campaign_20260913'
# The longer PPO run is allowed to finish; do not overlap GPU workers.
while Path('/proc/35718/stat').exists():
 if Path('/proc/35718/stat').read_text().split()[2]=='Z':break
 time.sleep(2)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
try:
 subprocess.run([sys.executable,'-u','scripts/run_v3_campaign.py','--output-dir',str(out/'seed0_breadth'),'--steps','81920','--future-horizon','8','--predictor-min-labels','4','--deadline','1789274100'],cwd=root,env=env)
finally:
 subprocess.run([sys.executable,'analysis/report_v3.py','--campaign',str(out),'--output-dir','docs/v3/report'],cwd=root,env=env)
