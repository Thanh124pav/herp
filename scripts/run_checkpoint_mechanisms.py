"""Run diagnostics as checkpoints from an active local training run become available."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-cell',required=True)
p.add_argument('--output-dir',required=True)
p.add_argument('--timeout-minutes',type=float,default=180)
args=p.parse_args()
root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
for step,kind,regions in [(8192,'sigma',50),(32768,'sigma',50),(32768,'p',30)]:
    out=root/f'{kind}_{step}'
    if (out/'summary.json').exists():continue
    while True:
        choices=list(Path(args.run_cell).glob(f'*/checkpoint_{step}.pt'))
        if choices:break
        if time.monotonic()-start>args.timeout_minutes*60:raise TimeoutError('Training checkpoint did not arrive')
        time.sleep(5)
    if len(choices)!=1:raise ValueError('Ambiguous checkpoint selection')
    print(f'Starting {kind}, checkpoint {step}, {regions} regions',flush=True)
    command=[sys.executable,'analysis/mechanisms.py','--checkpoint',str(choices[0]),'--kind',kind,
             '--regions',str(regions),'--oracle-probes','64','--eval-episodes','50','--output-dir',str(out)]
    with (root/f'{kind}_{step}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Finished {kind}, checkpoint {step}',flush=True)
