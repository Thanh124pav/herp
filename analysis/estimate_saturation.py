"""Report provisional saturation T from observed pilot curves; never infer T from budget."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
from scripts.merge_cross_backbone_results import read_run
from herp.experiment_protocol import saturation_candidate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--window',type=int,default=5);p.add_argument('--tolerance',type=float,default=.02)
    p.add_argument('--minimum-steps',type=int,default=50000)
    a=p.parse_args();groups=defaultdict(list)
    for path in a.root.rglob('metrics.jsonl'):
        for r in read_run(path):groups[(r.family,r.method,r.task,r.seed,r.source)].append(r.to_dict())
    a.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for key,rows in groups.items():
        candidate=saturation_candidate(rows,a.window,a.tolerance,a.minimum_steps)
        results.append(dict(family=key[0],method=key[1],task=key[2],seed=key[3],source=key[4],
             candidate_T=candidate,status='extend_to_confirm' if candidate is not None else 'inconclusive',
             last_env_steps=max(r['env_steps'] for r in rows)))
    (a.output_dir/'saturation.json').write_text(json.dumps(dict(window=a.window,tolerance=a.tolerance,
        minimum_steps=a.minimum_steps,runs=results),indent=2))
    if not groups:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for key,rows in groups.items():
        rows=sorted(rows,key=lambda r:r['env_steps'])
        for ax,metric in zip(axes,('success_once','success_at_end','eval_return')):
            valid=[r for r in rows if r[metric] is not None]
            ax.plot([r['env_steps'] for r in valid],[r[metric] for r in valid],label=f'{key[1]} {key[2]} s{key[3]}')
            ax.set(xlabel='Training environment interactions',ylabel=metric)
    axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(a.output_dir/'pilot_curves.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
