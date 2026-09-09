"""Aggregate completed matched-budget runs; never silently include partial runs."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--threshold',type=float,default=.8)
    opt=parser.parse_args();out=Path(opt.output_dir);out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list);curves=defaultdict(list);seen=set();partial=[]
    backbone_configs={};provenance_by_group={}
    for config_path in sorted(Path(opt.root).rglob('config.json')):
        run=config_path.parent
        if not (run/'complete.json').exists():partial.append(str(run));continue
        cfg=json.loads(config_path.read_text());budget=cfg['total_timesteps']
        key=(cfg['env_id'],cfg['method'],budget,cfg['p_estimator'],cfg['sigma_estimator'])
        backbone_key=(cfg['env_id'],budget)
        backbone={k:v for k,v in cfg.items() if k not in ('method','seed','output_dir','p_estimator','sigma_estimator')}
        if backbone_key in backbone_configs and backbone_configs[backbone_key]!=backbone:
            raise ValueError(f'Mismatched backbone/evaluation settings in {run}')
        backbone_configs[backbone_key]=backbone
        provenance=json.loads((run/'provenance.json').read_text())
        # Only enforce that source code and simulator versions match across the group;
        # seed / total_timesteps / device may legitimately vary per run.
        provenance_key={k:provenance[k] for k in ('source_hashes','benchmark','benchmark_version',
                                                  'torch','python','numpy') if k in provenance}
        if backbone_key in provenance_by_group and provenance_by_group[backbone_key]!=provenance_key:
            raise ValueError(f'Implementation changed within comparison group: {run}')
        provenance_by_group[backbone_key]=provenance_key
        unique=(*key,cfg['seed'])
        if unique in seen:raise ValueError(f'Duplicate completed seed: {unique}')
        seen.add(unique)
        rows=read_rows(run/'metrics.csv')
        previous=0
        for row in rows:
            step=int(row['global_env_steps'])
            sources=('normal_steps','probe_steps','allocated_steps','reference_steps')
            if step-previous!=sum(int(row[k]) for k in sources) or step!=sum(int(row['cumulative_'+k]) for k in sources):
                raise ValueError(f'Interaction accounting failed: {run}, {step}')
            for field,value in row.items():
                if value not in ('',None) and field!='allocation_histogram' and not math.isfinite(float(value)):
                    raise ValueError(f'Nonfinite metric: {run}, {field}')
            previous=step
        evaluations=[r for r in rows if r.get('eval_success','')!='']
        complete=json.loads((run/'complete.json').read_text())
        assert complete['global_env_steps']==budget==sum(complete['counts'].values())
        initial=json.loads((run/'initial_eval.json').read_text())
        x=np.array([0]+[int(r['global_env_steps']) for r in evaluations])
        y=np.array([initial['eval_success']]+[float(r['eval_success']) for r in evaluations])
        returns=np.array([initial['eval_return']]+[float(r['eval_return']) for r in evaluations])
        if x[-1]!=budget:raise ValueError(f'Missing final evaluation: {run}')
        auc=float(np.trapezoid(y,x)/budget)
        above=x[y>=opt.threshold]
        groups[key].append(dict(seed=cfg['seed'],success=y[-1],return_=returns[-1],auc=auc,
                                wall_time=complete['wall_time'],threshold_steps=int(above[0]) if len(above) else None,
                                path=str(run)))
        curves[key].append((x,y,returns))
    table=[]
    for key,runs in sorted(groups.items()):
        env,method,budget,p,sigma=key
        row=dict(env_id=env,method=method,budget=budget,p_estimator=p,sigma_estimator=sigma,seeds=len(runs),
                 seed_ids=json.dumps(sorted(r['seed'] for r in runs)),runs=json.dumps([r['path'] for r in runs]))
        for metric in ('success','return_','auc','wall_time'):
            values=np.array([r[metric] for r in runs]);std=float(values.std(ddof=1)) if len(values)>1 else None
            row[metric+'_mean']=float(values.mean());row[metric+'_std']=std
            row[metric+'_ci95_halfwidth']=float(t.ppf(.975,len(values)-1)*std/np.sqrt(len(values))) if std is not None else None
        row['threshold_reached_seeds']=sum(r['threshold_steps'] is not None for r in runs)
        reached=[r['threshold_steps'] for r in runs if r['threshold_steps'] is not None]
        row['threshold_steps_mean_reached_only']=float(np.mean(reached)) if reached else None
        table.append(row)
    if table:
        with (out/'main_table.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        text=['# Measured results','', 'Mean ± sample standard deviation across training seeds. Confidence intervals use a Student t interval; these are imprecise with three seeds. Evaluation success means success at any point in an ordinary-reset episode.','',
              '| Task | Method | Steps/run | Seeds | Success | Return | Success AUC |','|---|---|---:|---:|---:|---:|---:|']
        latex=['Task & Method & Steps & Seeds & Success & Return & AUC \\\\','\\hline']
        for r in table:
            sd=r['success_std'];formatted=f"{r['success_mean']:.3f}"+(f' ± {sd:.3f}' if sd is not None else '')
            text.append(f"| {r['env_id']} | {r['method']} | {r['budget']} | {r['seeds']} | {formatted} | {r['return__mean']:.2f} | {r['auc_mean']:.3f} |")
            latex.append(f"{r['env_id']} & {r['method'].replace('_',chr(92)+'_')} & {r['budget']} & {r['seeds']} & {r['success_mean']:.3f} & {r['return__mean']:.2f} & {r['auc_mean']:.3f} \\\\")
        text+=['',f'Incomplete runs excluded: {len(partial)}. Raw return is reported without invented normalization constants. Zero success at this budget is not evidence that a method can never solve the task.']
        (out/'RESULTS.md').write_text('\n'.join(text)+'\n');(out/'main_table.tex').write_text('\n'.join(latex)+'\n')
        for env,budget in sorted(set((k[0],k[2]) for k in groups)):
            fig,axes=plt.subplots(1,2,figsize=(9,3.4))
            for key,items in sorted(curves.items()):
                if key[0]!=env or key[2]!=budget:continue
                common=items[0][0]
                if any(not np.array_equal(x,common) for x,_,_ in items):raise ValueError('Evaluation grids differ')
                for ax,index in zip(axes,(1,2)):
                    data=np.stack([item[index] for item in items]);mean=data.mean(0)
                    se=data.std(0,ddof=1)/np.sqrt(len(items)) if len(items)>1 else np.zeros_like(mean)
                    ax.plot(common,mean,label=key[1]);ax.fill_between(common,mean-se,mean+se,alpha=.15)
            axes[0].set_ylabel('Success rate');axes[1].set_ylabel('Return')
            for ax in axes:ax.set_xlabel('Training interactions (all sources charged)');ax.legend(fontsize=7)
            fig.suptitle(env+'; shading = ±1 SE across seeds');fig.tight_layout()
            fig.savefig(out/f'{env}_{budget}_learning.pdf');fig.savefig(out/f'{env}_{budget}_learning.png',dpi=180);plt.close(fig)
    (out/'aggregation.json').write_text(json.dumps(dict(completed_runs=len(seen),incomplete_runs=partial,groups=table),indent=2))


if __name__=='__main__':main()
