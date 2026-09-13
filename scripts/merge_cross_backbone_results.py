"""Merge canonical evaluations and native PPO/SAC logs, keeping protocols separate."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from herp.experiment_protocol import EvaluationRecord
from herp.result_tables import render_main_table


def read_run(path):
    config_path = path.parent/'config.json'
    cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
    cfg = cfg.get('args', cfg)
    method = cfg.get('method', 'sac' if 'training_freq' in cfg else 'unknown')
    family = 'SAC' if method.startswith('sac') else 'PPO'
    if cfg.get('phase')=='ablation':
        method += f" [sigma={cfg.get('sigma_mode','shrinkage')}, M={cfg.get('future_horizon',32)}, kappa={cfg.get('sigma_kappa',8.)}]"
    common = dict(method=method, family=family, task=cfg.get('env_id','unknown'),
                  seed=cfg.get('seed',0), source=str(path), phase=cfg.get('phase','unspecified'),
                  protocol=json.dumps({k:cfg.get(k) for k in ('control_mode','reward_mode','eval_episodes','num_eval_steps','num_eval_envs')},sort_keys=True))
    grouped = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if 'success_once' in row and 'family' in row:
            yield EvaluationRecord(**row)
        elif row.get('type') == 'evaluation':
            yield EvaluationRecord(**common, env_steps=row['step'], success_once=row.get('eval_success'),
                success_at_end=row.get('eval_success_final'), eval_return=row.get('eval_return'),
                eval_episodes=row.get('eval_episodes'), eval_env_steps=row.get('eval_steps'))
        elif row.get('tag','').startswith('eval/'):
            grouped[row['env_steps']][row['tag'][5:]] = row['value']
    for step, row in sorted(grouped.items()):
        yield EvaluationRecord(**common, env_steps=step, success_once=row.get('success_once'),
              success_at_end=row.get('success_at_end'), eval_return=row.get('return'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+', type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--budget',type=int,default=1000000)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    paths=sorted({p for root in args.roots for p in root.rglob('metrics.jsonl')})
    rows=[r.to_dict() for path in paths for r in read_run(path)]
    if not rows:
        raise SystemExit('No evaluation records found')
    with (args.output_dir/'evaluations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for phase in ('performance','ablation'):
        (args.output_dir/f'{phase}.html').write_text(render_main_table(rows,args.budget,phase))
    # Never combine different horizons, control modes or seeds implicitly.
    latest={}
    for row in rows:
        key=tuple(row[k] for k in ('family','method','task','seed','phase','protocol','source'))
        if key not in latest or row['env_steps']>latest[key]['env_steps']:
            latest[key]=row
    with (args.output_dir/'final_per_run.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(latest.values())
    lines=['# Results by run','', 'Entries are observed final evaluations, not matched-budget aggregates. Missing metrics are N/A.', '',
           '| Group | Method | Task | Seed | Steps | success_once | success_at_end | Phase |',
           '|---|---|---|---:|---:|---:|---:|---|']
    for r in sorted(latest.values(),key=lambda x:(x['family'],x['method'],x['task'],x['seed'])):
        vals=[r[k] for k in ('family','method','task','seed','env_steps','success_once','success_at_end','phase')]
        lines.append('| '+' | '.join('N/A' if v is None else str(v) for v in vals)+' |')
    (args.output_dir/'results.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
