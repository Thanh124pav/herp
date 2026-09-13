"""Build auditable paper artifacts solely from saved run/diagnostic outputs."""
import argparse,json,subprocess,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output-dir',required=True)
a=p.parse_args();root=Path(a.campaign).resolve();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
names={'ppo':'PPO','rnd':'PPO + RND','disagreement':'PPO + Disagreement','uniform':'Uniform Revisit','state_radius_uniform':'State-Radius + Uniform','state_radius_psigma':'State-Radius + pσ','plr_region':'PLR-Region (adapted)','sacl_style':'SACL-style (adapted)','herp_sigma':'HERP-σ','herp_p':'HERP-p','herp':'HERP-pσ'}
rows=[];curves={};diagnostics={}
suite='seed0_breadth' if (root/'seed0_breadth').exists() else 'seed0_pilot'
short=suite=='seed0_breadth'
budget=81920 if short else 327680
horizon=8 if short else 32
min_labels=4 if short else 16
for method,name in names.items():
 d=root/suite/method
 if not short and method=='herp' and (root/'seed0_pilot/herp_corrected').exists():d=root/'seed0_pilot/herp_corrected'
 metrics=[]
 if (d/'metrics.jsonl').exists():
  for line in (d/'metrics.jsonl').read_text().splitlines():
   try:metrics.append(json.loads(line))
   except json.JSONDecodeError:pass
 ev=[r for r in metrics if r.get('type')=='evaluation'];tr=[r for r in metrics if r.get('type')=='training']
 complete=(d/'summary.json').exists();summary=json.loads((d/'summary.json').read_text()) if complete else {}
 last=ev[-1] if ev else {};steps=summary.get('training_steps',tr[-1]['step'] if tr else 0)
 auc=np.trapezoid([r['eval_success'] for r in ev],[r['step'] for r in ev])/ev[-1]['step'] if len(ev)>1 and ev[-1]['step'] else np.nan
 rows.append(dict(method=name,key=method,status='completed' if complete else 'partial' if metrics else 'not run',train_steps=steps,
  evaluation_step=last.get('step',0),success=last.get('eval_success',np.nan),return_mean=last.get('eval_return',np.nan),
  eval_episodes=last.get('eval_episodes',0),success_auc=auc,activation_step=summary.get('activation_step',tr[-1].get('activation_step') if tr else None),
  region_steps=summary.get('budget',tr[-1].get('budget',{}) if tr else {}).get('REGION_ACQUISITION',0),
  wall_seconds=summary.get('wall_seconds',tr[-1].get('wall_seconds',0) if tr else 0)))
 if ev:curves[method]=ev
 if tr:diagnostics[method]=tr
frame=pd.DataFrame(rows);frame.to_csv(out/'results.csv',index=False)
colors=plt.cm.tab20(np.linspace(0,1,len(names)))
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for color,(method,name) in zip(colors,names.items()):
 if method not in curves:continue
 e=curves[method];x=np.array([r['step'] for r in e])/1e3
 for ax,metric in zip(axes,['eval_success','eval_return']):
  ax.plot(x,[r[metric] for r in e],label=name,color=color,lw=2 if method=='herp' else 1.3,marker='o',ms=2.5)
for ax in axes:ax.set_xlabel('Counted training interactions (thousands)');ax.grid(alpha=.2)
axes[0].set_ylabel('Ordinary-reset success rate');axes[0].set_ylim(-.02,1.02);axes[1].set_ylabel('Mean episodic return')
axes[1].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1,1));fig.suptitle('PickCube-v1 · seed 0 · preliminary engineering pilot',fontsize=11)
for ext in ['pdf','svg','png']:fig.savefig(out/f'learning_curves.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,2.8),layout='constrained')
for key,label in [('herp','HERP-pσ'),('uniform','Uniform'),('herp_sigma','HERP-σ'),('herp_p','HERP-p')]:
 if key not in diagnostics:continue
 rs=diagnostics[key];x=np.array([r['step'] for r in rs])/1e3
 for ax,metric in zip(axes,['num_regions','root_fraction','allocation_entropy']):ax.plot(x,[r[metric] for r in rs],label=label)
for ax,y in zip(axes,['Regions (including root)','Root fraction of recorded jobs','Allocation entropy']):
 ax.set(xlabel='Training interactions (thousands)',ylabel=y);ax.grid(alpha=.2)
axes[0].legend(fontsize=7)
for ext in ['pdf','svg']:fig.savefig(out/f'allocation_diagnostics.{ext}',bbox_inches='tight')
plt.close(fig)
# Frozen-policy analysis is included only if actually available.
sigma_dir=root/'oracle_validated'/'analysis';sigma_text='Frozen-policy oracle collection or analysis has not completed.'
relevance_path=root/'relevance_validated'/'summary.json'
relevance_text='The controlled PPO-delta relevance gate has not yet been established.'
if relevance_path.exists():
 r=json.loads(relevance_path.read_text());relevance_text=f"Controlled full-PPO-delta over {r['regions']} regions: Spearman {r['spearman']:.3f}, Pearson {r['pearson']:.3f}, sign accuracy {r['sign_accuracy']:.3f}. This is a frozen-checkpoint mechanism test, not a training-seed result."
 (out/'relevance_summary.json').write_text(json.dumps(r,indent=2))
if (sigma_dir/'ranking_resamples.csv').exists():
 ranks=pd.read_csv(sigma_dir/'ranking_resamples.csv');bv=pd.read_csv(sigma_dir/'bias_variance.csv')
 ranks.groupby(['K','estimator']).mean(numeric_only=True).to_csv(out/'sigma_ranking.csv');bv.to_csv(out/'sigma_bias_variance.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
 for name in ['direct_holdout','ridge_holdout','shrinkage_holdout']:
  r=ranks[ranks.estimator==name].groupby('K').spearman.mean()
  axes[0].plot(r.index,r.values,'o-',label=name.replace('_holdout',''))
  b=bv[bv.estimator==name];axes[1].plot(b.K,b.mse,'o-',label=name.replace('_holdout',''))
 axes[0].set(xlabel='Low-pool sample count K',ylabel='Mean held-out Spearman correlation',ylim=(-1.05,1.05))
 axes[1].set(xlabel='Low-pool sample count K',ylabel='Mean squared error of q');axes[1].set_yscale('symlog',linthresh=.01)
 for ax in axes:ax.legend();ax.grid(alpha=.2)
 for ext in ['pdf','svg']:fig.savefig(out/f'sigma_mechanism.{ext}',bbox_inches='tight')
 plt.close(fig)
 sigma_text='Independent-pool frozen-policy analysis is available for 25 successfully collected regions from the development HERP checkpoint. Collection stopped on a restore-consistency failure in a later region; the atomic saved pool is intact. These are exploratory pre-correction mechanism results. Direct and learned estimators use the same held-out regions. Resampling variability is not training-seed uncertainty.'
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
ncomplete=int((frame.status=='completed').sum());best=frame[frame.status=='completed']
headline=f'{ncomplete}/11 methods completed at generation time. '
if len(best):headline+=f'The highest observed final success rate among completed runs is {best.success.max():.1%}; this is descriptive, not evidence of superiority.'
md=f'''# HERP v3 — local experimental report

Generated: {now}

**Status: single-seed engineering pilot, not a publication-level benchmark.**

{headline}

## Experimental protocol

PickCube-v1; seed 0; {budget:,} counted training interactions per planned method; 512 GPU environments; {horizon}-step acquisition windows; the existing 3×256 PPO network and optimizer; 64 ordinary-reset evaluation episodes. All training root, region and reference transitions are counted. Evaluation and frozen-policy mechanism interactions are reported separately. The 5M-step planning budget in EXPERIMENTS.md was reduced for this time-bounded pilot. No additional training seeds were started.

Root-only data-availability warm-up requires {min_labels} predictor labels, so a substantial fraction of this short GPU pilot can precede activation. The activation step and actual region interactions are included in results.csv. Partial evaluations must not be compared as equal-budget final results.

The shorter breadth protocol is a horizon/warm-up engineering variant, not the default M=32 method. Longer development runs are kept separate and excluded from this equal-budget table.

## Observed results

{frame[['method','status','train_steps','evaluation_step','success','return_mean','region_steps']].to_markdown(index=False,floatfmt='.4f')}

![Learning curves](learning_curves.png)

## Mechanism evidence

{sigma_text}

{relevance_text}

## Implementation and validation

The v3 path implements Gaussian-KL chain boundaries, entry-context clustering, fixed-M dispersion, weighted ridge regression, signed-cosine EMA relevance, root-inclusive pσ allocation, and independent-fragment GAE. The PPO optimizer is reused. `train.py` now dispatches to v3; `--legacy-v2` selects the previous runner. Code provenance for the main pilot is archived in `{suite}/source.tar.gz` and per-run source archives.

ManiSkill GPU required `LD_LIBRARY_PATH=/usr/lib/wsl/lib` on this WSL host. The measured raw throughput at 512 environments was approximately 4,717 transitions/s. Snapshot restoration includes simulator state, controller state, elapsed clocks and contact-derived observation memory. This restores the observation at acquisition entry; it does not claim to serialize every internal PhysX solver cache. An early CPU non-contact replay check measured next-observation error 5.36e-7. Contact-specific replay evidence is insufficient where no qualifying contact snapshots were found.

## Limitations and interpretation

- The non-root region cap of 64 is reached early in the HERP pilot. Capped region-count curves cannot establish absence of region explosion.
- One training seed supports descriptive curves only. No across-seed confidence intervals, aggregate 18-task IQM, or probability-of-improvement claim is made.
- The pilot budget is far below the planned ManiSkill benchmark budgets. Zero success at this budget cannot establish algorithm failure; nonzero differences cannot establish superiority.
- Child-entry total variance is treated as a root proxy. Conditional root means and child-acquisition variance need not define the same root-time distribution. Missing child means use an explicitly logged common prior.
- The predictor and relevance mechanism acceptance gates are not all established. These are engineering pilots, not expensive main-paper experiments that passed every gate in IMPLEMENTATION.md.
- PLR-Region and SACL-style are adaptations. SACL-style uses fixed-state value change with uncertainty coefficient zero. They are not official published implementations.
- RFCL, Active RL, BRO and MaxInfoRL official codebases were downloaded and commit-locked, but their training results are not available. RFCL requires demonstrations; Active RL uses an offline-to-online setting. They are excluded from the same-information table.
- Two early development runs stopped on snapshot observation mismatch. Their trajectories and any partial results are excluded from the main table.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deeplearning
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --method herp --num-envs 512 --num-eval-envs 32 --future-horizon {horizon} --predictor-min-labels {min_labels} --total-timesteps {budget} --eval-episodes 64 --eval-interval 65536 --checkpoint-interval 65536 --output-dir outputs/reproduction_herp_seed0
```

Publication artifacts: learning_curves.pdf/svg, allocation_diagnostics.pdf/svg, results.csv, results_table.tex, report.pdf. Completed oracle analysis additionally produces sigma_mechanism.pdf/svg and numeric tables. Raw logs and checkpoints remain in the campaign output directory.
'''
(out/'REPORT.md').write_text(md)
def tex(s):
 return str(s).replace('σ',r'$\sigma$').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & Steps & Success & Return & Region steps \\',r'\midrule']
for r in rows:
 success='---' if not np.isfinite(r['success']) else f"{r['success']*100:.1f}\\%"
 ret='---' if not np.isfinite(r['return_mean']) else f"{r['return_mean']:.2f}"
 label=tex(r['method'])+(' (partial)' if r['status']=='partial' else ' (not run)' if r['status']=='not run' else '')
 lines.append(f"{label} & {r['train_steps']:,} & {success} & {ret} & {r['region_steps']:,}"+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular}']);(out/'results_table.tex').write_text('\n'.join(lines))
latex=r'''\documentclass[10pt]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{graphicx,booktabs,xcolor,hyperref,amsmath}
\definecolor{teal}{HTML}{126B73}
\hypersetup{colorlinks=true,urlcolor=teal,linkcolor=teal}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
{\color{teal}\LARGE\bfseries HERP v3}\hfill {\small Local experiment report}

{\large Behavior-aware interaction acquisition for robotic PPO}

\textbf{Single-seed engineering pilot --- preliminary evidence}

Generated: '''+tex(now)+r'''

'''+tex(headline)+r'''

\section*{Protocol and scope}
PickCube-v1, seed 0; '''+f'{budget:,}'+r''' training transitions per planned method; 512 GPU environments; fixed continuation horizon $M='''+str(horizon)+r'''$; 64 ordinary-reset evaluation episodes. PPO uses the existing three-layer, 256-unit network and shared optimizer. Root, region and reference transitions all count toward training. Evaluation and mechanism interactions are separate.

The planned 5M interaction budget was shortened to fit the local time window. '''+str(min_labels)+r''' predictor labels are required before activation; the warm-up can dominate a short run. These experiments do not establish the full paper acceptance gates.

\begin{center}\small\input{results_table.tex}\end{center}
\textit{Partial runs are explicitly marked. Their final available evaluation can precede the listed training step; see results.csv. No training-seed confidence intervals are inferred from evaluation episodes.}

\includegraphics[width=\linewidth]{learning_curves.pdf}
\newpage
\section*{Method and acquisition diagnostics}
Behavioral boundaries use symmetric diagonal-Gaussian policy KL. Regions cluster normalized entry contexts, preserving the separation between acquisition source and current chain region. Fixed-window future dispersion estimates
\[
 \widehat q_v=\frac{1}{K_v(K_v-1)}\sum_{i<j}d_M(\xi_i,\xi_j),
 \qquad n_v\propto p_v\sqrt{\widetilde q_v+\epsilon_\sigma^2}.
\]
Complete, independent windows support the unbiased U-statistic interpretation. Every independent fragment has its own GAE boundary; artificial ends bootstrap and true terminations do not.

\includegraphics[width=\linewidth]{allocation_diagnostics.pdf}

\section*{Mechanism validation}
'''+tex(sigma_text+' '+relevance_text)+r'''

\IfFileExists{sigma_mechanism.pdf}{\includegraphics[width=\linewidth]{sigma_mechanism.pdf}}{}

\section*{Reproducibility and limits}
The runtime is Miniconda \texttt{deeplearning}, PyTorch 2.11.0+cu128, ManiSkill 3.0.1 and SAPIEN 3.0.3 on a GTX 1650 with 4 GB VRAM. WSL requires \texttt{LD\_LIBRARY\_PATH=/usr/lib/wsl/lib}. Raw simulator throughput at 512 environments was approximately 4,717 transitions/s; training wall time also includes collection processing and optimization.

Snapshots include controller state, clocks and contact-derived observation history. The non-contact CPU replay test measured next-observation error $5.36\times10^{-7}$. This does not prove exact serialization of contact-solver internals. Early development runs that failed restoration checks are excluded.

Root child-entry total variance is a proxy, not automatically an exact root-time conditional decomposition. Missing child means use a logged prior. PLR-Region and SACL-style are labelled adaptations; the latter uses fixed-state value change without a critic-uncertainty term.

One seed and one task cannot establish statistical superiority or cross-task generalization. Official RFCL, Active RL, BRO and MaxInfoRL repositories were downloaded and commit-locked, but their baseline results are not available. The remaining priority is mechanism acceptance, longer equal-budget runs, task coverage, then additional seeds.
\end{document}
'''
(out/'report.tex').write_text(latex)
result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','report.tex'],cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(out/'latex_build.log').write_text(result.stdout)
print(json.dumps(dict(completed=ncomplete,pdf_built=result.returncode==0,output=str(out))))
