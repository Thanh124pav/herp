"""Paper assets: method diagram and clearly scoped synthetic partition checks."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
from herp.config import HERPV3Config
from herp.chain_partition import BoundaryDetector
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':9})
fig,ax=plt.subplots(figsize=(10,2));ax.set(xlim=(0,10),ylim=(0,2));ax.axis('off')
labels=['Online trajectories','Policy-KL chains','Entry-context\nregions','Relevance ×\ndispersion','Acquire → PPO']
for i,label in enumerate(labels):
 x=.1+2*i
 ax.add_patch(FancyBboxPatch((x,.7),1.65,.8,boxstyle='round,pad=.08',facecolor='#EAF4F3',edgecolor='#126B73'))
 ax.text(x+.825,1.1,label,ha='center',va='center')
 if i<4:ax.annotate('',xy=(x+1.9,1.1),xytext=(x+1.72,1.1),arrowprops={'arrowstyle':'->','color':'#126B73'})
ax.text(5,.2,'Root resets and restored chain entries share a counted acquisition budget',ha='center',color='#444444')
for ext in ['pdf','svg']:fig.savefig(out/f'method_diagram.{ext}',bbox_inches='tight')
plt.close(fig)
rows=[]
for scale in [1,10,100,1000]:
 detector=BoundaryDetector(HERPV3Config());count=1
 for t in range(1000):count+=int(detector.observe(0.,float(scale)/1000,t)[0])
 x=np.linspace(0,scale,1000);centroids=[]
 for v in x:
  if not centroids or min(abs(v-c) for c in centroids)>1:centroids.append(v)
 normalized=(x-x.mean())/x.std();nc=[]
 for v in normalized:
  if not nc or min(abs(v-c) for c in nc)>1:nc.append(v)
 rows.append(dict(distance_scale=scale,policy_chain_count=count,raw_state_radius_count=len(centroids),normalized_state_radius_count=len(nc)))
pd.DataFrame(rows).to_csv(out/'synthetic_partition.csv',index=False)
fig,ax=plt.subplots(figsize=(5.4,3.1))
for key,label in [('policy_chain_count','Policy-KL chains'),('raw_state_radius_count','Raw-state radius'),('normalized_state_radius_count','Refitted normalized-state radius')]:
 ax.plot([r['distance_scale'] for r in rows],[r[key] for r in rows],'o-',label=label)
ax.set(xscale='log',yscale='log',xlabel='Synthetic motion scale',ylabel='Number of chains / clusters');ax.legend(fontsize=7);ax.grid(alpha=.2)
ax.set_title('Constant policy: normalization also protects state clustering',fontsize=9)
fig.tight_layout()
for ext in ['pdf','svg']:fig.savefig(out/f'synthetic_partition.{ext}',bbox_inches='tight')
