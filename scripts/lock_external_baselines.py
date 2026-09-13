"""Record exact downloaded upstream sources without claiming executed baselines."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'rfcl':('RFCL','https://github.com/StoneT2000/rfcl','Demonstrations; native SAC/JAX; ManiSkill overlap'),
 'ActiveRL':('Active RL','https://github.com/sml-iisc/ActiveRL','Offline dataset plus online acquisition; MuJoCo overlap'),
 'BRO':('BRO','https://github.com/naumix/BiggerRegularizedOptimistic','Native off-policy JAX; continuous-control overlap'),
 'MaxInfoRL':('MaxInfoRL','https://github.com/sukhijab/maxinforl_torch','Native off-policy PyTorch; continuous-control overlap')}
rows=[]
for directory,(name,url,setting) in SOURCES.items():
 p=ROOT/'third_party'/directory
 rows.append(dict(name=name,url=url,path=str(p.relative_to(ROOT)),
   commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),
   setting=setting,status='downloaded; not yet executed'))
(ROOT/'docs/v3/external_baselines.lock.json').write_text(json.dumps(rows,indent=2)+'\n')
