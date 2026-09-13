"""Save a reviewable source/environment snapshot before a run starts."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import time
import tarfile


def save_provenance(output, root):
    output=Path(output);root=Path(root)
    if (output/'provenance.json').exists():
        output=output/f'provenance-resume-{time.time_ns()}'
        output.mkdir()
    files=sorted((root/'src/herp').rglob('*.py'))+sorted((root/'scripts').glob('*.py'))+[root/'train.py',root/'pyproject.toml']
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with tarfile.open(output/'source.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(root)))
    data=dict(python=sys.executable,git_sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
              source_hashes=hashes,packages={d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']})
    (output/'provenance.json').write_text(json.dumps(data,indent=2))
