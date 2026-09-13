"""Guard the causal comparison against changes to native SAC internals."""
import ast
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_sac_architecture_replay_and_update_match_locked_upstream():
    source=ROOT/'third_party/ManiSkill/examples/baselines/sac/sac.py'
    if not source.exists():pytest.skip('Locked upstream checkout not installed')
    upstream=ast.parse(source.read_text())
    wrapper=ast.parse((ROOT/'scripts/sac_official.py').read_text())
    for name in ('Actor','SoftQNetwork','ReplayBuffer','ReplayBufferSample'):
        a=next(n for n in upstream.body if isinstance(n,ast.ClassDef) and n.name==name)
        b=next(n for n in wrapper.body if isinstance(n,ast.ClassDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b), name
    def update_loop(tree):
        return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='local_update')
    assert ast.dump(update_loop(upstream))==ast.dump(update_loop(wrapper))


def test_tdmpc_protocol_fixes_compile(tmp_path):
    from scripts.tdmpc2_official import prepare
    if not (ROOT/'third_party/ManiSkill/.git').exists():pytest.skip('Missing upstream')
    entry=prepare(tmp_path/'native')
    for path in entry.parent.rglob('*.py'):
        compile(path.read_text(),str(path),'exec')


def test_shrinkage_zero_samples_and_zero_kappa():
    from herp.sigma_predictor import combined_q
    assert combined_q(2.,3.,0,kappa=0)==3.
    assert combined_q(2.,3.,2,kappa=0)==2.
    with pytest.raises(ValueError):combined_q(2.,3.,2,kappa=-1)
