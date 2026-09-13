"""Cross-backbone reporting. Missing observations are never zero successes."""
from dataclasses import dataclass, asdict
import math

FAMILIES = ('PPO', 'SAC', 'MBRL')

@dataclass(frozen=True)
class EvaluationRecord:
    method: str
    family: str
    task: str
    seed: int
    env_steps: int
    success_once: float | None
    success_at_end: float | None
    eval_return: float | None
    source: str
    phase: str = 'performance'
    eval_episodes: int | None = None
    eval_env_steps: int | None = None
    protocol: str = 'unspecified'

    def __post_init__(self):
        if self.family not in FAMILIES or self.env_steps < 0:
            raise ValueError('Invalid family or interaction count')
        for key in ('success_once', 'success_at_end'):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'Invalid {key}: {value}')
        if self.eval_return is not None and not math.isfinite(self.eval_return):
            raise ValueError('Nonfinite return')

    def to_dict(self):
        return asdict(self)


def saturation_candidate(rows, window=5, tolerance=.02, minimum_steps=50000):
    """Provisional T: first sustained plateau in success AND normalized return.

    This is a diagnostic, not proof of convergence. A single-seed plateau must
    be confirmed by extending training; all-zero flat success alone is insufficient.
    """
    rows = sorted(rows, key=lambda r: r['env_steps'])
    for end in range(window, len(rows)+1):
        segment = rows[end-window:end]
        if segment[0]['env_steps'] < minimum_steps:
            continue
        if any(r.get(k) is None for r in segment for k in ('success_once','eval_return')):
            continue
        success = [r['success_once'] for r in segment]
        returns = [r['eval_return'] for r in segment]
        scale = max(1., max(abs(v) for v in returns))
        if max(success) > 0 and max(success)-min(success) <= tolerance and (max(returns)-min(returns))/scale <= tolerance:
            return segment[0]['env_steps']
    return None


def balanced_order(jobs):
    """Seed first; round-robin families; recent methods then easier tasks."""
    result = []
    for seed in sorted({j['seed'] for j in jobs}):
        queues = {f: sorted((j for j in jobs if j['seed']==seed and j['family']==f),
                    key=lambda j: (-j.get('year',0), j.get('difficulty',0), j['method'], j['task'])) for f in FAMILIES}
        while any(queues.values()):
            for family in FAMILIES:
                if queues[family]:
                    result.append(queues[family].pop(0))
    return result
