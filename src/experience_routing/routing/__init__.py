"""Routing subpackage: baselines + proposed greedy/UOT routers (PLAN.md section 17)."""

from .base import Router, RoutingContext
from .greedy import GreedyRouter
from .no_share import NoShareRouter
from .qmp_style import QMPStyleRouter
from .random import RandomRouter
from .share_all import ShareAllRouter
from .td_priority import TDPriorityRouter
from .uot import UOTRouter

ROUTERS = {
    cls.name: cls
    for cls in [
        NoShareRouter,
        ShareAllRouter,
        RandomRouter,
        TDPriorityRouter,
        QMPStyleRouter,
        GreedyRouter,
        UOTRouter,
    ]
}


def make_router(name: str) -> Router:
    if name not in ROUTERS:
        raise ValueError(f"unknown router '{name}'. choices: {sorted(ROUTERS)}")
    return ROUTERS[name]()


__all__ = ["Router", "RoutingContext", "ROUTERS", "make_router"]
