"""Online training + routing loop (PLAN.md section 15).

``OnlineTrainer`` wires the whole MVP pipeline together:

    parallel SAC  -> segmentation -> experience grouping -> validity filter
    -> competence matrix -> deficit/supply -> router -> routed replay -> updates

It is env-agnostic (synthetic or Meta-World) and router-agnostic (any entry in
``routing.ROUTERS``). Every ``eval_interval`` it records the history needed to
emit the section 28 artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .competence.deficit import deficit_matrix
from .competence.tracker import CompetenceConfig, CompetenceTracker
from .evaluation.budget import BudgetAccountant
from .evaluation.evaluator import summarize_population
from .experience.bank import ExperienceBank
from .experience.segmenter import Segmenter
from .experience.validity import ValidityConfig, ValidityFilter
from .experience.vocabulary import ExperienceVocabulary
from .population.population import Population, default_env_factory
from .population.rollout_manager import RolloutManager
from .routing import make_router
from .routing.base import RoutingContext


@dataclass
class PipelineConfig:
    # population
    population_size: int = 4
    total_env_steps: int = 40_000
    warmup_steps: int = 1_000
    batch_size: int = 128
    route_batch_fraction: float = 0.25
    # backbone strength (PLAN.md section 5 / backbone upgrade)
    hidden: int = 128
    utd: int = 1  # gradient updates per env step
    device: str = "cpu"  # "cuda" for GPU (bigger nets / high UTD)
    obs_norm: bool = False
    lr: float = 3e-4
    # experience / segmentation
    k_seg: int = 6
    median_window: int = 7
    min_chunk_len: int = 4
    eps_experience: float = 1.1  # calibrated from NN descriptor distances (PLAN section 24)
    eps_merge: float | None = None
    alpha_pre: float = 0.5
    alpha_eff: float = 0.5
    min_success_support: int = 3
    success_support_threshold: float = 0.05
    refresh_interval: int = 10_000
    segmenter_fit_after: int = 4_000
    max_experiences: int | None = None  # hard cap on vocabulary size (anti-fragmentation)
    # competence
    beta_alpha: float = 1.0
    beta_beta: float = 1.0
    min_opportunities: int = 5
    horizon: int = 8
    eps_pre: float = 2.0
    eps_effect: float = 1.5
    # routing
    router: str = "uot"
    routing_interval: int = 5_000
    budget_chunks: int = 32
    max_chunks_per_experience: int = 200
    max_routed_experiences: int = 40  # cap the OT problem size (bounded runtime)
    demand_power: float = 1.0
    # evaluation
    eval_interval: int = 10_000
    episodes_per_policy: int = 20
    seed: int = 0
    verbose: bool = True


class OnlineTrainer:
    def __init__(self, config: PipelineConfig | None = None, env_factory=default_env_factory,
                 logger=None):
        self.cfg = config or PipelineConfig()
        self.logger = logger  # optional RunLogger (W&B + local JSONL), PLAN.md section 21
        c = self.cfg
        self.population = Population(
            size=c.population_size,
            env_factory=env_factory,
            route_batch_fraction=c.route_batch_fraction,
            base_seed=c.seed,
            hidden=c.hidden, utd=c.utd, device=c.device, obs_norm=c.obs_norm, lr=c.lr,
        )
        behavior_selector = None
        if c.router == "qmp_style":
            # B5: receiver-Q behavior sharing acts at collection time, not via
            # replay routing (PLAN.md section 17). Each policy still trains its
            # own SAC on its own buffer; only exploration data is QMP-selected.
            from .routing.qmp_style import qmp_behavior_selector
            behavior_selector = qmp_behavior_selector
        self.rollout = RolloutManager(
            self.population, warmup_steps=c.warmup_steps,
            batch_size=c.batch_size, behavior_selector=behavior_selector,
        )
        self.spec = self.population[0].env.spec

        self.segmenter = Segmenter(self.spec, c.k_seg, c.median_window, c.min_chunk_len, seed=c.seed)
        self.vocabulary = ExperienceVocabulary(
            c.eps_experience, c.eps_merge, c.alpha_pre, c.alpha_eff,
            max_experiences=c.max_experiences,
        )
        self.validity = ValidityFilter(
            ValidityConfig(c.min_success_support, c.success_support_threshold)
        )
        self.bank = ExperienceBank(c.max_chunks_per_experience)
        self.competence = CompetenceTracker(
            c.population_size, self.spec, self.segmenter.normalizer,
            CompetenceConfig(c.beta_alpha, c.beta_beta, c.min_opportunities,
                             c.horizon, c.eps_pre, c.eps_effect),
        )
        self.router = make_router(c.router)
        self.budget = BudgetAccountant()
        self._rng = np.random.default_rng(c.seed)

        self._traj_pool: list = []  # for the initial segmenter fit
        self._segmenter_fitted = False
        self.valid_ids: list[int] = []
        self.history = {
            "eval_steps": [], "success": {i: [] for i in range(c.population_size)},
            "return": {i: [] for i in range(c.population_size)},
            "critic_loss": {i: [] for i in range(c.population_size)},
            "actor_loss": {i: [] for i in range(c.population_size)},
            "num_experiences": [], "num_valid_experiences": [],
        }
        self.timing = {"routing_time": 0.0, "eval_time": 0.0}
        self.last_routes = []
        self.last_gamma = None
        self._last_matrix = None  # (ids, C, n_success) snapshot from last real routing round
        # cumulative routing distributions + transfer accounting (PLAN.md section 21)
        self.route_stats = {
            "donor_counts": {}, "receiver_counts": {}, "experience_counts": {},
            "route_count": 0, "chunk_count": 0, "routing_rounds_with_routes": 0,
        }
        self._routing_started = False
        self.transfer = {"positive": 0, "negative": 0, "neutral": 0}

    # -- pipeline stages ---------------------------------------------------
    def _process_trajectory(self, traj) -> None:
        if not self._segmenter_fitted:
            self._traj_pool.append(traj)
            return
        chunks = self.segmenter.segment(traj)
        for chunk in chunks:
            eid = self.vocabulary.assign(chunk)
            self.bank.add(eid, chunk)
            self.validity.observe_chunk(eid, traj.policy_id, traj.episode_id, traj.success)
        # Competence only matters for experiences we may route (the valid ones).
        # Bound the scan to the routable set so runtime stays independent of K.
        ids = [e for e in self.valid_ids if e in self.vocabulary.experiences]
        if len(ids) > self.cfg.max_routed_experiences:
            ids = sorted(ids, key=lambda e: -self.validity.n_success_support(e))
            ids = ids[: self.cfg.max_routed_experiences]
        if ids:
            self.competence.observe(traj, {e: self.vocabulary.experiences[e] for e in ids})

    def _maybe_fit_segmenter(self) -> None:
        if self._segmenter_fitted:
            return
        if self.population.total_env_steps >= self.cfg.segmenter_fit_after and self._traj_pool:
            self.segmenter.fit(self._traj_pool)
            self._segmenter_fitted = True
            if self.cfg.verbose:
                print(f"[fit] segmenter fitted on {len(self._traj_pool)} trajectories")

    def _refresh(self) -> None:
        self.vocabulary.merge_close()
        self.valid_ids = self.validity.update(self.vocabulary)

    def _valid_matrix(self):
        """Competence/supply over the *valid* experiences only.

        Routing operates exclusively on valid experiences (never the full
        vocabulary), and the set is capped for a bounded OT problem. Returns an
        empty matrix when nothing is valid yet -- routing is then skipped.
        """
        ids = [i for i in self.valid_ids if i in self.vocabulary.experiences]
        # cap the routed set (bound the OT problem) by success support
        if len(ids) > self.cfg.max_routed_experiences:
            ids = sorted(ids, key=lambda e: -self.validity.n_success_support(e))
            ids = ids[: self.cfg.max_routed_experiences]
        if not ids:
            N = self.cfg.population_size
            return [], np.zeros((N, 0)), np.zeros((N, 0), dtype=int)
        C, trials, succ = self.competence.matrix(ids)
        n_success_chunks = np.array(
            [[self.bank.n_successful(p, e) for e in ids] for p in range(self.cfg.population_size)]
        )
        return ids, C, n_success_chunks

    def _route(self) -> None:
        # Transition-level baselines (share_all/random/td_priority) route from raw
        # buffers and only need warmup data; experience-level routers (greedy/uot)
        # additionally need a valid competence matrix and guard on it internally.
        if self.population.total_env_steps < self.cfg.warmup_steps:
            return
        if self._segmenter_fitted and self.vocabulary.experiences:
            ids, C, n_success_chunks = self._valid_matrix()
        else:
            N = self.cfg.population_size
            ids, C, n_success_chunks = [], np.zeros((N, 0)), np.zeros((N, 0), dtype=int)
        if C.size:
            self._last_matrix = (ids, C, n_success_chunks)  # snapshot for final artifacts
        t0 = time.time()
        ctx = RoutingContext(
            competence=C, experience_ids=ids, n_success_chunks=n_success_chunks,
            bank=self.bank, vocabulary=self.vocabulary,
            budget_chunks=self.cfg.budget_chunks,
            budget_transitions=self.cfg.budget_chunks * self.cfg.min_chunk_len * 2,
            max_chunks_per_experience=self.cfg.max_chunks_per_experience,
            demand_power=self.cfg.demand_power, rng=self._rng,
        )
        routes = self.router.route(self.population, ctx)
        self.timing["routing_time"] += time.time() - t0
        self.budget.record_routing(routes)
        self.last_routes = routes
        self._accumulate_route_stats(routes)
        if self.logger is not None:
            step = self.population.total_env_steps
            chunks = sum(len(r.chunk_ids) or 1 for r in routes) if routes else 0
            self.logger.log({
                "routing/route_count": len(routes),
                "routing/routed_chunks": chunks,
                "routing/num_valid_experiences": len(self.valid_ids),
                "compute/routing_time_s": self.timing["routing_time"],
            }, step=step)
        if self.cfg.router == "uot" and C.size:
            from .routing.uot import UOTConfig, solve_transport
            try:
                self.last_gamma = solve_transport(C, n_success_chunks, UOTConfig())
            except Exception:
                self.last_gamma = None
        if self.cfg.verbose and routes:
            print(f"[route:{self.cfg.router}] step={self.population.total_env_steps} "
                  f"routes={len(routes)} chunks={sum(len(r.chunk_ids) for r in routes)}")

    def _accumulate_route_stats(self, routes) -> None:
        """Cumulative donor/receiver/experience distributions (PLAN.md section 21)."""
        if not routes:
            return
        self._routing_started = True
        s = self.route_stats
        s["routing_rounds_with_routes"] += 1
        for rt in routes:
            n = max(1, len(rt.chunk_ids))
            s["route_count"] += 1
            s["chunk_count"] += n
            s["donor_counts"][rt.donor_id] = s["donor_counts"].get(rt.donor_id, 0) + n
            s["receiver_counts"][rt.receiver_id] = s["receiver_counts"].get(rt.receiver_id, 0) + n
            s["experience_counts"][rt.experience_id] = (
                s["experience_counts"].get(rt.experience_id, 0) + n
            )

    def _log_eval(self, step: int, results: dict) -> None:
        """Emit the section-21 metric taxonomy for one eval point."""
        data: dict = {}
        for pid, r in results.items():
            data[f"return/policy_{pid}"] = r["mean_return"]
            data[f"success/policy_{pid}"] = r["success_rate"]
            losses = self.population[pid].last_losses
            if "critic_loss" in losses:
                data[f"critic_loss/policy_{pid}"] = losses["critic_loss"]
                data[f"actor_loss/policy_{pid}"] = losses["actor_loss"]
        succ = [r["success_rate"] for r in results.values()]
        data["success/mean"] = float(np.mean(succ))
        data["success/worst"] = float(np.min(succ))
        data["experience/num_experiences"] = len(self.vocabulary)
        data["experience/num_valid_experiences"] = len(self.valid_ids)
        # competence frontier / deficit summary over the valid set
        ids, C, _ = self._valid_matrix()
        if C.size:
            fr = C.max(axis=0)
            data["competence/frontier_mean"] = float(fr.mean())
            data["competence/deficit_mean"] = float(deficit_matrix(C).mean())
        t = self.transfer
        graded = t["positive"] + t["negative"]
        if graded:
            data["routing/positive_transfer_rate"] = t["positive"] / graded
            data["routing/negative_transfer_rate"] = t["negative"] / graded
        data["routing/routed_chunks_total"] = self.route_stats["chunk_count"]
        data["compute/eval_time_s"] = self.timing["eval_time"]
        self.logger.log(data, step=step)

    def _evaluate(self) -> None:
        t0 = time.time()
        results = self.population.evaluate_all(self.cfg.episodes_per_policy)
        self.timing["eval_time"] += time.time() - t0
        step = self.population.total_env_steps
        # positive/negative-transfer accounting: how per-policy success moves once
        # routing is active (PLAN.md section 21). Compared against the previous eval.
        if self._routing_started and self.history["eval_steps"]:
            for pid, r in results.items():
                prev = self.history["success"][pid][-1]
                delta = r["success_rate"] - prev
                if delta > 1e-6:
                    self.transfer["positive"] += 1
                elif delta < -1e-6:
                    self.transfer["negative"] += 1
                else:
                    self.transfer["neutral"] += 1
        self.history["eval_steps"].append(step)
        for pid, r in results.items():
            self.history["success"][pid].append(r["success_rate"])
            self.history["return"][pid].append(r["mean_return"])
            losses = self.population[pid].last_losses
            self.history["critic_loss"][pid].append(float(losses.get("critic_loss", float("nan"))))
            self.history["actor_loss"][pid].append(float(losses.get("actor_loss", float("nan"))))
        self._refresh()
        self.history["num_experiences"].append(len(self.vocabulary))
        self.history["num_valid_experiences"].append(len(self.valid_ids))
        if self.logger is not None:
            self._log_eval(step, results)
        if self.cfg.verbose:
            summ = summarize_population(results)
            print(f"[eval] step={step} mean_success={summ['mean_success']:.2f} "
                  f"worst={summ['worst_success']:.2f} K={len(self.vocabulary)} "
                  f"valid={len(self.valid_ids)}")

    # -- main loop ---------------------------------------------------------
    def train(self) -> dict:
        c = self.cfg
        n = c.population_size
        rounds = c.total_env_steps // n
        refresh_r = max(1, c.refresh_interval // n)
        routing_r = max(1, c.routing_interval // n)
        eval_r = max(1, c.eval_interval // n)

        for r in range(1, rounds + 1):
            finished = self.rollout.step(train=True)
            for traj in finished:
                self._process_trajectory(traj)
            self._maybe_fit_segmenter()
            if r % refresh_r == 0:
                self._refresh()
            if r % routing_r == 0:
                self._route()
            if r % eval_r == 0:
                self.budget.record_population(self.population)
                self._evaluate()

        self.budget.record_population(self.population)
        return self.results()

    def results(self) -> dict:
        ids, C, n_success_chunks = self._valid_matrix()
        if C.size == 0 and self._last_matrix is not None:
            # fall back to the last real routing snapshot for a stable final heatmap
            ids, C, n_success_chunks = self._last_matrix
        final_eval = self.population.evaluate_all(self.cfg.episodes_per_policy)
        t = self.transfer
        graded = t["positive"] + t["negative"]
        transfer = {
            **t,
            "positive_transfer_rate": t["positive"] / graded if graded else 0.0,
            "negative_transfer_rate": t["negative"] / graded if graded else 0.0,
        }
        return {
            "config": self.cfg,
            "history": self.history,
            "competence": C,
            "deficit": deficit_matrix(C) if C.size else C,
            "experience_ids": ids,
            "gamma": self.last_gamma,
            "final_eval": final_eval,
            "population_summary": summarize_population(final_eval),
            "budget": self.budget.summary(),
            "timing": self.timing,
            "last_routes": self.last_routes,
            "route_stats": self.route_stats,
            "transfer": transfer,
        }
