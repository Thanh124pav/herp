"""Backbone-independent serial acquisition decisions over behavioral regions.

The normalizer is calibrated on counted ordinary interactions and then frozen.
This keeps archived centroids and future labels in one coordinate system.
"""
from collections import defaultdict, deque
from functools import partial
import math
from types import SimpleNamespace
import torch
from .archive import RegionArchive
from .chain_features import RunningFeatureNormalizer
from .v3_observer import PartitionObserver
from .allocator import v3_priority_distribution
from .sigma import apply_terminal_noop_padding, direct_q_estimate, sigma_fragment_eligible
from .sigma_predictor import LinearVariancePredictor, region_predictor_features, combined_q


class AllocationController:
    def __init__(self, adapter, cfg, seed=0, sigma_mode='shrinkage'):
        self.cfg=cfg;self.adapter=adapter;self.sigma_mode=sigma_mode
        self.archive=RegionArchive(cfg.max_snapshots_per_region,seed);self.archive.ensure_root()
        self.normalizer=RunningFeatureNormalizer(eps=1e-3)
        self.observer=PartitionObserver(adapter,self.archive,self.normalizer,cfg)
        self.predictor=LinearVariancePredictor(ridge=cfg.predictor_ridge)
        self.cache=defaultdict(partial(deque,maxlen=cfg.max_sigma_fragments_per_region))
        self.generator=torch.Generator().manual_seed(seed+1234)
        self.active=False
        self.temporal_attempts=[0 for _ in range(max(1, int(getattr(cfg, 'temporal_bins', 3))))]
        self.temporal_selected=[0 for _ in self.temporal_attempts]
        self.temporal_fallback_root=[0 for _ in self.temporal_attempts]
        self.temporal_soft_fallback_root=0
        self.temporal_p_raw=[0. for _ in self.temporal_attempts]
        self.temporal_p_ema=[0. for _ in self.temporal_attempts]
        self.temporal_relevance_count=[0 for _ in self.temporal_attempts]
        self.temporal_last_probs=[1. / len(self.temporal_attempts)
                                  for _ in self.temporal_attempts]

    def calibrate(self, observations):
        if self.active:raise RuntimeError('Normalizer already frozen')
        self.normalizer.update(observations)

    def choose(self, method='herp'):
        probs=v3_priority_distribution(self.archive.regions,self.cfg,method)
        rid=0 if len(self.archive)-1<self.cfg.min_non_root_regions else int(torch.multinomial(probs,1,generator=self.generator))
        if rid == 0:
            return 0,None,probs
        if not getattr(self.cfg, 'temporal_stratification', False):
            return rid,self.archive.sample_snapshot(self.archive.regions[rid]),probs

        phase_probs, phase_candidates = self.temporal_phase_distribution(probs)
        keep=float(phase_probs.sum())
        if keep <= 0 or float(torch.rand((),generator=self.generator)) >= keep:
            self.temporal_soft_fallback_root += 1
            fallback = torch.zeros_like(probs);fallback[0] = 1.
            return 0,None,fallback
        phase = int(torch.multinomial(phase_probs,1,generator=self.generator))
        self.temporal_attempts[phase] += 1
        candidates = list(phase_candidates[phase])
        if not candidates:
            self.temporal_fallback_root[phase] += 1
            fallback = torch.zeros_like(probs);fallback[0] = 1.
            return 0,None,fallback
        weights = probs[candidates]
        pick = int(torch.multinomial(weights / weights.sum(),1,generator=self.generator))
        rid = int(candidates[pick])
        indices = [i for i,s in enumerate(self.archive.regions[rid].snapshots)
                   if self.snapshot_phase(s) == phase]
        snapshot,_ = self.archive.sample_snapshot_from_indices(
            self.archive.regions[rid], indices)
        self.temporal_selected[phase] += 1
        return rid,snapshot,probs

    def temporal_phase_distribution(self, region_probs):
        """Return soft-gated phase probability mass and eligible region IDs.

        The returned probabilities need not sum to one. Their sum is the
        probability of retaining a proposed non-root rollout; the remaining
        mass falls back to root. Mature phases use a sigmoid gate around the
        relevance threshold, so weak phases become rare but never permanently
        lose the probes needed to recover as the policy changes.
        """
        bins=len(self.temporal_attempts)
        candidates=[[] for _ in range(bins)]
        availability=torch.zeros(bins,dtype=torch.float64)
        for region in self.archive.regions[1:]:
            rid=region.region_id
            if rid >= len(region_probs) or region_probs[rid] <= 0:
                continue
            phases={self.snapshot_phase(s) for s in region.snapshots}
            for phase in phases:
                candidates[phase].append(rid)
                availability[phase] += region_probs[rid]
        available=availability > 0
        if not available.any():
            self.temporal_last_probs=[0. for _ in range(bins)]
            return torch.zeros(bins,dtype=torch.float64),candidates
        threshold=float(getattr(self.cfg,'relevance_threshold',0.))
        minimum=int(getattr(self.cfg,'temporal_min_measurements',3))
        mature_scores=[max(0.,p-threshold)
                       for p,n in zip(self.temporal_p_ema,self.temporal_relevance_count)
                       if n >= minimum]
        optimistic=max(mature_scores,default=1.)
        if optimistic <= 0:
            optimistic=1.
        utility=torch.tensor([
            (max(0.,self.temporal_p_ema[i]-threshold)
             if self.temporal_relevance_count[i] >= minimum else optimistic)
            for i in range(bins)],dtype=torch.float64)
        scores=availability*utility
        if scores.sum() <= 0:
            scores=availability.clone()
        adaptive=scores/scores.sum()
        uniform=available.to(torch.float64)/available.sum()
        explore=float(getattr(self.cfg,'temporal_exploration_mix',.15))
        prior=(1.-explore)*adaptive+explore*uniform
        temperature=max(
            float(getattr(self.cfg,'temporal_gate_temperature',.03)),1e-6)
        gates=[]
        for i in range(bins):
            if self.temporal_relevance_count[i] < minimum:
                gates.append(1.)
                continue
            logit=(self.temporal_p_ema[i]-threshold)/temperature
            gates.append(1./(1.+math.exp(-max(-60.,min(60.,logit)))))
        gate=torch.tensor(gates,dtype=torch.float64)
        phase_probs=prior*gate
        self.temporal_last_probs=[float(x) for x in phase_probs]
        return phase_probs,candidates

    def snapshot_phase(self, snapshot):
        bins=max(1,int(getattr(self.cfg,'temporal_bins',3)))
        horizon=max(1,int(getattr(self.adapter,'max_episode_steps',1000)))
        usable=max(1,horizon-int(getattr(self.cfg,'future_horizon',32)))
        elapsed=max(0,min(int(getattr(snapshot,'elapsed_steps',0)),usable))
        return min(bins-1,(elapsed*bins)//(usable+1))

    def temporal_diagnostics(self):
        available=[0 for _ in self.temporal_attempts]
        for region in self.archive.regions[1:]:
            for snapshot in region.snapshots:
                available[self.snapshot_phase(snapshot)] += 1
        return dict(temporal_available=available,
                    temporal_attempts=list(self.temporal_attempts),
                    temporal_selected=list(self.temporal_selected),
                    temporal_fallback_root=list(self.temporal_fallback_root),
                    temporal_soft_fallback_root=self.temporal_soft_fallback_root,
                    temporal_p_raw=list(self.temporal_p_raw),
                    temporal_p_ema=list(self.temporal_p_ema),
                    temporal_relevance_count=list(self.temporal_relevance_count),
                    temporal_phase_probs=list(self.temporal_last_probs))

    def choose_batch(self, method='herp', n=1, warmup=False, ordinary=False):
        """Sample `n` region IDs at once for the vector collector.

        Root fallback preserves the same warmup semantics as `choose()`: if
        we have fewer than `min_non_root_regions` behavioral regions yet, or
        the caller explicitly forces `warmup=True` / `ordinary=True`, every
        slot resets from the initial-state distribution instead of restoring.
        """
        probs = v3_priority_distribution(self.archive.regions, self.cfg, method)
        force_root = ordinary or warmup or (len(self.archive) - 1 < self.cfg.min_non_root_regions)
        if force_root:
            return torch.zeros(n, dtype=torch.long), [None] * n, probs

        rids = torch.multinomial(probs, n, replacement=True, generator=self.generator)
        snapshots = [None] * n
        if not getattr(self.cfg, 'temporal_stratification', False):
            snapshots = [None if int(r) == 0
                         else self.archive.sample_snapshot(self.archive.regions[int(r)])
                         for r in rids]
            return rids, snapshots, probs

        phase_probs, phase_candidates = self.temporal_phase_distribution(probs)
        keep=float(phase_probs.sum())
        for slot in torch.where(rids > 0)[0].tolist():
            if (keep <= 0 or
                    float(torch.rand((),generator=self.generator)) >= keep):
                self.temporal_soft_fallback_root += 1
                rids[slot] = 0
                continue
            phase = int(torch.multinomial(phase_probs, 1, generator=self.generator))
            self.temporal_attempts[phase] += 1
            candidates = phase_candidates[phase]
            if not candidates:
                self.temporal_fallback_root[phase] += 1
                rids[slot] = 0
                continue
            weights = probs[candidates]
            draw = int(torch.multinomial(
                weights / weights.sum(), 1, generator=self.generator))
            rid = int(candidates[draw])
            indices = [i for i,s in enumerate(self.archive.regions[rid].snapshots)
                       if self.snapshot_phase(s) == phase]
            snapshots[slot],_ = self.archive.sample_snapshot_from_indices(
                self.archive.regions[rid], indices)
            rids[slot] = rid
            self.temporal_selected[phase] += 1
        return rids, snapshots, probs

    def begin_round(self):
        return {r.region_id:region_predictor_features(r) for r in self.archive}

    def finish_round(self, fragments, reference_obs, learner, version, pre_features):
        # Diagnostics must not consume the learner's action/replay RNG stream.
        devices=[learner.device.index or 0] if learner.device.type=='cuda' else []
        groups=defaultdict(list)
        for f in fragments:
            groups[f.region_id].append(f)
        with torch.random.fork_rng(devices=devices):
            def signature(obs):
                # Common random numbers substantially reduce variance in the
                # cosine: reference and region gradients see the same SAC
                # reparameterization-noise stream without perturbing training.
                seed = 17011 + int(version)
                torch.manual_seed(seed)
                if learner.device.type == 'cuda':
                    torch.cuda.manual_seed_all(seed)
                return learner.gradient_signature({'obs':obs})

            gref=signature(reference_obs) if len(reference_obs) else None
            def update_relevance(rid, obs):
                if gref is None or len(obs) == 0:
                    return
                g=signature(obs)
                r=self.archive.regions[rid]
                r.actor_gradient_norm=float(g.norm())
                r.reference_gradient_norm=float(gref.norm())
                r.p_raw=float(torch.dot(g,gref)/(g.norm()*gref.norm()+1e-8))
                if r.relevance_count == 0:
                    r.p_ema=r.p_raw
                else:
                    r.p_ema=self.cfg.relevance_ema_tau*r.p_ema+(1-self.cfg.relevance_ema_tau)*r.p_raw
                r.relevance_count += 1
                r.last_scored_step = version

            for rid,fs in groups.items():
                update_relevance(rid,torch.cat([f.obs for f in fs]))
                valid=[f for f in fs if (f.root_started or rid>0)
                       and sigma_fragment_eligible(f,self.cfg.future_horizon)]
                self.cache[rid].extend(valid)
                q,pairs=direct_q_estimate(valid,self.cfg.min_common_steps)
                if rid in pre_features and math.isfinite(q):
                    self.predictor.add_label(pre_features[rid],q,len(valid),version)

            # Measure temporal utility directly from fragments restored in
            # each episode phase. This costs at most one signature per phase
            # represented in a round and does not assume equal phase value.
            phase_obs=defaultdict(list)
            for f in fragments:
                snapshot=getattr(f,'restart_snapshot',None)
                if f.region_id > 0 and snapshot is not None and len(f.obs):
                    phase_obs[self.snapshot_phase(snapshot)].append(f.obs)
            if gref is not None:
                for phase,observations in phase_obs.items():
                    g=signature(torch.cat(observations))
                    p=float(torch.dot(g,gref)/(g.norm()*gref.norm()+1e-8))
                    self.temporal_p_raw[phase]=p
                    n=self.temporal_relevance_count[phase]
                    if n == 0:
                        self.temporal_p_ema[phase]=p
                    else:
                        tau=float(getattr(self.cfg,'relevance_ema_tau',.9))
                        self.temporal_p_ema[phase]=tau*self.temporal_p_ema[phase]+(1.-tau)*p
                    self.temporal_relevance_count[phase]=n+1

            # Cold-start scoring breaks the otherwise circular dependency
            # "a region needs p to be selected, but needs selection to get p".
            # Archived entry observations are sufficient for an actor-gradient
            # alignment probe and consume no simulator interactions.  Once a
            # region has the configured evidence count, normal allocated
            # fragments take over its p updates.
            minimum=int(getattr(self.cfg,'min_relevance_measurements',0))
            if gref is not None and minimum > 0:
                for r in self.archive.regions[1:]:
                    if r.region_id not in groups and r.relevance_count < minimum and r.snapshots:
                        update_relevance(r.region_id,torch.stack([s.obs for s in r.snapshots]))
        if len(self.predictor.y)>=self.cfg.predictor_min_labels:self.predictor.fit()
        finite=[]
        for r in self.archive:
            fs=[f for f in self.cache[r.region_id]
                if version-f.policy_version<=self.cfg.max_sigma_policy_lag
                and sigma_fragment_eligible(f,self.cfg.future_horizon)]
            r.q_direct,_=direct_q_estimate(fs,self.cfg.min_common_steps);r.sigma_sample_count=len(fs)
            if math.isfinite(r.q_direct):finite.append(r.q_direct)
        prior=float(torch.tensor(finite).median()) if finite else 0.
        for r in self.archive:
            r.q_pred=self.predictor.predict(region_predictor_features(r)) if self.predictor.coef is not None else prior
            q=combined_q(r.q_direct,r.q_pred,r.sigma_sample_count,self.cfg.sigma_predictor_kappa)
            if self.sigma_mode=='direct':q=r.q_direct if math.isfinite(r.q_direct) else prior
            elif self.sigma_mode=='predictor':q=r.q_pred
            r.q_combined=q;r.sigma_raw=math.sqrt(q+self.cfg.sigma_floor**2)
        # Root uses its measured ordinary-reset futures (no invented child means).
        return dict(num_regions=len(self.archive),predictor_labels=len(self.predictor.y),
                    root_q=self.archive.regions[0].q_combined)

    def fragment(self,rid,obs,next_obs,actions,version,root_started,*,rewards=None,
                 terminated=None,truncated=None,restart_snapshot=None,category=None,
                 fragment_id=0,successes=None):
        z=self.normalizer.normalize(next_obs.cpu())
        low=self.adapter.action_low().cpu().reshape(-1);high=self.adapter.action_high().cpu().reshape(-1)
        an=2*(actions.cpu()-low)/(high-low).clamp_min(1e-8)-1
        feature=torch.cat([z,self.cfg.action_feature_weight*an],-1)
        padded=torch.zeros(self.cfg.future_horizon,feature.shape[-1]);mask=torch.zeros(self.cfg.future_horizon,dtype=torch.bool)
        padded[:len(feature)]=feature;mask[:len(feature)]=True
        n=len(feature)
        rewards=torch.zeros(n) if rewards is None else rewards.cpu()
        terminated=torch.zeros(n,dtype=torch.bool) if terminated is None else terminated.cpu()
        truncated=torch.zeros(n,dtype=torch.bool) if truncated is None else truncated.cpu()
        fragment=SimpleNamespace(
            region_id=rid,source_region_id=rid,obs=obs,states=obs,
            next_states=next_obs,state_features=z,actions=actions,
            rewards=rewards,terminated=terminated,truncated=truncated,
            policy_version=version,root_started=root_started,traj_features=padded,
            valid_mask=mask,fragment_id=fragment_id,slot_id=0,
            category=category or ("REGION_ACQUISITION" if rid else "ROOT_ACQUISITION"),
            chain_region_id=None,successes=successes,restart_region_id=rid if restart_snapshot else None,
            restart_snapshot_index=None,
            restart_snapshot_timestep=None if restart_snapshot is None else restart_snapshot.timestep,
            restart_snapshot_elapsed_steps=None if restart_snapshot is None else restart_snapshot.elapsed_steps,
            restart_snapshot=restart_snapshot)
        apply_terminal_noop_padding(fragment,self.adapter.obs_dim,self.cfg.future_horizon)
        return fragment
