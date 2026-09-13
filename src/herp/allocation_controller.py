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
from .sigma import direct_q_estimate
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

    def calibrate(self, observations):
        if self.active:raise RuntimeError('Normalizer already frozen')
        self.normalizer.update(observations)

    def choose(self, method='herp'):
        probs=v3_priority_distribution(self.archive.regions,self.cfg,method)
        rid=0 if len(self.archive)-1<self.cfg.min_non_root_regions else int(torch.multinomial(probs,1,generator=self.generator))
        snapshot=None if rid==0 else self.archive.sample_snapshot(self.archive.regions[rid])
        return rid,snapshot,probs

    def begin_round(self):
        return {r.region_id:region_predictor_features(r) for r in self.archive}

    def finish_round(self, fragments, reference_obs, learner, version, pre_features):
        # Diagnostics must not consume the learner's action/replay RNG stream.
        devices=[learner.device.index or 0] if learner.device.type=='cuda' else []
        groups=defaultdict(list)
        for f in fragments:
            groups[f.region_id].append(f)
        with torch.random.fork_rng(devices=devices):
            gref=learner.gradient_signature({'obs':reference_obs}) if len(reference_obs) else None
            for rid,fs in groups.items():
                if gref is not None:
                    g=learner.gradient_signature({'obs':torch.cat([f.obs for f in fs])})
                    r=self.archive.regions[rid]
                    r.actor_gradient_norm=float(g.norm())
                    r.reference_gradient_norm=float(gref.norm())
                    r.p_raw=float(torch.dot(g,gref)/(g.norm()*gref.norm()+1e-8))
                    r.p_ema=self.cfg.relevance_ema_tau*r.p_ema+(1-self.cfg.relevance_ema_tau)*r.p_raw
                valid=[f for f in fs if f.root_started or rid>0]
                self.cache[rid].extend(valid)
                q,pairs=direct_q_estimate(valid,self.cfg.min_common_steps)
                if rid in pre_features and math.isfinite(q):
                    self.predictor.add_label(pre_features[rid],q,len(valid),version)
        if len(self.predictor.y)>=self.cfg.predictor_min_labels:self.predictor.fit()
        finite=[]
        for r in self.archive:
            fs=[f for f in self.cache[r.region_id] if version-f.policy_version<=self.cfg.max_sigma_policy_lag]
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

    def fragment(self,rid,obs,next_obs,actions,version,root_started):
        z=self.normalizer.normalize(next_obs.cpu())
        low=self.adapter.action_low().cpu().reshape(-1);high=self.adapter.action_high().cpu().reshape(-1)
        an=2*(actions.cpu()-low)/(high-low).clamp_min(1e-8)-1
        feature=torch.cat([z,self.cfg.action_feature_weight*an],-1)
        padded=torch.zeros(self.cfg.future_horizon,feature.shape[-1]);mask=torch.zeros(self.cfg.future_horizon,dtype=torch.bool)
        padded[:len(feature)]=feature;mask[:len(feature)]=True
        return SimpleNamespace(region_id=rid,obs=obs,policy_version=version,root_started=root_started,
                               traj_features=padded,valid_mask=mask)
