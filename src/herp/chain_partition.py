"""Policy-regime boundaries and clustering of entry contexts, never endpoints."""
from collections import deque
from dataclasses import dataclass
import math
import torch
from .chain_features import RunningFeatureNormalizer, symmetric_gaussian_kl, chain_entry_distance
from .archive import Snapshot

@dataclass
class Chain:
    episode_id: int
    start_t: int
    end_t: int
    state_features: torch.Tensor
    actions: torch.Tensor
    action_mean: torch.Tensor
    action_logstd: torch.Tensor
    buffer_indices: torch.Tensor
    entry_snapshot: object = None
    entry_feature: torch.Tensor | None = None

class BoundaryDetector:
    def __init__(self,cfg):
        self.cfg = cfg
        self.channels = RunningFeatureNormalizer()
        self.scores = torch.empty(cfg.boundary_score_buffer,2,dtype=torch.float64)
        self.score_count = 0
        self.score_cursor = 0
        self.threshold = float('inf')

    def observe(self,policy_change,state_change,chain_length):
        raw = torch.tensor([policy_change,state_change],dtype=torch.float64)
        # Compare historical channels in one current scale, avoiding percentile
        # comparisons between z-scores from different normalization versions.
        self.channels.update(raw)
        n=min(self.score_count,len(self.scores))
        if n<16:
            self.scores[self.score_cursor]=raw
            self.score_cursor=(self.score_cursor+1)%len(self.scores);self.score_count+=1
            return False,0.
        weights=torch.tensor([self.cfg.boundary_lambda_policy,self.cfg.boundary_lambda_state],dtype=torch.float64)
        scores = self.channels.normalize(self.scores[:n])@weights
        self.threshold = float(torch.quantile(scores,self.cfg.boundary_percentile))
        score = float(self.channels.normalize(raw)@weights)
        self.scores[self.score_cursor]=raw
        self.score_cursor=(self.score_cursor+1)%len(self.scores);self.score_count+=1
        return score>self.threshold and chain_length>=self.cfg.boundary_min_chain_len,score

class ChainBuilder:
    """Small independently testable temporal builder; artificial ends are censored."""
    def __init__(self,detector):
        self.detector = detector
        self.rows = []
        self.previous = None

    def append(self,z,action,mu,logstd,index,snapshot=None,done=False):
        closed = None
        if self.previous is not None:
            pz,pmu,pls = self.previous
            boundary,_ = self.detector.observe(float(symmetric_gaussian_kl(pmu,pls,mu,logstd)),
                                               float((z-pz).norm()),len(self.rows))
            if boundary and self.rows:
                closed = self.close()
        self.rows.append((z,action,mu,logstd,index,snapshot))
        self.previous = (z,mu,logstd)
        if done:
            final = self.close(); self.previous = None
            return [c for c in (closed,final) if c is not None]
        return [] if closed is None else [closed]

    def close(self):
        if not self.rows:
            return None
        rows,self.rows = self.rows,[]
        z,a,mu,ls,idx,snap = zip(*rows)
        return Chain(0,idx[0],idx[-1],torch.stack(z),torch.stack(a),torch.stack(mu),
                     torch.stack(ls),torch.tensor(idx),snap[0])

class ChainRegionizer:
    def __init__(self,archive,cfg):
        self.archive,self.cfg = archive,cfg
        self.archive.ensure_root()

    def assign_entry(self,entry_feature,snapshot,step,policy_change=0.,entropy=0.,state_change=0.):
        h = entry_feature.detach().cpu().float()
        candidates = self.archive.regions[1:]
        if candidates:
            distances = chain_entry_distance(torch.stack([r.centroid for r in candidates]),h)
            near = int(distances.argmin()); region = candidates[near]
        if not candidates or (float(distances[near])>self.cfg.chain_radius and len(candidates)<self.cfg.max_regions):
            region = self.archive.add_region(h,step)
        else:
            region.centroid.lerp_(h,self.cfg.centroid_tau)
        region.num_entries += 1; region.num_chains += 1
        n = region.num_entries
        for key,value in [('mean_policy_change',policy_change),('mean_action_entropy',entropy),('mean_state_change',state_change)]:
            setattr(region,key,getattr(region,key)+(value-getattr(region,key))/n)
        if snapshot is not None:
            self.archive.add_snapshot(region.region_id,snapshot)
        return region.region_id

    def assign(self,chain,step=0):
        h = chain.entry_feature
        if h is None:
            h = torch.cat([chain.state_features[0],chain.state_features[0]])
        rid = self.assign_entry(h,chain.entry_snapshot,step)
        r = self.archive.regions[rid]
        r.mean_chain_len += (len(chain.actions)-r.mean_chain_len)/r.num_chains
        return rid
