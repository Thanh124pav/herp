"""Streaming chain-entry archive for the local runner."""
import math
import torch
from .archive import Snapshot
from .chain_features import symmetric_gaussian_kl
from .chain_partition import BoundaryDetector,ChainRegionizer
from .region_graph import RegionGraph

class PartitionObserver:
    def __init__(self,adapter,archive,normalizer,cfg,state_radius=False):
        self.adapter,self.archive,self.normalizer,self.cfg = adapter,archive,normalizer,cfg
        self.detector = BoundaryDetector(cfg)
        self.regionizer = ChainRegionizer(archive,cfg)
        self.graph = RegionGraph()
        self.state_radius = state_radius
        self.previous = None
        self.current_region = None
        self.chain_length = 0
        self.chains = 0
        self.completed_lengths = []
        self.events = []
        self.path_source = 0
        self.is_root_start = True

    def new_fragment(self,source,is_root):
        self._close_chain(censored=True)
        self.previous = None; self.current_region = None
        self.path_source,self.is_root_start = source,is_root

    def _close_chain(self,censored=False):
        if self.current_region is not None and self.chain_length:
            r = self.archive.regions[self.current_region]
            r.mean_chain_len += (self.chain_length-r.mean_chain_len)/max(1,r.num_chains)
            self.completed_lengths.append((self.chain_length,censored))
        self.chain_length = 0

    def end_episode(self):
        self._close_chain()
        self.previous = None; self.current_region = None
        self.path_source,self.is_root_start = 0,True

    def observe(self,obs,mu,logstd,step):
        z = self.normalizer.normalize(obs.detach().cpu())
        mu,logstd = mu.detach().cpu(),logstd.detach().cpu()
        pc=sc=score=0.; boundary=False
        if self.previous is not None:
            pz,pmu,pls = self.previous
            pc = float(symmetric_gaussian_kl(pmu,pls,mu,logstd)); sc=float((z-pz).norm())
            boundary,score = self.detector.observe(pc,sc,self.chain_length)
        else:
            pz = z
        if self.state_radius:
            # Legacy normalized state radius, in the same acquisition pipeline.
            boundary = True
            pz = z
        if boundary or self.current_region is None:
            old = self.current_region
            self._close_chain(censored=False)
            snap = Snapshot(self.adapter.save_state(torch.tensor([0]))[0],obs.detach().cpu().clone(),step,
                            elapsed_steps=int(self.adapter.elapsed_steps()[0]))
            entropy = float((logstd+.5*math.log(2*math.pi*math.e)).sum())
            self.current_region = self.regionizer.assign_entry(torch.cat([pz,z]),snap,step,pc,entropy,sc)
            self.graph.observe_path([self.path_source if old is None else old,self.current_region])
            self.chains += 1
            self.events.append(dict(step=step,region=self.current_region,policy_change=pc,state_change=sc,
                                    score=score,threshold=self.detector.threshold,elapsed=snap.elapsed_steps,
                                    fragment_start=old is None))
        self.chain_length += 1
        self.previous = (z,mu,logstd)
        return self.current_region
