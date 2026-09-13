"""Stable online features and analytic diagonal-Gaussian policy divergence."""
import torch

class RunningFeatureNormalizer:
    def __init__(self, eps=1e-6):
        self.count = 0
        self.mean = None
        self.m2 = None
        self.eps = eps

    def update(self, x):
        x = torch.as_tensor(x).detach().double().cpu()
        if x.ndim == 1:
            x = x[None]
        n = x.shape[0]
        if not n:
            return
        mean, m2 = x.mean(0), ((x-x.mean(0))**2).sum(0)
        if self.mean is None:
            self.count, self.mean, self.m2 = n, mean, m2
            return
        delta = mean-self.mean
        total = self.count+n
        self.m2 += m2+delta.square()*self.count*n/total
        self.mean += delta*n/total
        self.count = total

    def normalize(self, x):
        if self.mean is None:
            return x
        scale = (self.m2/max(self.count, 1)).sqrt().clamp_min(self.eps)
        return (x-self.mean.to(x))/scale.to(x)


def diagonal_gaussian_kl(mu_p, logstd_p, mu_q, logstd_q):
    return (logstd_q-logstd_p + .5*((2*(logstd_p-logstd_q)).exp()
            +(mu_p-mu_q).square()*(-2*logstd_q).exp()-1)).sum(-1)


def symmetric_gaussian_kl(mu_p, logstd_p, mu_q, logstd_q):
    return .5*(diagonal_gaussian_kl(mu_p,logstd_p,mu_q,logstd_q)
               +diagonal_gaussian_kl(mu_q,logstd_q,mu_p,logstd_p))


def chain_entry_distance(h1, h2):
    d = h1.shape[-1]//2
    return .5*((h1[..., :d]-h2[..., :d]).norm(dim=-1)
               +(h1[..., d:]-h2[..., d:]).norm(dim=-1))
