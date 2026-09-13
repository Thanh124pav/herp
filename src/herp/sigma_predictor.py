"""Interpretable weighted ridge with pre-acquisition inputs and no label leakage."""
import math
import torch

class LinearVariancePredictor:
    def __init__(self, feature_dim=5, ridge=1e-3):
        self.feature_dim, self.ridge = feature_dim, ridge
        self.X, self.y, self.weight, self.versions = [], [], [], []
        self.coef = None

    def add_label(self, x, q_hat, weight, policy_version=0):
        if not math.isfinite(q_hat) or weight <= 0:
            return
        self.X.append(torch.as_tensor(x).detach().double().cpu().clone())
        self.y.append(float(q_hat)); self.weight.append(float(weight))
        self.versions.append(int(policy_version))

    def fit(self):
        X = torch.stack(self.X)
        y, w = torch.tensor(self.y,dtype=torch.float64), torch.tensor(self.weight,dtype=torch.float64)
        self.mean = X.mean(0); self.mean[0] = 0
        self.scale = X.std(0,unbiased=False).clamp_min(1e-8); self.scale[0] = 1
        Z = (X-self.mean)/self.scale
        reg = torch.eye(self.feature_dim,dtype=torch.float64)*self.ridge; reg[0,0] = 0
        A, b = Z.T@(w[:,None]*Z)+reg, Z.T@(w*y)
        self.coef = torch.linalg.lstsq(A,b).solution
        return self.coef

    def predict(self,x):
        if self.coef is None:
            return 0.
        z = (torch.as_tensor(x).double().cpu()-self.mean)/self.scale
        return max(0.,float(z@self.coef))

    def raw_coefficients(self):
        coef = self.coef/self.scale
        coef = coef.clone(); coef[0] -= float((self.mean*coef).sum())
        return coef


def region_predictor_features(region):
    return torch.tensor([1.,region.mean_policy_change,region.mean_action_entropy,
                         region.mean_state_change,math.log1p(region.num_entries)],dtype=torch.float64)


def combined_q(q_direct,q_pred,direct_sample_count,kappa=8.):
    q_pred = max(0.,q_pred)
    if not math.isfinite(q_direct):
        return q_pred
    lam = direct_sample_count/(direct_sample_count+kappa)
    return lam*max(0.,q_direct)+(1-lam)*q_pred
