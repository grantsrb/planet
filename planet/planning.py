"""
Description:
    - Handles in game policy contruction
    - Uses prexisting models and plans best route
    - Creates intial action sequences as unit gaussians
    - Iteratively samples, predicts total reward, selects best trajectories, repeats
"""
import numpy as np
import torch

class MeanPlanner:
    """
    Simple testing planner to determine intuition for k,
    """
    def __init__(self):
        pass

    def plan(self, init_mean, evaluator, n_samples=1000, n_iters=7, k=100):
        """
        init_mean: tensor (B, hrzn, A)
        """
        mean = init_mean
        std = torch.ones_like(mean)
        batch_size, horizon, a_size = mean.shape
        for _ in range(n_iters):
            a_sample = mean + std*torch.randn(n_samples, batch_size, horizon, a_size)
            returns = evaluator(a_sample) # Shape (n_samples, batch)
            _, rows = returns.topk(k, dim=0, largest=True, sorted=False) 
            cols = torch.arange(batch_size)[None].repeat(k, 1).long()
            elites = a_sample[rows, cols]
            mean = elites.mean(0)
            std = elites.std(0)
        return mean

class Planner:
    def __init__(self, dynamics, rew_model):
        self.dynamics = dynamics
        self.rew_model = rew_model

    def evaluate(self, hs, mus, sigmas, a_sample, loop=True):
        """
        hs: tensor (B, H)
        mus: tensor (B, S)
        sigmas: tensor (B, S)
        a_sample: tensor (N, B, Hrz, A)

        returns: tensor (N, B)
        """
        n_samples, batch_size, horizon, a_size = a_sample.shape
        with torch.no_grad():
            if loop:
                returns = []
                for actions in a_sample:
                    tup = self.dynamics(None, hs, actions, None, horizon=a_sample.shape[1], 
                                                            overshoot=True, prev_mus=mus, prev_sigmas=sigmas)
                    hs, ss, _, _, _, _, _ = tup
                    hs = torch.cat(hs, dim=0)
                    ss = torch.cat(ss, dim=0)
                    cat = torch.cat([hs,ss], dim=-1)
                    rews = self.rew_model(cat).view(horizon, batch_size)
                    returns.append(rews.sum(0).cpu())
                returns = torch.stack(returns, dim=0)
            else:
                # Join batch and n_samples
                actions = a_sample.view(-1, horizon, a_size).contiguous()
                hs = hs.repeat(n_samples,1) # Shape (n*batch, h_size)
                mus = mus.repeat(n_samples,1) # Shape (n*batch, s_size)
                sigmas = sigmas.repeat(n_samples,1) 
                tup = self.dynamics(None, hs, actions, None, horizon=horizon, overshoot=True, 
                                                                    prev_mu=mus, prev_sigma=sigmas)
                hs, ss, _, _, _, _, _ = tup
                hs = torch.cat(hs, dim=0) # (hrzn+1, n*b, h) -> ((hrzn+1)*n*b,h)
                ss = torch.cat(ss, dim=0)
                cat = torch.cat([hs,ss], dim=-1)
                returns = self.rew_model(cat)
                returns = returns.view(horizon+1, n_samples, batch_size).sum(0)
            return returns

    def plan(self, observs, hs, a_size, horizon, n_samples=1000, n_iters=10, k=100, loop=False):
        """
        observs: tensor (B, C, H, W)
        hs: tensor (B, h_size)
        a_size: int
        horizon: int
        """
        batch_size = observs.shape[0]
        device = self.dynamics.get_device()
        mean = torch.zeros(batch_size, horizon, a_size, device=device)
        std = torch.ones(batch_size, horizon, a_size, device=device)
        mus, sigmas = self.dynamics.encoder(observs, hs)
        for _ in range(n_iters):
            a_sample = mean + std*torch.randn(n_samples, batch_size, horizon, a_size, device=device)
            returns = self.evaluate(hs, mus, sigmas, a_sample, loop=loop) # Shape (n_samples, batch)
            _, rows = returns.topk(k, dim=0, largest=True, sorted=False) # shape (k,batch)
            # Need to index such that we grab the best
            cols = torch.arange(batch_size)[None].repeat(k, 1).long()
            elites = a_sample[rows, cols]
            mean = elites.mean(0)
            std = elites.std(0)
        return mean[:,0]

