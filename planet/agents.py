import numpy as np
import torch
from planet.planning import Planner
from planet.torch_utils import cuda_if
import heapq

class Agent(torch.nn.Module):
    def __init__(self, obs_shape, a_size, action_repeat=3):
        super(Agent, self).__init__()
        self.obs_shape = obs_shape
        self.obs_depth = obs_shape[0]
        self.a_size = a_size
        self.action_repeat = action_repeat

    def fwd_numpy(self, obs):
        """
        Recieves an ndarray and returns an ndarray

        obs: ndarray
            the observation as a numpy array

        Returns
            ndarray: the action
        """
        a_shape = (len(obs), self.a_size)
        return np.zeros(a_shape)

    def forward(self, obs):
        a_shape = (len(obs), self.a_size)
        return torch.zeros(a_shape)

class DynamicsAgent(Agent):
    def __init__(self, obs_shape, a_size, hyps, dynamics, rew_model):
        super().__init__(obs_shape, a_size, hyps['action_repeat'])
        self.h_size = hyps['h_size']
        self.s_size = hyps['s_size']
        self.horizon = hyps['plan_horizon'] if hyps['plan_horizon'] is not None else hyps['horizon']
        self.n_keep = hyps['n_keep_samples']
        self.n_try = hyps['n_action_samples']
        assert self.n_keep < self.n_try
        self.n_loops = hyps['n_plan_loops']
        self.dynamics = dynamics
        self.rssm = dynamics.rssm
        self.encoder = dynamics.encoder
        self.rew_model = rew_model
        self.planner = Planner(dynamics, rew_model)

    def fwd_numpy(self, obs):
        obs = cuda_if(torch.from_numpy(obs).float(), self.dynamics)
        return self.forward(obs).detach().cpu().numpy()

    def forward(self, obs):
        device = self.dynamics.get_device()
        h = torch.zeros(obs.shape[0], self.h_size, device=device)
        with torch.no_grad():
            action = self.planner.plan(obs, h, self.a_size, self.horizon, n_samples=self.n_try, 
                                            n_iters=self.n_loops, k=self.n_keep, loop=False)
        return action

    def sample_actions(self, h, s, act_mus, act_sigmas):
        rew_sum = cuda_if(torch.zeros(self.n_try), self.dynamics)
        actions = cuda_if(torch.empty(self.horizon, self.n_try, self.a_size), self.dynamics)
        with torch.no_grad():
            for t in range(self.horizon):
                actions[t] = act_mus[t] + act_sigmas[t]*cuda_if(torch.randn(self.n_try, self.a_size), self.dynamics)
                h, s_mu, s_sigma = self.rssm(h, s, actions[t])
                s = s_mu + s_sigma*torch.randn_like(s_sigma)
                h_s_cat = torch.cat([h,s], dim=-1)
                rew_pred = self.rew_model(h_s_cat)
                rew_sum += rew_pred.squeeze()
        top_idxs = self.top_k_args(rew_sum)
        k_best = actions[:,top_idxs]
        rews = rew_sum[top_idxs]
        return k_best, rews

    def top_k_args(self, arr):
        n_keep = self.n_keep
        heap = arr[:n_keep].detach().cpu().tolist()
        valtoidxs = dict()
        for k in range(n_keep):
            if heap[k] not in valtoidxs:
                valtoidxs[heap[k]] = [k]
            else:
                valtoidxs[heap[k]].append(k)
        heapq.heapify(heap)
        for i in range(n_keep, len(arr)):
            val = arr[i].item()
            if val > heap[0]:
                del valtoidxs[heap[0]][0] # Delete first idx for this value
                if val not in valtoidxs:
                    valtoidxs[val] = []
                valtoidxs[val].append(i)
                heapq.heappushpop(heap, val) # Replace smallest value in heap
        top_idxs = []
        for val,idxs in valtoidxs.items():
            top_idxs += idxs
        return top_idxs

class RandnAgent(Agent):
    def __init__(self, obs_shape, a_size, action_repeat=3, means=0, stds=1):
        super().__init__(obs_shape, a_size, action_repeat)
        if type(means) == type(float()) or type(means) == type(int()):
            means = [means]
        if type(stds) == type(float()) or type(stds) == type(int()):
            stds = [stds]
        self.means = torch.FloatTensor(means)
        self.stds = torch.FloatTensor(stds)

    def fwd_numpy(self, obs):
        return self.means.numpy() + np.random.randn(len(obs), self.a_size)*self.stds.numpy()

    def forward(self, obs):
        actions = self.means + torch.randn((len(obs), self.a_size))*self.stds
        if obs.is_cuda:
            return actions.to(obs.get_device())
        return actions

class VRRandnAgent(Agent):
    def fwd_numpy(self, obs):
        velocity = np.random.randn(1)*5 + 10
        direction = np.clip(np.random.randn(1)*25, -49, 49)
        return [float(velocity), float(direction)]

    def forward(self, obs):
        velocity = torch.randn(1)*5 + 10
        direction = torch.clamp(torch.randn(1)*25, -49, 49)
        return [velocity.item(), direction.item()]


