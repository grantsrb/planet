from planning import *
from agents import *
from experience import *
import torch
import numpy as np
import unittest

#class TestPlanner(unittest.TestCase):
#    class Dynamics:
#        def __init__(self, means=None, size=5):
#            """
#            means: torch tensor (horizon, batch_size, 1, action_size)
#            """
#            if means is None:
#                self.means = torch.randn(size)
#            else:
#                self.means = means
#        
#        def __call__(self, x):
#            return -((x-self.means.repeat(1,1,x.shape[2],1))**2).mean(-1).mean(0)
#
#    def test_plan(self):
#        horizon, b_size, a_size = 3,2,4
#        means = torch.ones(horizon*b_size*a_size).view(horizon, b_size, 1, a_size).contiguous().float()
#        evaluator = TestPlanner.Dynamics(means)
#        planner = MeanPlanner()
#        init_mean = torch.zeros(horizon, b_size, a_size)
#        plan_means = planner.plan(init_mean, evaluator, n_samples=1000, n_iters=10, k=200)
#        diff = ((plan_means-means.squeeze())**2).mean()
#        print("diff:", diff)
#        self.assertTrue(diff.item() < .01)

class TestMeanPlanner(unittest.TestCase):
    class Evaluator:
        def __init__(self, means=None, size=5):
            """
            means: torch tensor (batch_size, horizon, action_size)
            """
            if means is None:
                self.means = torch.randn(size)
            else:
                self.means = means
        
        def __call__(self, x):
            """
            x: tensor (N, batch, horizon, a_size)
            """
            return -((x-self.means)**2).mean(-1).mean(-1)

    def test_plan(self):
        horizon, b_size, a_size = 3,2,4
        means = torch.ones(b_size, horizon, a_size).float()
        evaluator = TestMeanPlanner.Evaluator(means)
        planner = MeanPlanner()
        init_mean = torch.zeros_like(means)
        plan_means = planner.plan(init_mean, evaluator, n_samples=1000, n_iters=10, k=200)
        diff = ((plan_means-means.squeeze())**2).mean()
        print("diff:", diff)
        self.assertTrue(diff.item() < .01)

class TestAgents(unittest.TestCase):
    def test_randn(self):
        obs = torch.zeros(1000, 40, 50, 50)
        a_size = 5
        agent = RandnAgent(obs.shape, a_size)
        actions = agent(obs)
        self.assertEqual(len(actions), len(obs))
        mean = actions.mean()
        std = actions.std()
        self.assertTrue(mean < .1 and mean > -.1)
        self.assertTrue(std > .9 and std < 1.1)

class TestExperienceReplay(unittest.TestCase):
    def test_init(self):

        exp_replay = ExperienceReplay()
        for k,v in exp_replay.data.items():
            self.assertTrue(v is None)

        n_obs = 10
        data = {
            "observs": np.random.random((n_obs, 40, 50, 50)),
            "rews": np.random.randint(0,2, (n_obs,)),
            "actions": torch.randn(n_obs).numpy(),
            "dones": torch.zeros(n_obs)
        }
        exp_replay = ExperienceReplay(data)
        for k in data.keys():
            for v1, v2 in zip(data[k], exp_replay.data[k]):
                self.assertTrue(np.array_equal(v1, v2))
    
    def test_add_new_data(self):
        # Add initial data
        n_obs = 10
        data = {
            "observs": np.random.random((n_obs, 40, 50, 50)),
            "rews": np.random.randint(0,2, (n_obs,)),
            "actions": torch.randn(n_obs).numpy(),
            "dones": torch.zeros(n_obs)
        }
        exp_replay = ExperienceReplay()
        exp_replay.add_new_data(data)
        for k in data.keys():
            for v1, v2 in zip(data[k], exp_replay.data[k]):
                self.assertTrue(np.array_equal(v1, v2))
        # Add more data
        more_data = {
            "observs": np.random.random((n_obs, 40, 50, 50)),
            "rews": np.random.randint(0,2, (n_obs,)),
            "actions": torch.randn(n_obs).numpy(),
            "dones": torch.zeros(n_obs)
        }
        exp_replay.add_new_data(more_data)
        for k in more_data.keys():
            l = len(more_data[k]) + len(data[k])
            self.assertEqual(len(exp_replay.data[k]), l)

    def test_max_size(self):
        n_obs = 15
        max_size = 10
        data = {
            "observs": np.random.random((n_obs, 40, 50, 50)),
            "rews": np.random.randint(0,2, (n_obs,)),
            "actions": torch.randn(n_obs).numpy(),
            "dones": torch.zeros(n_obs)
        }
        exp_replay = ExperienceReplay(data, max_size=max_size)
        self.assertTrue(len(exp_replay.data['dones']) == max_size)

        more_data = {
            "observs": np.random.random((n_obs, 40, 50, 50)),
            "rews": np.random.randint(0,2, (n_obs,)),
            "actions": torch.randn(n_obs).numpy(),
            "dones": torch.ones(n_obs)
        }
        exp_replay.add_new_data(more_data)
        self.assertTrue(len(exp_replay.data['dones']) == max_size)
        for k in data.keys():
            truth = np.concatenate([data[k], more_data[k]], axis=0)
            self.assertTrue(np.array_equal(exp_replay.data[k], truth[-max_size:]))
    
    def test_sample(self):
        n_obs = 1000
        batch_size = 256
        horizon = 9
        data = {
            "observs": np.random.random((n_obs, 40, 50, 50)),
            "rews": np.random.randint(0,2, (n_obs,)),
            "actions": torch.randn(n_obs).numpy(),
            "dones": torch.zeros(n_obs)
        }
        exp_replay = ExperienceReplay(data)

        sample = exp_replay.sample(batch_size, horizon)
        truth_shape = np.array([batch_size, horizon+1, 40, 50, 50])
        self.assertTrue(np.array_equal(sample['obs_seq'].shape, truth_shape))
        truth_shape = np.array([batch_size, horizon+1])
        for k in sample.keys():
            self.assertTrue(np.array_equal(sample[k].shape[:2], truth_shape))

if __name__=="__main__":
    unittest.main()

