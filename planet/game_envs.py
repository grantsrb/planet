"""
Description:
    - Wrapper class for game environments
    - Mainly used to allow for sequential steps through game connected by server
"""
from collections import deque
import numpy as np
import gym
import torch

class GameEnv:
    def __init__(self, env_name, env_type, prep_fxn, render=False):
        self.env_type = env_type
        self.env = self.get_env(env_name, env_type)
        self.prep_fxn = prep_fxn
        obs_shape, a_size = self.get_game_characteristics()
        self.a_size = a_size
        self.obs_shape = obs_shape
        self.render = render

    def get_game_characteristics(self):
        assert self.env is not None

        if self.env_type == "gym":
            obs = self.reset()
            prepped = self.prep_fxn(obs)
            obs_shape = prepped.shape
            try:
                a_size = self.env.action_space.n
            except AttributeError as e:
                a_size = self.env.action_space.shape[0]
            return obs_shape, a_size
        return None, None

    def get_env(self, env_name, env_type):
        if env_type == "gym":
            return gym.make(env_name)
        else:
            # Create data reading system
            # Also create Unity server api
            assert False

    def step(self, action):
        if self.env_type == "gym":
            obs, rew, done, _ = self.env.step(action)
            if self.render:
                self.env.render()
            obs = self.reshape(obs)
            return obs, rew, done, _
        else:
            # Avoid if data reading system
            # Ping from server if Unity Server system
            assert False

    def reset(self):
        if self.env_type == "gym":
            obs = self.env.reset()
            obs = self.reshape(obs)
            return obs
        else:
            pass

    def reshape(self, obs):
        if len(obs.shape) == 1:
            new_shape = (obs.shape[0],1,1)
            obs = np.reshape(obs, new_shape)
        elif len(obs.shape) == 2:
            obs = obs[...,None]
        obs = np.transpose(obs, (2, 0, 1))
        return obs

    def collect_sode(self, agent):
        """
        agent: Agent object
            obj that takes actions based on observations
        """
        data = {
            "observs":[],
            "rews":[],
            "dones":[],
            "actions":[]
        }
        obs = self.reset()
        done = False
        rew = 0
        obs_depth = agent.obs_depth
        window = deque([np.zeros(self.obs_shape) for i in range(obs_depth-1)], maxlen=obs_depth)
        agent.eval()
        with torch.no_grad():
            while not done:
                prepped = self.prep_fxn(obs)
                window.appendleft(prepped)
                game_obs = np.vstack(window)
                data['observs'].append(game_obs)
                data['rews'].append(rew)
                data['dones'].append(done)
                action = np.squeeze(np.asarray(agent.fwd_numpy(game_obs[None])))[None]
                data['actions'].append(action)
                rew = 0
                for i in range(agent.action_repeat):
                    obs, r, done, _ = self.step(action)
                    rew += r
        agent.train()
        return data
            

    def collect_sodes(self, agent, n_sodes, maxsize=None):
        """
        agent: Agent object
            network to take actions based on observations
        n_sodes: int
            number of episodes to collect
        maxsize: None or int
            sets the maximum size for the data to be collected, 
            if None no limit is set on the datasize
        """
        data = self.collect_sode(agent)
        for ep in range(n_sodes-1):
            new_data = self.collect_sode(agent)
            for k in data.keys():
                data[k] = np.concatenate([data[k], new_data[k]], axis=0)
            if maxsize is not None and len(data['dones']) > maxsize:
                for k in data.keys():
                    data[k] = data[k][-maxsize:]
                break
        return data

