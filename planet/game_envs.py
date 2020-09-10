"""
Description:
    - Wrapper class for game environments
    - Mainly used to allow for sequential steps through game connected by server
"""
from collections import deque
import numpy as np
import gym
import torch
from planet.utils import discount
from mlagents_envs.environment import UnityEnvironment                     
from gym_unity.envs import UnityToGymWrapper                               
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import time

class GameEnv:
    def __init__(self, env_name, env_type, prep_fxn, rew_discount=0,
                                                     seed=None,
                                                     action_repeat=1):
        self.seed = seed if seed is not None else time.time()
        self.env_type = env_type
        self.env_name = env_name
        self.env = self.get_env(env_name, env_type)
        self.prep_fxn = prep_fxn
        obs_shape, a_size = self.get_game_characteristics()
        self.a_size = a_size
        self.obs_shape = obs_shape
        self.rew_discount = rew_discount
        self.action_repeat = action_repeat

    def get_game_characteristics(self):
        assert self.env is not None

        if self.env_type == "gym" or self.env_type == "unity":
            obs = self.reset()
            prepped = self.prep_fxn(obs)
            obs_shape = prepped.shape
            if self.env_name == "Pong-v0":
                a_size = 3
            elif hasattr(self.env.action_space, "n"):
                a_size = self.env.action_space.n
            elif self.env_type == "unity":
                a_size = 5
                print("hard coded action size! try to change this!")
            else:
                a_size = self.env.action_space.shape[0]
            return obs_shape, a_size
        return None, None

    def get_env(self, env_name, env_type):
        if env_type == "gym":
            return gym.make(env_name)
        elif env_type == "mlagents":
            env = UnityEnvironment(file_name=env_name,
                                   seed=self.seed)
            unity_env = UnityEnvironment(file_name=game_path)
            env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
            return env
        else:
            # Create data reading system
            # Also create Unity server api
            assert False

    def step(self, action, render=False):
        if self.env_type == "gym":
            if hasattr(self.env.action_space, "n"): # Discrete action space
                action = np.argmax(action).squeeze()
            if self.env_name == "Pong-v0":
                action += 1
            obs, rew, done, _ = self.env.step(action)
            for i in range(self.action_repeat-1):
                obs,r,d,_ = self.env.step(action)
                rew += r
                done = d+done > 0
            if render:
                self.env.render()
            obs = self.reshape(obs)
            return obs, rew, done, _
        else:
            # Avoid if data reading system
            # Ping from server if Unity Server system
            assert False

    def reset(self):
        if self.env_type == "gym" or self.env_type == "unity":
            obs = self.env.reset()
            if self.env_name == "Pong-v0":
                obs, _, _, _ = self.env.step(1) # Fire initial shot
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

    def collect_sode(self, agent, render=False):
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
                action = np.asarray(agent.fwd_numpy(game_obs[None])).reshape((agent.a_size,))
                data['actions'].append(action)
                rew = 0
                for i in range(agent.action_repeat):
                    obs, r, done, _ = self.step(action, render=render)
                    rew += r
            prepped = self.prep_fxn(obs)
            window.appendleft(prepped)
            game_obs = np.vstack(window)
            data['observs'].append(game_obs)
            data['rews'].append(rew)
            data['dones'].append(done)
            data['actions'].append(np.zeros_like(data['actions'][-1]))
            if self.rew_discount > 0:
                data['raw_rews'] = data['rews']
                data['rews'] = discount(data['rews'], data['dones'], self.rew_discount)
                data['rews'] = (data['rews']-data['rews'].mean())/(data['rews'].std()+1e-5)
        agent.train()
        return data

    def collect_sodes(self, agent, n_sodes, maxsize=None, render=False):
        """
        agent: Agent object
            network to take actions based on observations
        n_sodes: int
            number of episodes to collect
        maxsize: None or int
            sets the maximum size for the data to be collected, 
            if None no limit is set on the datasize
        """
        data = {
            "observs":[],
            "rews":[],
            "dones":[],
            "actions":[]
        }
        # Collect data
        running_size = 0
        for ep in range(n_sodes-1):
            new_data = self.collect_sode(agent, render=render)
            running_size += len(new_data['dones'])
            for k in data.keys():
                data[k].append(np.asarray(new_data[k]))
            if maxsize is not None and running_size > maxsize:
                break

        # Concatenate data
        for k in data.keys():
            data[k] = np.concatenate(data[k], axis=0)
        # Potentially shorten data
        if maxsize is not None and len(data['dones']) > maxsize:
            for k in data.keys():
                data[k] = data[k][-maxsize:]

        return data

