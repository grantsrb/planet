"""
Description:
    - System for storing game experience
    - Can read and write experience blocks
    - Can randomly sample experience blocks
"""

import numpy as np
from collections import deque
from deepretina.experiments import rolling_window

class ExperienceReplay:
    def __init__(self, intl_data=None, max_size=20000):
        self.max_size = max_size
        self.data = {
            "observs":None,
            "rews":None,
            "actions":None,
            "dones":None
        }
        # Fill main dict with with intl_data
        if intl_data is not None:
            self.add_new_data(intl_data)

    def add_new_data(self, new_data):
        # Append new data
        for k in self.data.keys():
            if k in new_data:
                new_data[k] = np.asarray(new_data[k]).astype(np.float32)
                if self.data[k] is not None:
                    self.data[k] = np.append(self.data[k], new_data[k], axis=0)
                else:
                    self.data[k] = new_data[k]
        # Remove stale data
        if self.__len__() > self.max_size:
            for k in self.data.keys():
                self.data[k] = self.data[k][-self.max_size:]

    def sample(self, batch_size=256, horizon=9):
        idxs = np.random.randint(0, len(self.data['dones'])-(horizon+1), size=(batch_size,))
        sample = dict()
        # Roll for sequence of steps in game to train on
        sample['obs_seq'] = np.asarray(rolling_window(self.data['observs'], horizon+1)[idxs]).astype(np.float32) 
        sample['rew_seq'] = np.asarray(rolling_window(self.data['rews'], horizon+1)[idxs]).astype(np.float32) 
        sample['done_seq'] = np.asarray(rolling_window(self.data['dones'], horizon+1)[idxs]).astype(np.float32) 
        sample['action_seq'] = np.asarray(rolling_window(self.data['actions'], horizon+1)[idxs]).astype(np.float32) 
        return sample

    def get_data(self, idxs, horizon=9):
        sample = dict()
        sample['obs_seq'] = np.asarray(rolling_window(self.data['observs'], horizon+1)[idxs]).astype(np.float32) 
        sample['rew_seq'] = np.asarray(rolling_window(self.data['rews'], horizon+1)[idxs]).astype(np.float32) 
        sample['done_seq'] = np.asarray(rolling_window(self.data['dones'], horizon+1)[idxs]).astype(np.float32) 
        sample['action_seq'] = np.asarray(rolling_window(self.data['actions'], horizon+1)[idxs]).astype(np.float32) 
        return sample
    
    def __len__(self):
        if self.data['dones'] is None:
            return 0
        return len(self.data['dones'])

