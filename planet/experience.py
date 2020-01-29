"""
Description:
    - System for storing game experience
    - Can read and write experience blocks
    - Can randomly sample experience blocks
"""

import numpy as np
import torch
import pickle
from collections import deque
from planet.utils import rolling_window

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
        """
            new_data: dict
                keys:
                    "observs":list,
                    "rews":list,
                    "actions":list,
                    "dones":list
        """
        # Append new data
        for k in self.data.keys():
            if k in new_data and len(new_data['dones']) > 0:
                try:
                    new_data[k] = np.asarray(new_data[k], dtype=np.float)
                except:
                    print(type(new_data))
                    pass
                if self.data[k] is not None:
                    self.data[k] = np.append(self.data[k], new_data[k], axis=0)
                else:
                    self.data[k] = new_data[k]
        # Remove stale data
        if self.__len__() > self.max_size:
            for k in self.data.keys():
                self.data[k] = self.data[k][-self.max_size:] # Remove oldest data

    def load(self, save_name):
        with open(save_name, 'rb') as f:
            data = pickle.load(f)
        self.add_new_data(data)

    def save(self, save_name):
        with open(save_name, 'wb') as f:
            pickle.dump(self.data, f)

    def priority_sample(self, batch_size=256, horizon=9):
        """
        Samples data points that contain rewards. Potentially 
        returns a batch of data smaller than the batch size.
        """
        event_idxs = (self.data['rews']!=0)
        idxs = np.random.permutation(event_idxs.sum())[:batch_size]
        sample = dict()
        # Roll for sequence of steps in game to train on
        sample['obs_seq'] = np.asarray(rolling_window(self.data['observs'], horizon+1)[event_idxs][idxs]).astype(np.float32) 
        sample['rew_seq'] = np.asarray(rolling_window(self.data['rews'], horizon+1)[event_idxs][idxs]).astype(np.float32) 
        sample['done_seq'] = np.asarray(rolling_window(self.data['dones'], horizon+1)[event_idxs][idxs]).astype(np.float32) 
        sample['action_seq'] = np.asarray(rolling_window(self.data['actions'], horizon+1)[event_idxs][idxs]).astype(np.float32) 
        return sample

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

class MPExperienceReplay:
    def __init__(self, shared_tensors, intl_data=None):
        self.data_fill = torch.zeros(1, dtype=torch.int).share_memory_()
        keys = ["observs", "rews", 'actions', 'dones']
        self.data = {k:shared_tensors[k] for k in keys}
        self.max_size = len(self.data[keys[0]])
        # Fill main dict with with intl_data
        if intl_data is not None:
            self.add_new_data(intl_data)

    def add_new_data(self, new_data):
        """
            new_data: dict
                keys:
                    "observs":list,
                    "rews":list,
                    "actions":list,
                    "dones":list
        """
        # Append new data
        for k in self.data.keys():
            if k in new_data and len(new_data['dones']) > 0:
                try:
                    if type(new_data[k]) == type(torch.IntTensor([])): new_data[k] = new_data[k].type(torch.float32)
                    new_data[k] = torch.FloatTensor(new_data[k]).type(self.data[k].dtype)
                except Exception as e:
                    print(e)
                    print("Error converting new data to float tensor")
                    print("Key:", k)
                    print("type:", type(new_data[k]))
                try:
                    if self.data_fill.item() == 0:
                        end = min(len(new_data[k]), len(self.data[k]))
                        self.data[k][:end] = new_data[k][-end:]
                    else:
                        cat = torch.cat([self.data[k][:self.data_fill.item()], new_data[k].squeeze()], dim=0)
                        self.data[k][:] = cat[-self.max_size:]
                except Exception as e:
                    print(e)
                    print("Error transferring new data to exp replay data tensors")
                    print("key:", k)
                    print("data:", self.data[k].shape, type(self.data[k]))
                    print("new_data:", new_data[k].shape, type(new_data[k]))
        if self.data_fill.item() < self.max_size:
            self.data_fill[0] += len(new_data["dones"])
        self.data_fill[0] = min(self.data_fill.item(), self.max_size)

    def load(self, save_name):
        with open(save_name, 'rb') as f:
            data = pickle.load(f)
        self.add_new_data(data)

    def save(self, save_name):
        with open(save_name, 'wb') as f:
            pickle.dump(self.data, f)

    def sample(self, batch_size=256, horizon=9):
        idxs = np.random.randint(0, len(self.data['dones'])-(horizon+1), size=(batch_size,))
        sample = dict()
        # Roll for sequence of steps in game to train on
        sample['obs_seq'] = np.asarray(rolling_window(self.data['observs'].numpy(), horizon+1)[idxs]).astype(np.float32) 
        sample['rew_seq'] = np.asarray(rolling_window(self.data['rews'].numpy(), horizon+1)[idxs]).astype(np.float32) 
        sample['done_seq'] = np.asarray(rolling_window(self.data['dones'].numpy(), horizon+1)[idxs]).astype(np.float32) 
        sample['action_seq'] = np.asarray(rolling_window(self.data['actions'].numpy(), horizon+1)[idxs]).astype(np.float32) 
        return sample

    def get_data(self, idxs, horizon=9):
        sample = dict()
        sample['obs_seq'] = np.asarray(rolling_window(self.data['observs'].numpy(), horizon+1)[idxs]).astype(np.float32) 
        sample['rew_seq'] = np.asarray(rolling_window(self.data['rews'].numpy(), horizon+1)[idxs]).astype(np.float32) 
        sample['done_seq'] = np.asarray(rolling_window(self.data['dones'].numpy(), horizon+1)[idxs]).astype(np.float32) 
        sample['action_seq'] = np.asarray(rolling_window(self.data['actions'].numpy(), horizon+1)[idxs]).astype(np.float32) 
        return sample
    
    def __len__(self):
        return self.data_fill.item()

