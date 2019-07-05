"""
Description:
    - System for storing game experience
    - Can read and write experience blocks
    - Can randomly sample experience blocks
"""

import numpy as np
from collections import deque

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

def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension.

    Taken from deepretina package (https://github.com/baccuslab/deep-retina/)

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window <= array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - (window-1), window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr

