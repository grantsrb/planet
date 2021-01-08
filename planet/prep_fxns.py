import numpy as np
import torch
import cv2

def no_op(obs):
    return obs

def pong(obs):
    """
    obs: ndarray (C, H, W)
    """
    obs = obs[:, 35:195] # crop
    obs = obs[0,::2,::2] # downsample by factor of 2
    obs[obs == 144] = 0 # erase background (background type 1)
    obs[obs == 109] = 0 # erase background (background type 2)
    obs[obs != 0] = 1 # everything else (paddles, ball) just set to 1
    obs = cv2.resize(obs, dsize=(50,50), interpolation=cv2.INTER_CUBIC)
    return obs[None]

def simple_orb(obs):
    """
    obs: ndarray (C, H, W)
    """
    obs = obs.astype(np.float32)
    obs = obs.mean(0)
    obs = cv2.resize(obs, dsize=(50,50), interpolation=cv2.INTER_CUBIC)
    obs = obs/255-.5
    return obs[None]

def center_zero2one(obs):
    """
    obs: ndarray (C, H, W)
        values must range from 0-1
    """
    obs = obs.astype(np.float32)
    obs = 3*(obs-.5)/.5
    if len(obs.shape)==2:
        return obs[None]
    return obs

