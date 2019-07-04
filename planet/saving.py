import pickle
import torch
import numpy as np
import json

def save_data(data, save_path):
    """
    data: serializable object
        A list or a dict of data
    save_path: string
    """
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def read_data(save_path):
    """
    save_path: string
    """
    with open(save_path, 'rb') as f:
        return pickle.load(f)

def read_json(save_path):
    """
    save_path: string
    """
    with open(save_path) as f:
        return json.load(f)

def update_checkpoint(save_dict, save_folder):
    """
    torch_obj: serializable obj
        Some object containing torch objects
    save_path: string
    """
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chckpt_path = os.path.join(save_path, "checkpoint.p")
    torch.save(torch_obj, chckpt_path)

