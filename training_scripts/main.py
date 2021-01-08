import numpy as np
import os
import torch
import pickle
import time
from planet.training import *
from planet.experience import ExperienceReplay
from planet.agents import *
from planet.models import *
from planet.game_envs import GameEnv
import planet.prep_fxns as prep_fxns
import sys
import json

DEVICE = torch.device("cuda:0")

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def cuda_if(module):
    if torch.cuda.is_available():
        return module.to(DEVICE)
    return module

if __name__=="__main__":
    starttime = time.time()

    # Load hyps
    if len(sys.argv) < 2:
        hyps = load_json("hyps.json")
    else:
        hyps = load_json(sys.argv[1])

    if hyps['exp_name'] == "test":
        hyps['n_train_loops'] = 2
        hyps['n_new_sodes'] = 2
        hyps['n_epochs'] = 5

    # Initiallize saving variables
    save_folder = hyps['exp_name']
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    _, _, file_names = next(os.walk(save_folder))
    save_num = 0
    for fname in file_names:
        if "state_dict" in fname:
            num = int(fname.split("_")[-1].split(".")[0])
            if num+1 > save_num:
                save_num = num+1
    file_name = "state_dicts_"+str(save_num)+".pth"
    save_name = os.path.join(save_folder, file_name)

    # Prepare game and random agent
    prep_fxn = getattr(prep_fxns, hyps['prep_fxn'])
    game = GameEnv(hyps['env_name'], hyps['env_type'], prep_fxn,
                                    rew_discount=hyps['rew_discount'])
    if len(game.obs_shape) > 1:
        depth = hyps['obs_depth']*game.obs_shape[0]
        obs_shape = (depth, *game.obs_shape[1:])
    else:
        obs_shape = (hyps['obs_depth'], *game.obs_shape)
    action_repeat = hyps['action_repeat']
    discrete = hasattr(game.env.action_space, "n")
    rand_agent = RandnAgent(obs_shape, hyps['obs_depth'], game.a_size,
                                       action_repeat=action_repeat,
                                       discrete=discrete)
    # Collect Initial Data
    data = game.collect_sodes(rand_agent, hyps['n_sodes'],
                                          maxsize=hyps['max_steps'])
    if hyps['env_name'] == "Pong-v0":
        rew = data['rews'].sum()/(data['dones'].sum())
    else:
        rew = data['rews'].mean()
    print("Baseline Rew:", rew)

    # Make Experience Replay
    exp_replay = ExperienceReplay(intl_data=data, max_size=hyps['max_steps'])
    
    # Make Models
    h_size = hyps['h_size']
    s_size = hyps['s_size']
    a_size = data['actions'].shape[-1]
    min_sigma = hyps['min_sigma']
    obs_shape = data['observs'].shape[1:]
    env_name = hyps['env_name']
    bnorm = hyps['bnorm']
    dynamics = Dynamics(obs_shape, h_size, s_size, a_size, bnorm=bnorm, env_name=env_name, min_sigma=min_sigma)
    if env_name=="Pendulum-v0":
        decoder = SimpleDecoder(dynamics.encoder.emb_shape, obs_shape, h_size, s_size, bnorm=bnorm)
    else:
        decoder = Decoder(dynamics.encoder.emb_shape, obs_shape, h_size, s_size, bnorm=bnorm)
    rew_model = RewardModel(h_size, s_size, bnorm=bnorm)

    # Put them up on the gpu
    dynamics = cuda_if(dynamics)
    decoder = cuda_if(decoder)
    rew_model = cuda_if(rew_model)

    # Make Agent
    obs_depth = hyps['obs_depth']
    agent = DynamicsAgent(obs_shape, obs_depth, a_size, hyps, dynamics, rew_model, discrete=discrete)

    # Make optimizer
    params = list(dynamics.parameters()) + list(decoder.parameters()) + list(rew_model.parameters())
    optim_type = hyps['optim_type']
    lr = hyps['lr']
    optimizer = getattr(torch.optim, optim_type)(params, lr=lr)

    # Potentially restore previous models
    if "load_file" in hyps and hyps['load_file'] is not None:
        try:
            state_dicts = torch.load(hyps['load_file'])
            dynamics.load_state_dict(state_dicts['dynamics'])
            decoder.load_state_dict(state_dicts['decoder'])
            rew_model.load_state_dict(state_dicts['rew_model'])
            optimizer.load_state_dict(state_dicts['optimizer'])
        except Exception as e:
            print(e)
            print("Failed to load state dicts")

    # Make Trainer
    trainer = Trainer(dynamics, decoder, rew_model, optimizer, exp_replay)

    # Training Loop
    n_epochs = hyps['n_epochs']
    n_train_loops = hyps['n_train_loops']
    steps_seen = len(exp_replay)
    for epoch in range(n_epochs):
        log_str = ""
        s = "\n\nEpoch: {}".format(epoch)
        print(s)
        log_str += s + "\n"
        temptime = time.time()
        s = "Data steps: {} | Tot running time: {}"
        s = s.format(steps_seen,temptime-starttime)
        print(s)
        log_str += s + "\n"
        for loop in range(n_train_loops):
            looptime = temptime
            losses = trainer.train(hyps)
            for k in losses.keys():
                print(k+":", np.mean(losses[k]))
            temptime = time.time()
            print("Execution Time:", temptime-looptime)
        print("Saving model to:", save_name)
        state_dicts = {
            "dynamics": dynamics.state_dict(),
            "decoder": decoder.state_dict(),
            "rew_model": rew_model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(state_dicts, save_name)
        print("\nCollecting new data...", end=" ")
        for _ in range(hyps['n_new_sodes']):
            new_sode = game.collect_sode(agent, render=hyps['render'])
            exp_replay.add_new_data(new_sode)
            steps_seen += len(new_sode['rews'])
        print("| Exec Time:", time.time()-temptime)
        rews = new_sode['rews'] if "raw_rews" not in new_sode else new_sode['raw_rews']
        if hyps['env_name'] == "Pong-v0":
            s = "Inference rews: {}".format(rews.sum())
        elif hyps['env_name'] == "Pendulum-v0":
            quad = max(len(new_sode['rews'])//4, 1)
            rng = range(0,quad*4,quad)
            means = [str(np.mean(rews[i:i+quad])) for i in rng]
            strmeans = " | ".join(means)
            s="Inference rews by quadrant (early->later): "+strmeans
        else:
            s = "Avg Inference Rew: {:.5f}".format(np.mean(rews))
        print(s)
        log_str += s + "\n"
    s = "\nTotal running time: {}".format(time.time()-starttime)
    print(s)
    log_str += s + "\n"
    log_path = save_name.split(".")[0] + "_log.txt"
    with open(log_path,'a') as f:
        f.write(log_str)










