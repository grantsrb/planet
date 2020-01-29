import numpy as np
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

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def cuda_if(module):
    if torch.cuda.is_available():
        return module.cuda()
    return module

if __name__=="__main__":
    starttime = time.time()

    # Load hyps
    if len(sys.argv) < 2:
        hyps = load_json("hyps/hyperparams.json")
    else:
        hyps = load_json(sys.argv[1])

    # Collect Initial Data
    prep_fxn = getattr(prep_fxns, hyps['prep_fxn'])
    game = GameEnv(hyps['env_name'], hyps['env_type'], prep_fxn, hyps['render'])
    if len(game.obs_shape) > 1:
        obs_shape = (hyps['obs_depth'], *game.obs_shape[1:])
    else:
        obs_shape = (hyps['obs_depth'], *game.obs_shape)
    action_repeat = hyps['action_repeat']
    rand_agent = RandnAgent(obs_shape, game.a_size, action_repeat=action_repeat)
    data = game.collect_sodes(rand_agent, hyps['n_sodes'], maxsize=hyps['max_steps'])

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
    agent = DynamicsAgent(obs_shape, a_size, hyps, dynamics, rew_model)

    # Make optimizer
    params = list(dynamics.parameters()) + list(decoder.parameters()) + list(rew_model.parameters())
    optim_type = hyps['optim_type']
    lr = hyps['lr']
    optimizer = getattr(torch.optim, optim_type)(params, lr=lr)

    # Make Trainer
    trainer = Trainer(dynamics, decoder, rew_model, optimizer, exp_replay)

    # Training Loop
    n_epochs = hyps['n_epochs']
    n_train_loops = hyps['n_train_loops']
    steps_seen = len(exp_replay)
    for epoch in range(n_epochs):
        temptime = time.time()
        print("\n\nData steps:", steps_seen,"| Tot running time:",temptime-starttime)
        for loop in range(n_train_loops):
            looptime = temptime
            losses = trainer.train(hyps)
            for k in losses.keys():
                print(k+":", np.mean(losses[k]))
            temptime = time.time()
            print("Execution Time:", temptime-looptime)
        print("\nCollecting new data...", end=" ")
        new_sode = game.collect_sode(agent)
        exp_replay.add_new_data(new_sode)
        quad = len(new_sode['rews'])//4
        means = [str(np.mean(new_sode['rews'][i:i+quad])) for i in range(0,quad*4,quad)]
        steps_seen += len(new_sode['rews'])
        print("| Exec Time:", time.time()-temptime)
        print("Inference rews by quadrant:", " -- ".join(means))









