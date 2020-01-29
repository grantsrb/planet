import numpy as np
import torch
import pickle
from planet.training import Trainer, TrainAutoencoder
from planet.experience import ExperienceReplay
from planet.agents import RandnAgent
from planet.models import *
from planet.game_envs import GameEnv
import planet.prep_fxns as prep_fxns
import sys
import json

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

if __name__=="__main__":
    # Load hyps
    if len(sys.argv) < 2:
        hyps = load_json("hyps/hyperparams.json")
    else:
        hyps = load_json(sys.argv[1])

    # Collect data
    prep_fxn = getattr(prep_fxns, hyps['prep_fxn'])
    game = GameEnv(hyps['env_name'], hyps['env_type'], prep_fxn)
    if len(game.obs_shape) > 1:
        obs_shape = (hyps['obs_depth'], *game.obs_shape[1:])
    else:
        obs_shape = (hyps['obs_depth'], *game.obs_shape)
    rand_agent = RandnAgent(obs_shape, game.a_size)
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

    # Make optimizer
    params = list(dynamics.parameters()) + list(decoder.parameters()) + list(rew_model.parameters())
    optim_type = hyps['optim_type']
    lr = hyps['lr']
    optimizer = getattr(torch.optim, optim_type)(params, lr=lr)

    # Make Trainer
    trainer = TrainAutoencoder(dynamics.encoder, decoder, dynamics.rssm, optimizer, exp_replay)

    # Train models
    n_epochs = hyps['n_epochs']
    for epoch in range(n_epochs):
        loss = trainer.train(hyps)
        print("Avg Loss:", loss)









