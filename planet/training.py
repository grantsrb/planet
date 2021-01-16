"""
Description:
    - Trains the models with preexisting data
    - Takes untrained models and data
    - For n loops, samples data for training
    - Performs both general training and overshooting training
    - Returns control to main when more data needs to be collected
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
import numpy as np
from tqdm import tqdm
from planet.utils import sample_gauss

def cuda_if(tensor, tensor2):
    if tensor2.is_cuda and tensor2.get_device() >= 0:
        return tensor.to(tensor2.get_device())
    return tensor

class Trainer:
    def __init__(self, dynamics, decoder, rew_model, optimizer, exp_replay, gate_q=None, return_q=None):
        self.dynamics = dynamics
        self.decoder = decoder
        self.rew_model = rew_model
        self.optimizer = optimizer
        self.exp_replay = exp_replay

        # These are used for multiprocessing
        self.gate_q = gate_q
        self.return_q = return_q

    def save_state_dicts(self, save_name):
        state_dicts = {
            "dynamics": self.dynamics.state_dict(),
            "decoder": self.decoder.state_dict(),
            "rew_model": self.rew_model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state_dicts, save_name)

    def load_state_dicts(save_name):
        state_dicts = torch.load(save_name)
        self.dynamics.load_state_dict("dynamics") 
        self.decoder.load_state_dict("decoder")
        self.rew_model.load_state_dict("rew_model")
        self.optimizer.load_state_dict("optimizer")

    def loop_training(self, hyps):
        assert self.gate_q is not None and self.return_q is not None
        n_epochs = hyps['n_epochs']
        n_train_loops = hyps['n_train_loops']
        while True:
            _ = self.gate_q.get()
            for loop in range(n_train_loops):
                print("Train Loop", loop)
                losses = self.train(hyps)
                for k in losses.keys():
                    print(k+":", np.mean(losses[k]))
            self.return_q.put(None)


    def train(self, hyps):
        batch_size = hyps['batch_size']
        horizon = hyps['horizon']
        h_size = hyps['h_size']
        s_size = hyps['s_size']
        grad_norm = hyps['grad_norm']

        loss_keys = ["tot_loss", "kl_loss", "overs_kl",
                     "obs_loss", "rew_loss", "overs_rew"]
        loop_losses = {k:[] for k in loss_keys}

        self.dynamics.train()
        self.decoder.train()        
        self.rew_model.train()

        n_steps = len(self.exp_replay)-(horizon+1)
        perm = np.random.permutation(n_steps)
        n_loops = n_steps//batch_size
        for loop in tqdm(range(n_loops)):
            # Get data (keys: obs_seq, rew_seq, done_seq, action_seq)
            idxs = perm[loop*batch_size:(loop+1)*batch_size]
            sample = self.exp_replay.get_data(idxs, horizon=horizon)
            obs_seq = cuda_if(torch.FloatTensor(sample['obs_seq']),
                                                    self.dynamics)
            rew_seq = cuda_if(torch.FloatTensor(sample['rew_seq']),
                                                    self.rew_model)
            done_seq = cuda_if(torch.FloatTensor(sample['done_seq']),
                                                    self.dynamics)
            not_done_seq = 1-done_seq
            action_seq = cuda_if(torch.FloatTensor(sample['action_seq']),
                                                    self.dynamics)

            # Collect transition predictions
            prev_h = torch.zeros(batch_size, h_size)
            prev_h = cuda_if(prev_h, self.dynamics)
            outputs = self.dynamics(obs_seq, prev_h,actions=action_seq,
                                             not_dones=not_done_seq,
                                             horizon=horizon,
                                             overshoot=False)
            hs,s_truths,s_preds,mu_truths,mu_preds,\
                                       sigma_truths,sigma_preds=outputs
            #TODO:Flaw in that overshots are not reset if episode ends
            outputs = self.dynamics(obs_seq, prev_h,actions=action_seq,
                                                    not_dones=None, 
                                                    horizon=horizon,
                                                    overshoot=True)
            h_overshots,s_overshots,_,mu_overshots,_,sigma_overshots,_=outputs

            # Flatten out time dimension along dim 0
            # (Time, Batch, Size) -> (Batch*Time, Size) 
            hs_flat = torch.cat(hs, dim=0)

            mu_truths_flat = torch.cat(mu_truths, dim=0)
            sigma_truths_flat = torch.cat(sigma_truths, dim=0)
            s_truths_flat=sample_gauss(mu_truths_flat, sigma_truths_flat)

            rew_targs = rew_seq.permute(1,0).contiguous().view(-1)
            arange = torch.arange(len(obs_seq.shape))
            obs_targs = obs_seq.permute(1,0,*arange[2:]).contiguous()
            obs_targs = obs_targs.view(-1, *obs_seq.shape[2:])

            if len(s_preds) > 0:
                s_preds_flat = torch.cat(s_preds, dim=0)
                mu_preds_flat = torch.cat(mu_preds, dim=0)
                sigma_preds_flat = torch.cat(sigma_preds, dim=0)

            # Losses
            obs_loss = cuda_if(torch.zeros(1), self.dynamics)
            rew_loss = cuda_if(torch.zeros(1), self.dynamics)
            kl_loss = cuda_if(torch.zeros(1), self.dynamics)
            overs_rew_loss = cuda_if(torch.zeros(1), self.dynamics)
            overs_kl_loss = cuda_if(torch.zeros(1), self.dynamics)

            # Obs loss
            inpt = torch.cat([hs_flat, s_truths_flat], dim=-1)
            reconstrs = self.decoder(inpt)
            obs_loss += F.mse_loss(reconstrs, obs_targs)

            # Rew Loss
            rew_preds = self.rew_model(inpt)
            rew_loss += F.mse_loss(rew_preds.squeeze(),
                                   rew_targs.squeeze())
            # KL Loss
            if horizon > 0:
                mu_truths_flat = torch.cat(mu_truths[1:], dim=0)
                sigma_truths_flat = torch.cat(sigma_truths[1:], dim=0)
                normal = Normal(torch.zeros_like(mu_truths_flat),
                                torch.ones_like(sigma_truths_flat))
                truth_normal_nograd = Normal(mu_truths_flat.data,
                                             sigma_truths_flat.data)
                truth_normal_grad = Normal(mu_truths_flat,
                                           sigma_truths_flat)
                pred_normal = Normal(mu_preds_flat, sigma_preds_flat)
                kl_loss += kl_divergence(truth_normal_grad,
                                         normal).mean()
                kl_loss += kl_divergence(truth_normal_nograd,
                                         pred_normal).mean()
                # Overshoot Flatten
                h_overs_flat = torch.cat(h_overshots[1:], dim=0)
                mu_overs_flat = torch.cat(mu_overshots[1:], dim=0)
                sigma_overs_flat = torch.cat(sigma_overshots[1:], dim=0)

                # Overshoot Rew Loss
                rew_targs=rew_seq[:,1:].permute(1,0).contiguous().view(-1)
                rand=sigma_overs_flat*torch.randn_like(sigma_overs_flat)
                s_overs_flat = mu_overs_flat + rand
                inpt = torch.cat([h_overs_flat, s_overs_flat], dim=-1)
                rew_pred = self.rew_model(inpt)
                overs_rew_loss+=F.mse_loss(rew_pred.squeeze(),rew_targs)

                # Overshoot KL Loss
                overs_normal = Normal(mu_overs_flat, sigma_overs_flat)
                overs_kl_loss += kl_divergence(truth_normal_nograd,
                                               overs_normal).mean()

            # Total Loss
            loss = obs_loss + rew_loss + overs_rew_loss + kl_loss\
                                                        + overs_kl_loss
            vals = [loss.item(), kl_loss.item(), overs_kl_loss.item(),
                   obs_loss.item(),rew_loss.item(),overs_rew_loss.item()]
            for k,v in zip(loss_keys, vals):
                loop_losses[k].append(v)

            self.optimizer.zero_grad()
            loss.backward()
            params = self.optimizer.param_groups[0]['params']
            nn.utils.clip_grad_norm_(params, grad_norm, norm_type=2)
            self.optimizer.step()

        return loop_losses

class DebugTrainer:
    def __init__(self, dynamics, decoder, rew_model, optimizer, exp_replay):
        self.dynamics = dynamics
        self.decoder = decoder
        self.rew_model = rew_model
        self.optimizer = optimizer
        self.exp_replay = exp_replay

    def train(self, hyps):
        batch_size = hyps['batch_size']
        horizon = hyps['horizon']
        h_size = hyps['h_size']
        s_size = hyps['s_size']
        grad_norm = hyps['grad_norm']

        loss_keys = ["tot_loss", "kl_loss", "obs_loss", "rew_loss"]
        loop_losses = {k:[] for k in loss_keys}

        n_steps = len(self.exp_replay)-(horizon+1)
        perm = np.random.permutation(n_steps)
        n_loops = n_steps//batch_size
        for loop in tqdm(range(n_loops)):
            # Get data (keys: obs_seq, rew_seq, done_seq, action_seq)
            idxs = perm[loop*batch_size:(loop+1)*batch_size]
            sample = self.exp_replay.get_data(idxs, horizon=horizon)
            obs_seq = cuda_if(torch.FloatTensor(sample['obs_seq']), self.dynamics)
            rew_seq = cuda_if(torch.FloatTensor(sample['rew_seq']), self.rew_model)
            done_seq = cuda_if(torch.FloatTensor(sample['done_seq']), self.dynamics)
            not_done_seq = 1-done_seq
            action_seq = cuda_if(torch.FloatTensor(sample['action_seq']), self.dynamics)

            # Collect transition predictions
            prev_h = torch.zeros(batch_size, h_size)
            outputs = self.dynamics(obs_seq, prev_h, actions=action_seq, not_dones=not_done_seq, 
                                                               horizon=horizon, overshoot=False)
            hs, s_truths, s_preds, mu_truths, mu_preds, sigma_truths, sigma_preds = outputs
            outputs = self.dynamics(obs_seq, prev_h, actions=action_seq, not_dones=None, 
                                                        horizon=horizon, overshoot=True)
            h_overshots, s_overshots, _, mu_overshots, _, sigma_overshots, _ = outputs

            obs_loss = torch.zeros(1)
            rew_loss = torch.zeros(1)
            kl_loss = torch.zeros(1)
            normal = Normal(torch.zeros_like(mu_truths[0]), torch.ones_like(sigma_truths[0]))
            for i in range(horizon+1):
                s = mu_truths[i] + sigma_truths[i]*torch.randn_like(sigma_truths[i])
                inpt = torch.cat([hs[i], s], dim=-1)

                # Obs Loss
                reconstr = self.decoder(inpt)
                obs_loss += F.mse_loss(reconstr, obs_seq[:,i])

                # Rew loss
                rew_pred = self.rew_model(inpt)
                rew_loss += F.mse_loss(rew_pred.squeeze(), rew_seq[:,i].squeeze())
                
                # KL Loss
                if i > 0:
                    truth_normal = Normal(mu_truths[i], sigma_truths[i])
                    pred_normal = Normal(mu_preds[i-1], sigma_preds[i-1])
                    kl_loss += kl_divergence(truth_normal, normal).mean()
                    kl_loss += kl_divergence(truth_normal, pred_normal).mean()

                    # Overshoot Loss
                    s = mu_overshots[i] + sigma_overshots[i]*torch.randn_like(sigma_overshots[i])
                    inpt = torch.cat([h_overshots[i], s], dim=-1)
                    rew_pred = self.rew_model(inpt)
                    rew_loss += F.mse_loss(rew_pred.squeeze(), rew_seq[:,i].squeeze())
                    overshot_normal = Normal(mu_overshots[i], sigma_overshots[i])
                    kl_loss += kl_divergence(truth_normal, overshot_normal).mean()

            # Total Loss
            loss = obs_loss + rew_loss + kl_loss
            vals = [loss.item(), kl_loss.item(), obs_loss.item(), rew_loss.item()]
            for k,v in zip(loss_keys, vals):
                loop_losses[k].append(v)

            self.optimizer.zero_grad()
            loss.backward()
            params = self.optimizer.param_groups[0]['params']
            nn.utils.clip_grad_norm_(params, grad_norm, norm_type=2)
            self.optimizer.step()

        return loop_losses

class TrainAutoencoder:
    def __init__(self, encoder, decoder, rssm, optimizer, exp_replay):
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.optimizer = optimizer
        self.exp_replay = exp_replay

    def train(self, hyps):
        batch_size = hyps['batch_size']
        horizon = hyps['horizon']
        h_size = hyps['h_size']
        s_size = hyps['s_size']
        grad_norm = hyps['grad_norm']

        avg_loss = 0

        n_steps = len(self.exp_replay)-(horizon+1)
        perm = np.random.permutation(n_steps)
        n_loops = n_steps//batch_size
        for loop in range(n_loops):
            # Get data (keys: obs_seq, rew_seq, done_seq, action_seq)
            idxs = perm[loop*batch_size:(loop+1)*batch_size]
            sample = self.exp_replay.get_data(idxs, horizon=horizon)
            obs_seq = cuda_if(torch.FloatTensor(sample['obs_seq']),
                                                self.encoder)
            done_seq = cuda_if(torch.FloatTensor(sample['done_seq']),
                                                 self.encoder)
            not_done_seq = 1-done_seq
            action_seq=cuda_if(torch.FloatTensor(sample['action_seq']),
                                                 self.encoder)
            prev_h = torch.zeros(batch_size, h_size)

            mu, sigma = self.encoder(obs_seq[:,0], prev_h)
            s = mu + sigma*torch.randn_like(sigma)
            inpt = torch.cat([prev_h, s], dim=-1)
            reconstr = self.decoder(inpt)

            assert np.array_equal(reconstr.shape, obs_seq[:,0].shape)
            obs_loss = F.mse_loss(reconstr, obs_seq[:,0])

            for i in range(horizon):
                h,mu,sigma = self.rssm(prev_h, s, action_seq[:,i])
                mu, sigma = self.encoder(obs_seq[:,i+1], h)
                s = mu + sigma*torch.randn_like(sigma)
                inpt = torch.cat([h, s], dim=-1)
                reconstr = self.decoder(inpt)
                obs_loss += F.mse_loss(reconstr, obs_seq[:,i+1])

            self.optimizer.zero_grad()
            obs_loss.backward()
            params = self.optimizer.param_groups[0]['params']
            nn.utils.clip_grad_norm_(params, grad_norm, norm_type=2)
            self.optimizer.step()
            print("Obs Loss:", obs_loss.item(), end="            \r")
            avg_loss += obs_loss.item()

        return avg_loss/n_loops






