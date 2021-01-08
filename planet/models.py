import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Flatten, Reshape, GaussianNoise, update_shape, conv_block, deconv_block
import numpy as np

FIRST_KSIZE = 9
N_LAYERS = 4

def cuda_if(tensor1, tensor2):
    if tensor2.is_cuda:
        return tensor1.to(tensor2.get_device())
    return tensor1

class CustomModule:
    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except:
            return False

    def get_device(self):
        try:
            return next(self.parameters()).get_device()
        except:
            return -1

class RSSM(nn.Module, CustomModule):
    def __init__(self, h_size, s_size, a_size, rnn_type="GRU",
                                               min_sigma=0.0001):
        super(RSSM, self).__init__()
        """
        h_size - int
            size of belief vector h
        s_size - int
            size of state vector s
        a_size - int
            size of action space vector a

        min_sigma - float
            the minimum value that the state standard deviation can take
        """
        if rnn_type == "GRU":
            rnn_type = "GRUCell"
        assert rnn_type == "GRUCell" # Only supported type currently
        self.h_size = h_size
        self.s_size = s_size
        self.a_size = a_size
        self.rnn_type = rnn_type
        self.min_sigma = min_sigma

        self.rnn = getattr(nn, rnn_type)(input_size=(s_size+a_size),
                                    hidden_size=h_size) # Dynamics rnn
        # Creates mu and sigma for state gaussian
        self.state_layer = nn.Linear(h_size, 2*s_size) 

    def forward(self, h, s, a):
        x = torch.cat([s,a], dim=-1)
        h_new = self.rnn(x, h)
        musigma = self.state_layer(h_new)
        mu, sigma = torch.chunk(musigma, 2, dim=-1)
        sigma = F.softplus(sigma) + self.min_sigma
        return h_new, mu, sigma
    
    def extra_repr(self):
        s = "h_size={}, s_size={}, a_size={}, n_layers={}, min_sigma={}"
        return s.format(self.h_size, self.s_size, self.a_size,
                                self.n_layers, self.min_sigma)

class Encoder(nn.Module, CustomModule):
    def __init__(self, obs_shape, h_size, s_size, bnorm=True,
                                                  noise=0,
                                                  min_sigma=0.0001):
        super(Encoder, self).__init__()
        """
        obs_shape - list like
            shape of observation (C, H, W)
        h_size - int
            size of belief vector h
        s_size - int
            size of state vector s
        bnorm - bool
            optional, if true, model uses batchnorm
        noise - float
            standard deviation of gaussian noise added at each layer
        min_sigma - float
            the minimum value that the state standard deviation can take
        """
        self.obs_shape = obs_shape
        self.h_size = h_size
        self.s_size = s_size
        self.bnorm = bnorm
        self.noise = noise
        self.min_sigma = min_sigma

        channels, height, width = obs_shape
        first_ksize = FIRST_KSIZE
        ksize = 3
        depth = 12
        padding = 0
        stride=1
        modules = []
        self.sizes = []
        conv = conv_block(channels, depth, ksize=first_ksize, stride=stride, 
                                    padding=padding, bnorm=bnorm, noise=noise)
        height, width = update_shape((height,width), kernel=first_ksize)
        self.sizes.append((height, width))
        modules.append(conv)

        for i in range(N_LAYERS):
            stride = 2 if i % 3 == 2 else 1
            height, width = update_shape((height,width),kernel=ksize,
                                                        stride=stride)
            modules.append(conv_block(depth, depth, ksize, stride=stride,
                                                    padding=padding,
                                                    bnorm=bnorm,
                                                    noise=noise))
            self.sizes.append((height, width))

        stride = 1
        height, width = update_shape((height,width), kernel=ksize,
                                                     stride=stride)
        print("Encoded:", height, width)
        self.sizes.append((height, width))
        conv = [nn.Conv2d(depth, depth, ksize),
                Flatten(), nn.BatchNorm1d(height*width*depth), 
                GaussianNoise(std=noise), nn.ReLU()]
        modules.append(nn.Sequential(*conv))
        
        self.features = nn.Sequential(*modules)
        self.emb_shape = (depth,height,width) # Formerly named out_shape
        self.encoder = nn.Linear(depth*height*width+h_size, 2*s_size)

    def forward(self, obs, h):
        """
        obs - torch FloatTensor
            observation
        h - torch FloatTensor
            belief vector
        """
        feats = self.features(obs)
        inpt = torch.cat([feats, h], dim=-1)
        musigma = self.encoder(inpt)
        mu, sigma = torch.chunk(musigma, 2, dim=-1)
        sigma = F.softplus(sigma) + self.min_sigma
        return mu, sigma

    def extra_repr(self):
        s = "obs_shape={}, bnorm={}, noise={}"
        return s.format(self.obs_shape, self.bnorm, self.noise)

class SimpleEncoder(nn.Module, CustomModule):
    def __init__(self, obs_shape, h_size, s_size, bnorm=True,
                                                  noise=0,
                                                  min_sigma=0.0001):
        super(SimpleEncoder, self).__init__()
        """
        obs_shape - list like
            shape of observation (C, H, W)
        h_size - int
            size of belief vector h
        s_size - int
            size of state vector s
        bnorm - bool
            optional, if true, model uses batchnorm
        noise - float
            standard deviation of gaussian noise added at each layer
        min_sigma - float
            the minimum value that the state standard deviation can take
        """
        self.obs_shape = obs_shape
        self.h_size = h_size
        self.s_size = s_size
        self.bnorm = bnorm
        self.noise = noise
        self.min_sigma = min_sigma

        depth, height, width = obs_shape
        self.emb_shape = obs_shape
        if bnorm:
            self.encoder = nn.Sequential(
                               nn.Linear(depth*height*width+h_size,
                                         depth*height*width+h_size),
                               nn.BatchNorm1d(depth*height*width+h_size),
                               nn.ReLU(),
                               nn.Linear(depth*height*width+h_size,
                                         2*s_size)
                           )
        else:
            self.encoder = nn.Sequential(
                              nn.Linear(depth*height*width+h_size,
                                        depth*height*width+h_size), 
                              nn.ReLU(),
                              nn.Linear(depth*height*width+h_size,
                                        2*s_size)
                           )

    def forward(self, obs, h):
        """
        obs - torch FloatTensor
            observation
        h - torch FloatTensor
            belief vector
        """
        feats = obs.view(obs.shape[0], -1)
        inpt = torch.cat([feats, h], dim=-1)
        musigma = self.encoder(inpt)
        mu, sigma = torch.chunk(musigma, 2, dim=-1)
        sigma = F.softplus(sigma) + self.min_sigma
        return mu, sigma

    def extra_repr(self):
        s = "obs_shape={}, bnorm={}, noise={}"
        return s.format(self.obs_shape, self.bnorm, self.noise)

class Decoder(nn.Module, CustomModule):
    def __init__(self, emb_shape, obs_shape, h_size, s_size,
                                                     bnorm=True,
                                                     noise=0):
        super(Decoder, self).__init__()
        """
        emb_shape - list like (C, H, W)
            the initial shape to reshape the embedding inputs (can
            take from encoder.emb_shape)
        obs_shape - list like (C, H, W)
            the final shape of the decoded tensor
        h_size - int
            size of belief vector h
        s_size - int
            size of state vector s
        bnorm - bool
            optional, if true, model uses batchnorm
        noise - float
            standard deviation of gaussian noise added at each layer
        """
        self.emb_shape = emb_shape
        self.obs_shape = obs_shape
        self.h_size = h_size
        self.s_size = s_size
        self.noise = noise
        self.bnorm = bnorm

        depth, height, width = emb_shape
        first_ksize = FIRST_KSIZE
        ksize = 3
        padding = 0
        modules = []
        self.sizes = []
        modules.append(Reshape((-1, depth, height, width)))
        deconv = deconv_block(depth, depth, ksize=ksize, stride=1,
                                                         padding=0,
                                                         bnorm=bnorm,
                                                         noise=noise)
        height, width = update_shape((height,width), kernel=ksize,
                                                     op="deconv")
        self.sizes.append((height, width))
        modules.append(deconv)

        for i in range(N_LAYERS-1):
            stride = 2 if i % 3 == 0 else 1
            modules.append(deconv_block(depth, depth, ksize=ksize,
                                        padding=padding, stride=stride,
                                        bnorm=self.bnorm, noise=noise))
            height, width = update_shape((height,width), kernel=ksize,
                                                         stride=stride,
                                                         op="deconv")
            self.sizes.append((height, width))

        modules.append(deconv_block(depth, depth, ksize=6,bnorm=bnorm,
                                                          noise=noise))
        height, width = update_shape((height,width),kernel=6,op="deconv")
        self.sizes.append((height, width))
        modules.append(deconv_block(depth, obs_shape[0],
                                           ksize=first_ksize,
                                           bnorm=False,
                                           activation=None,
                                           noise=0))
        height, width = update_shape((height,width),kernel=first_ksize,
                                                    op="deconv")
        print("decoder:", height, width)
        self.sizes.append((height, width))
        
        self.sequential = nn.Sequential(*modules)
        emb_size = int(np.prod(emb_shape))
        self.resize = nn.Sequential(nn.Linear(h_size+s_size, emb_size),
                                    Reshape((-1, *emb_shape)))

    def forward(self, x):
        """
        x - torch FloatTensor
            should be h and s concatenated
        """
        emb = self.resize(x)
        return self.sequential(emb)

    def extra_repr(self):
        s = "emb_shape={}, obs_shape={}, bnorm={}, noise={}"
        return s.format(self.emb_shape, self.obs_shape, self.bnorm,
                                                        self.noise)

class SimpleDecoder(nn.Module, CustomModule):
    def __init__(self, emb_shape, obs_shape, h_size, s_size, bnorm=True,
                                                             noise=0):
        super(SimpleDecoder, self).__init__()
        """
        emb_shape - list like (C, H, W)
            the initial shape to reshape the embedding inputs (can
            take from encoder.emb_shape)
        obs_shape - list like (C, H, W)
            the final shape of the decoded tensor
        h_size - int
            size of belief vector h
        s_size - int
            size of state vector s
        bnorm - bool
            optional, if true, model uses batchnorm
        noise - float
            standard deviation of gaussian noise added at each layer
        """
        self.emb_shape = emb_shape
        self.obs_shape = obs_shape
        self.h_size = h_size
        self.s_size = s_size
        self.noise = noise
        self.bnorm = bnorm

        depth, height, width = emb_shape
        emb_size = depth*height*width
        if bnorm:
            self.resize = nn.Sequential(
                                    nn.Linear(h_size+s_size,emb_size),
                                    nn.BatchNorm1d(emb_size),
                                    nn.ReLU(), 
                                    nn.Linear(emb_size, emb_size),
                                    Reshape((-1, *emb_shape)))
        else:
            self.resize = nn.Sequential(
                                    nn.Linear(h_size+s_size,emb_size),
                                    nn.ReLU(), 
                                    nn.Linear(emb_size, emb_size),
                                    Reshape((-1, *emb_shape)))

    def forward(self, x):
        """
        x - torch FloatTensor
            should be h and s concatenated
        """
        emb = self.resize(x)
        return emb

    def extra_repr(self):
        s = "emb_shape={}, obs_shape={}, bnorm={}, noise={}"
        return s.format(self.emb_shape, self.obs_shape, self.bnorm,
                                                        self.noise)

class RewardModel(nn.Module, CustomModule):
    def __init__(self, h_size, s_size, n_feats=256, n_layers=3,
                                                    noise=0,
                                                    bnorm=False):
        super(RewardModel, self).__init__()
        """
        h_size - int
            size of belief vector h
        s_size - int
            size of state vector s
        n_feats - int
            number of internal layers of reward model
        """
        self.h_size = h_size
        self.s_size = s_size
        self.n_feats = n_feats
        self.bnorm = bnorm
        self.noise = noise

        modules = [nn.Linear(h_size+s_size, n_feats)]
        if bnorm:
            modules.append(nn.BatchNorm1d(n_feats))
        modules += [GaussianNoise(std=noise), nn.ReLU()]
        
        for i in range(n_layers-2):
            modules = [nn.Linear(h_size+s_size, n_feats)]
            if bnorm:
                modules.append(nn.BatchNorm1d(n_feats))
            modules += [GaussianNoise(std=noise), nn.ReLU()]

        modules.append(nn.Linear(n_feats, 1))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        """
        x - torch FloatTensor (B, H*S)
            should be h and s concatenated
        """
        return self.sequential(x)

    def extra_repr(self):
        s = "h_size={}, s_size={}, bnorm={}, noise={}"
        return s.format(self.h_size,self.s_size,self.bnorm,self.noise)

class Dynamics(nn.Module, CustomModule):
    """
    This class is used to perform the planning and predicting for PlaNet
    """
    def __init__(self, obs_shape, h_size, s_size, a_size,
                                                  bnorm=False,
                                                  noise=0,
                                                  env_name=None,
                                                  min_sigma=1e-4):
        super(Dynamics, self).__init__()
        """
        obs_shape - list like
            shape of observation (C, H, W)
        h_size: int
            size of belief vector h
        s_size: int
            size of state vector s
        a_size: int
            size of action space
        horizon: int
            number of planning steps
        """
        self.obs_shape = obs_shape
        self.h_size = h_size
        self.s_size = s_size
        self.a_size = a_size
        self.min_sigma = min_sigma

        if env_name is not None and env_name == "Pendulum-v0":
            self.encoder = SimpleEncoder(obs_shape, h_size, 
                                                    s_size,
                                                    bnorm=bnorm,
                                                    noise=noise,
                                                    min_sigma=min_sigma)
        else:
            self.encoder = Encoder(obs_shape, h_size,
                                              s_size,
                                              bnorm=bnorm,
                                              noise=noise,
                                              min_sigma=min_sigma)
        self.rssm = RSSM(h_size, s_size, a_size, min_sigma=min_sigma) 

    def forward(self, observs, prev_h, actions, not_dones,
                                                horizon=5,
                                                overshoot=False,
                                                prev_mu=None,
                                                prev_sigma=None):
        """
        observs: torch FloatTensor (Batch, Horizon+1, D, H, W)
            The observations used by the agent during game play
        prev_h: torch FloatTensor (Batch, h_size)
            The prior h value
        actions: torch FloatTensor (Batch, horizon+1, a_size)
            The actions taken during game play
        not_dones: torch FloatTensor (Batch, horizon+1)
            binary array where 0 denotes end of an episode
        horizon: int
            only used if overshooting, denotes the number of overshot
            steps to take. If 0 only returns first encoding from
            the observation.
        overshoot: bool
            indicates that the network should make predictions for
            horizon steps
        prev_mu: tensor (Batch, s_size)
            optional, only used if observs is None
        prev_sigma: tensor (Batch, s_size)
            optional, only used if observs is None
        """
        hs,s_truths,s_preds,mu_truths = [],[],[],[]
        mu_preds,sigma_truths,sigma_preds = [],[],[]
        mu, sigma = prev_mu, prev_sigma
        if observs is not None:
            mu, sigma = self.encoder(observs[:,0], prev_h)
        mu_truths.append(mu)
        sigma_truths.append(sigma)
        prev_s = mu + sigma*torch.randn_like(sigma)
        s_truths.append(prev_s)
        hs.append(prev_h)
        horizon = min(horizon, actions.shape[1])

        for t in range(horizon):
            h, mu, sigma = self.rssm(hs[-1], s_truths[-1], actions[:,t])
            hs.append(h)
            if not overshoot:
                sigma_preds.append(sigma)
                mu_preds.append(mu)
                s = mu + sigma*torch.randn_like(sigma)
                s_preds.append(s)
                h = (h.permute(1,0)*not_dones[:,t]).permute(1,0)
                mu, sigma = self.encoder(observs[:,t+1], h)
            mu_truths.append(mu)
            sigma_truths.append(sigma)
            s = mu + sigma*torch.randn_like(sigma)
            s_truths.append(s)

        # Note that preds have length of horizon in non-overshoot case
        # and length of 0

        # in overshoot case. The truth arrays are used for overshooting
        # with the encoded mu and sigma as the first elements.
        return hs, s_truths, s_preds, mu_truths, mu_preds,sigma_truths,\
                                                          sigma_preds

