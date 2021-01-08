import torch
import torch.nn as nn
import numpy as np

def sample_gauss(mu, sig):
    return mu + torch.randn_like(sig)*sig

def discount(rews, dones, disc_factor=0.99):
    discounts = np.zeros(len(rews))
    discounts[-1] = rews[-1]
    for i in reversed(range(len(rews)-1)):
        discounts[i] = rews[i] + (1-dones[i])*disc_factor*discounts[i+1]
    return discounts

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
        if type(array) == type(np.array([])):
            array = array.T
        elif len(array.shape) >= 2:
            l = list([i for i in range(len(array.shape))])
            array = array.transpose(*reversed(l))

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

def one_hot_encode(sample, dim=-1):
    """
    Creates a one-hot-encoding from the sample. Uses the maximum idx as the 1.

    sample: tensor (B,A)
    """
    assert len(sample.shape) == 2
    cols = torch.argmax(sample, dim=dim).long()
    rows = torch.arange(len(cols)).long()
    sample = torch.zeros_like(sample)
    sample[rows, cols] = 1
    return sample

def cuda_if(tensor1, tensor2):
    if tensor2.is_cuda:
        return tensor1.to(tensor2.get_device())
    return tensor1

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        """
        std - float
            the standard deviation of the noise to add to the layer.
        """
        super(GaussianNoise, self).__init__()
        self.std = std
        self.sigma = nn.Parameter(torch.ones(1)*std, requires_grad=False)
    
    def forward(self, x):
        if not self.training or self.std == 0: # No noise during evaluation
            return x
        self.sigma.requires_grad = False
        if self.sigma.is_cuda:
            noise = self.sigma * torch.randn(x.size()).to(self.sigma.get_device())
        else:
            noise = self.sigma * torch.randn(x.size())
        return x + noise

    def extra_repr(self):
        return 'std={}'.format(self.std)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

def update_shape(shape, kernel=3, padding=0, stride=1, op="conv"):
    """
    Calculates the new shape of the tensor following a convolution or deconvolution
    """
    if type(shape) != type(int()):
        shape = np.asarray(shape)
        if type(kernel) == type(int()):
            kernel = np.asarray([kernel for i in range(len(shape))])
        if type(padding) == type(int()):
            padding = np.asarray([padding for i in range(len(shape))])
        if type(stride) == type(int()):
            stride = np.asarray([stride for i in range(len(shape))])

    if op == "conv":
        shape = (shape - kernel + 2*padding)/stride + 1
    elif op == "deconv" or op == "conv_transpose":
        shape = (shape - 1)*stride + kernel - 2*padding
    return (int(s) for s in shape)

def deconv_block(in_depth, out_depth, ksize=3, stride=1, padding=0, bnorm=False, activation='relu', noise=0.05):
    block = []
    block.append(nn.ConvTranspose2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
    if bnorm:
        block.append(nn.BatchNorm2d(out_depth))
    block.append(GaussianNoise(std=noise))
    if activation is None:
        pass
    elif activation.lower() == 'relu':
        block.append(nn.ReLU())
    elif activation.lower() == 'tanh':
        block.append(nn.Tanh())
    return nn.Sequential(*block)
    
def conv_block(in_depth, out_depth, ksize=3, stride=1, padding=0, bnorm=False, activation='relu', noise=0.05):
    block = []
    block.append(nn.Conv2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
    if bnorm:
        block.append(nn.BatchNorm2d(out_depth))
    block.append(GaussianNoise(std=noise))
    if activation is None:
        pass
    elif activation.lower() == 'relu':
        block.append(nn.ReLU())
    elif activation.lower() == 'tanh':
        block.append(nn.Tanh())
    return nn.Sequential(*block)
    
