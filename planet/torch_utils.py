import torch
import torch.nn as nn
import numpy as np

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
    
