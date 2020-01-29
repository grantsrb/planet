import torch
import numpy as np

class RandnAgent:
    def __call__(self, obs):
        velocity = torch.randn(1)*5 + 10
        direction = torch.clamp(torch.randn(1)*25, -49, 49)
        return [velocity.item(), direction.item()]

