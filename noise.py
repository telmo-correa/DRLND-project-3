import torch
import numpy as np


# adapted from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, device, action_dimension, scale=0.1, mu=0, theta=0.2, sigma=0.2):
        self.device = device
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dimension).to(self.device) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.tensor(np.random.randn(len(x))).float().to(self.device)
        self.state = x + dx
        return self.state * self.scale
