import torch
import torch.nn as nn

from itertools import chain


class Actor(nn.Module):

    def __init__(self, state_size, action_size, layers):
        super(Actor, self).__init__()

        iterator = chain([state_size], layers, [action_size])

        last_size = None
        args = []
        for layer_size in iterator:
            if last_size is not None:
                args.append(nn.BatchNorm1d(last_size))
                args.append(nn.Linear(last_size, layer_size))
                args.append(nn.ReLU())
            last_size = layer_size

        # Replace last ReLU layer with tanh
        del args[-1]
        args.append(nn.Tanh())

        self.network = nn.Sequential(*args)

    def forward(self, inputs):
        return self.network(inputs)


class Critic(nn.Module):

    def __init__(self, state_size, action_size, layers):
        super(Critic, self).__init__()

        iterator = chain([state_size + action_size], layers, [1])

        last_size = None
        args = []
        for layer_size in iterator:
            if last_size is not None:
                args.append(nn.Linear(last_size, layer_size))
                args.append(nn.ReLU())
            last_size = layer_size

        # Remove last ReLU layer
        del args[-1]

        self.normalize_state = nn.BatchNorm1d(state_size)
        self.network = nn.Sequential(*args)

    def forward(self, state, action):
        normalized_state = self.normalize_state(state)
        critic_inputs = torch.cat((normalized_state, action), dim=1)
        return self.network(critic_inputs)
