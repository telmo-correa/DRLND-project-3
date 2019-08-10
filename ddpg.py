import torch
from torch.optim import Adam

from model import Actor, Critic
from noise import OUNoise


class DDPGAgent:

    def __init__(
        self,
        state_size,
        action_size,
        num_agents,
        actor_network_units,
        critic_network_units,
        optimizer_learning_rate_actor=1e-3,
        optimizer_learning_rate_critic=1e-3,
        actor_weight_decay=0,
        critic_weight_decay=0,
        noise_scale=0.1,
        noise_theta=0.2,
        noise_sigma=0.2,
        device=None
    ):
        """ Initializes the training instance for a single agent.

        :param state_size:  (int) Space size for state observations per agent
        :param action_size:  (int) Space size for actions per agent
        :param num_agents: (int) Number of agents used in problem
        :param actor_network_units:  (list of ints) Network topology for actor networks
        :param critic_network_units:  (list of ints) Network topology for critic networks
        :param optimizer_learning_rate_actor:  (float)  Learning rate for actor loss optimizer
        :param optimizer_learning_rate_critic:  (float)  Learning rate for critic loss optimizer
        :param optimizer_weight_decay_actor:  (float) Weight decay for actor loss optimizer
        :param optimizer_weight_decay_critic:  (float)  Weight decay for critic loss optimizer
        :param noise_scale:  (float)  Scale for noise process
        :param noise_theta:  (float)  Theta parameter for noise process
        :param noise_sigma:  (float)  Sigma parameter for noise process
        :param device:  (torch.device)  Object representing the device where to allocate tensors
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_size, action_size, actor_network_units).to(device)
        self.target_actor = Actor(state_size, action_size, actor_network_units).to(device)

        self.critic = Critic(state_size * num_agents, action_size * num_agents, critic_network_units).to(device)
        self.target_critic = Critic(state_size * num_agents, action_size * num_agents, critic_network_units).to(device)

        self.noise = OUNoise(device, action_size, scale=noise_scale, mu=0, theta=noise_theta, sigma=noise_sigma)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=optimizer_learning_rate_actor, weight_decay=actor_weight_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=optimizer_learning_rate_critic, weight_decay=critic_weight_decay)

        self.hard_update()

    def act(self, states, target=False, noise=0.0, train=False):
        """ Returns the selected actions for the given states according to the current policy

        :param state: (array-like) Current states
        :param target:  (boolean, default False) Whether to use local networks or target networks
        :param noise:  (float, default 0)  Scaling parameter for noise process
        :param train:  (boolean, default False)  Whether to keep gradients for training purposes
        :return: action (array-like)  List of selected actions
        """

        actor_network = self.target_actor if target else self.actor
        if not train:
            actor_network.eval()
        action = actor_network(states)
        if not train:
            actor_network.train()

        if noise != 0:
            noisy_action = action + noise * self.noise.noise()
            return noisy_action.clamp(-1, 1)

        return action

    def hard_update(self):
        """Performs a hard update on the target networks (copying the values from the local networks). """
        DDPGAgent._hard_update(self.target_actor, self.actor)
        DDPGAgent._hard_update(self.target_critic, self.critic)

    def soft_update(self, tau):
        """  Performs a soft update on the target networks:
            target_params := target_params * (1 - tau) + local_params * tau

        :param tau: (float) Update scaling parameter
        """
        DDPGAgent._soft_update(self.target_actor, self.actor, tau)
        DDPGAgent._soft_update(self.target_critic, self.critic, tau)

    @staticmethod
    def _hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

