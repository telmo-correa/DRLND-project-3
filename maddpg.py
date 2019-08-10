import torch
import torch.nn.functional as F

import numpy as np

from ddpg import DDPGAgent
from memory import ReplayBuffer


class MADDPG:

    def __init__(
        self,
        state_size,
        action_size,
        num_agents=2,
        actor_network_units=(64, 64),
        critic_network_units=(64, 64),
        optimizer_learning_rate_actor=1e-3,
        optimizer_learning_rate_critic=1e-3,
        optimizer_weight_decay_actor=0,
        optimizer_weight_decay_critic=0,
        noise_scale=0.1,
        noise_theta=0.2,
        noise_sigma=0.2,
        gamma=0.99,
        tau=1e-3,
        gradient_clip_actor=1.0,
        gradient_clip_critic=1.0,
        buffer_size=int(1e5),
        batch_size=128,
        update_every=1,
        device=None
    ):
        """Initializes a multi-agent training instance.

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
        :param gamma:  (float)  Discount rate for rewards
        :param tau:  (float)  Update parameter for network soft updates
        :param gradient_clip_actor:  (float)  Gradient clipping parameter for actor loss optimizer
        :param gradient_clip_critic:  (float)  Gradient clipping parameter for critic loss optimizer
        :param buffer_size:  (int)  Size of replay memory buffer
        :param batch_size:  (int)  Size of training minibatches
        :param update_every:  (int)  Number of steps between training
        :param device:  (torch.device)  Object representing the device where to allocate tensors
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.gamma = gamma
        self.tau = tau
        self.gradient_clip_actor = gradient_clip_actor
        self.gradient_clip_critic = gradient_clip_critic

        self.update_every = update_every
        self.batch_size = batch_size
        self.t_step = 0
        self.episode = 0

        self.agents = []
        for i in range(num_agents):
            self.agents.append(DDPGAgent(
                state_size=state_size,
                action_size=action_size,
                actor_network_units=actor_network_units,
                critic_network_units=critic_network_units,
                num_agents=num_agents,
                optimizer_learning_rate_actor=optimizer_learning_rate_actor,
                optimizer_learning_rate_critic=optimizer_learning_rate_critic,
                actor_weight_decay=optimizer_weight_decay_actor,
                critic_weight_decay=optimizer_weight_decay_critic,
                noise_scale=noise_scale,
                noise_theta=noise_theta,
                noise_sigma=noise_sigma,
                device=device
            ))

        # Replay memory
        self.memory = ReplayBuffer(
            buffer_size=buffer_size,
            device=device
        )

    def step(self, state, action, reward, next_state, done):
        """ Store a single agent step, learning every N steps

         :param state: (array-like) Initial states on the visit
         :param action: (array-like) Actions on the visit
         :param reward: (array-like) Rewards received on the visit
         :param next_state:  (array-like) States reached after the visit
         :param done:  (array-like) Flag whether the next states are terminal states
         """

        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random batch and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

        # Keep track of episode number
        if np.any(done):
            self.episode += 1

    def act(self, states, target=False, noise=1.0):
        """ Returns the selected actions for the given states according to the current policy

        :param states: (array-like) Current states
        :param target:  (boolean, default False) Whether to use local networks or target networks
        :param noise:  (float, default 1)  Scaling parameter for noise process
        :return: action (array-like)  List of selected actions
        """

        if type(states) == np.ndarray:
            states = torch.from_numpy(states).float().to(self.device)

        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                agent = self.agents[i]
                action = agent.act(states[i, :].view(1, -1), target=target, noise=noise)
                actions.append(action.squeeze())
        actions = torch.stack(actions)

        return actions.cpu().data.numpy()

    def learn(self, experiences):
        """ Performs training for each agent based on the selected set of experiencecs

        :param experiences:   Batch of experience tuples (s, a, r, s', d) collected from the replay buffer
        """

        state, action, rewards, next_state, done = experiences

        state = state.view(-1, self.num_agents, self.state_size)
        action = action.view(-1, self.num_agents, self.action_size)
        rewards = rewards.view(-1, self.num_agents)
        next_state = next_state.view(-1, self.num_agents, self.state_size)
        done = done.view(-1, self.num_agents)

        # Select agent being updated based on ensemble at time of samples
        for agent_number in range(self.num_agents):
            agent = self.agents[agent_number]

            # Compute the critic loss
            target_actions = []
            for i in range(self.num_agents):
                i_agent = self.agents[i]
                i_action = i_agent.act(next_state[:, i, :], target=True, noise=0.0, train=True)
                target_actions.append(i_action.squeeze())
            target_actions = torch.stack(target_actions)
            target_actions = target_actions.permute(1, 0, 2).contiguous()

            with torch.no_grad():
                flat_next_state = next_state.view(-1, self.num_agents * self.state_size)
                flat_target_actions = target_actions.view(-1, self.num_agents * self.action_size)
                Q_targets_next = agent.target_critic(flat_next_state, flat_target_actions).squeeze()

            Q_targets = rewards[:, agent_number] + self.gamma * Q_targets_next * (1 - done[:, agent_number])

            flat_state = state.view(-1, self.num_agents * self.state_size)
            flat_action = action.view(-1, self.num_agents * self.action_size)
            Q_expected = agent.critic(flat_state, flat_action).squeeze()

            critic_loss = F.mse_loss(Q_targets, Q_expected)

            # Minimize the critic loss
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.gradient_clip_critic)
            agent.critic_optimizer.step()

            # Compute the actor loss
            Q_input = []
            for i in range(self.num_agents):
                i_agent = self.agents[i]
                Q_input.append(i_agent.actor(state[:, i, :]))
            Q_input = torch.stack(Q_input)
            Q_input = Q_input.permute(1, 0, 2).contiguous()
            flat_Q_input = Q_input.view(-1, self.num_agents * self.action_size)

            actor_loss = -agent.critic(flat_state, flat_Q_input).mean()

            # Minimize the actor loss
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.gradient_clip_actor)
            agent.actor_optimizer.step()

            # soft update target
            agent.soft_update(self.tau)

    def save(self, filename):
        """Saves the model networks to a file.

        :param filename:  Filename where to save the networks
        """
        checkpoint = {}

        for index, agent in enumerate(self.agents):
            checkpoint['actor_' + str(index)] = agent.actor.state_dict()
            checkpoint['target_actor_' + str(index)] = agent.target_actor.state_dict()
            checkpoint['critic_' + str(index)] = agent.critic.state_dict()
            checkpoint['target_critic_' + str(index)] = agent.target_critic.state_dict()

        torch.save(checkpoint, filename)

    def load(self, filename):
        """Loads the model networks from a file.

        :param filename: Filename from where to load the networks
        """
        checkpoint = torch.load(filename)

        for i in range(self.num_agents):
            agent = self.agents[i]

            agent.actor.load_state_dict(checkpoint['actor_' + str(i)])
            agent.target_actor.load_state_dict(checkpoint['target_actor_' + str(i)])
            agent.critic.load_state_dict(checkpoint['critic_' + str(i)])
            agent.target_critic.load_state_dict(checkpoint['target_critic_' + str(i)])
