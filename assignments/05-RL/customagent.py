import gymnasium as gym
import random

import numpy as np
from collections import deque

import torch.nn as nn
import torch
import torch.optim as optim

# import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import deque, namedtuple


device = "cpu"
# BUFFER_SIZE = int(1e5)  # replay buffer size
BUFFER_SIZE = 5000
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 3e-4  # learning rate
# [226932, 63751, 134, 75381, 59013, 46209, 295641, 457141, 168622]
# [171192, 219784, 124087, 114731, 160432, 47441, 293429, 309208]
UPDATE_EVERY = 4  # how often to update the network
# 283642
# 217976
# 74921
# 65091 with 5000 buffer
# 44742 with 5000 buffer
# 45653


class Agent:
    """
    agent
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box, LR
    ):
        # Define the hyperparameters
        self.epsilon = 1  # Exploration rate
        self.epsilon_decay = 0.999  # Decay rate of exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        # self.epsilon_min = 0.05  # Minimum exploration rate

        self.action_space = action_space
        self.observation_space = observation_space

        self.state_size = 8
        self.action_size = 4
        seed = 42
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(8, 4, seed).to(device="cpu")
        self.qnetwork_target = QNetwork(8, 4, seed).to(device="cpu")
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.timestep = 0

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        act
        """

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            state = torch.from_numpy(observation).float().unsqueeze(0).to(device)
            # set the network into evaluation mode
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            # Back to training mode
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.randint(self.action_size)
        self.curr_action = action
        self.curr_obs = observation
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        learn
        """
        # Obtain random minibatch of tuples from D
        state = torch.tensor(self.curr_obs)
        action = self.curr_action
        done = False
        if terminated or truncated:
            done = True
        next_state = torch.tensor(observation)
        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                sampled_experiences = self.memory.sample()
                self.learn_after_step(sampled_experiences)

    def learn_after_step(self, experiences):
        """
        Learn from experience by training the q_network

        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        states, actions, rewards, next_states, dones = experiences
        # Get the action with max Q value
        action_values = self.qnetwork_target(next_states).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * max_action_values * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        # loss = F.smooth_l1_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()

        # Update fixed weights
        self.update_fixed_network(self.qnetwork_local, self.qnetwork_target)

    def update_fixed_network(self, q_network, fixed_network):
        """
        Update fixed network by copying weights from Q network using TAU param

        Parameters
        ----------
        q_network (PyTorch model): Q network
        fixed_network (PyTorch model): Fixed target network
        """
        for source_parameters, target_parameters in zip(
            q_network.parameters(), fixed_network.parameters()
        ):
            target_parameters.data.copy_(
                TAU * source_parameters.data + (1.0 - TAU) * target_parameters.data
            )


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        Build a fully connected neural network

        Parameters
        ----------
        state_size (int): State dimension
        action_size (int): Action dimension
        seed (int): random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Replay memory allow agent to record experiences and learn from them

        Parametes
        ---------
        buffer_size (int): maximum size of internal memory
        batch_size (int): sample size from experience
        seed (int): random seed
        """
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.state
                        for experience in experiences
                        if experience is not None
                    ]
                )
            )
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.action
                        for experience in experiences
                        if experience is not None
                    ]
                )
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.reward
                        for experience in experiences
                        if experience is not None
                    ]
                )
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.next_state
                        for experience in experiences
                        if experience is not None
                    ]
                )
            )
            .float()
            .to(device)
        )
        # Convert done from boolean to int
        dones = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.done
                        for experience in experiences
                        if experience is not None
                    ]
                ).astype(np.uint8)
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
