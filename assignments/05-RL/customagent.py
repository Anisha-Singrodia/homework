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
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
gamma = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network


class Agent:
    """
    agent
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        # Define the hyperparameters
        alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate of exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate

        self.action_space = action_space
        self.observation_space = observation_space
        self.q_network = QNetwork(8, 4)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)

        self.state_size = 8
        self.action_size = 4
        self.seed = random.seed(10)

        # Q-Network
        self.qnetwork_local = QNetwork(8, 4).to(device="cpu")
        self.qnetwork_target = QNetwork(8, 4).to(device="cpu")
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)

        # Replay memory
        self.memory = ReplayBuffer(4, BUFFER_SIZE, BATCH_SIZE, 10)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        act
        """
        state = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        self.curr_action = action
        self.curr_obs = observation
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
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
        states = self.curr_obs
        actions = self.curr_action
        dones = terminated or truncated
        next_states = observation
        rewards = reward

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        # print(self.qnetwork_target(torch.tensor(next_states)))
        q_targets_next = max(self.qnetwork_target(torch.tensor(next_states)))
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(torch.tensor(states))[actions]

        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


# Define the neural network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(10)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# Define the replay memory
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
