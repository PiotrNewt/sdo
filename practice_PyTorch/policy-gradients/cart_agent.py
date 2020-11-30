import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Bernoulli


# class pgn is a nn for state --> action (classification)
class PGN(nn.Module):
    def __init__(self):
        super(PGN, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = torch.sigmoid(self.fc3(out))
        return out


# class cart_agent need implement what policy gradient can do
class CartAgent(object):
    def __init__(self, lr, gamma):
        self.pgn = PGN()
        self.gamma = gamma
        self._init_memory()
        self.optimizer = torch.optim.Rprop(self.pgn.parameters(), lr=lr)

    def _init_memory(self):
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.steps = 0

    def memorize(self, state, action, reward):
        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)
        self.steps += 1

    def _adjust_reward(self):
        # backward weight
        running_add = 0
        for i in reversed(range(self.steps)):
            if self.reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + self.reward_pool[i]
                self.reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(self.reward_pool)
        reward_std = np.std(self.reward_pool)
        for i in range(self.steps):
            self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std

    def leaning(self):
        self._adjust_reward()

        # policy gradient
        self.optimizer.zero_grad()
        # -- loss backward start --
        for i in range(self.steps):
            state = self.state_pool[i]
            action = torch.FloatTensor([self.action_pool[i]])
            reward = self.reward_pool[i]

            probs = self.act(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward
            loss.backward()
        # -- loss backward end --
        self.optimizer.step()
        self._init_memory()

    # state --> action
    def act(self, state):
        return self.pgn(state)