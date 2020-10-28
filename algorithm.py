import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from random import sample
import torch.optim as optim
import numpy as np
import random

from utils import Net


# replay memory for DQN
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def memorize(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sampling(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, memory_size=50000,
                 batch_size=128,
                 gamma=0.99,
                 lr=1e-3, n_step=500000):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma

        # memory
        self.memory_size = memory_size
        self.Memory = ReplayMemory(self.memory_size)
        self.batch_size = batch_size

        # network
        self.target_net = Net().to(self.device)
        self.eval_net = Net().to(self.device)
        self.target_update()  # initialize same weight
        self.target_net.eval()

        # optim
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)

    def select_action(self, state, eps):
        prob = random.random()
        if prob > eps:
            return self.eval_net.act(state), False
        else:
            return (torch.tensor([[random.randrange(0, 9)]], device=self.device, dtype=torch.long,), True)

    def select_dummy_action(self, state):
        state = state.reshape(3, 3, 3)

        open_spots = state[:, :, 0].reshape(-1)

        p = open_spots / open_spots.sum()

        return np.random.choice(np.arange(9), p=p)

    def target_update(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        if self.Memory.__len__() < self.batch_size:
            return

        # random batch sampling
        transitions = self.Memory.sampling(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,)

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Q(s)
        Q_s = self.eval_net(state_batch).gather(1, action_batch)

        # maxQ(s') no grad for target_net
        Q_s_ = torch.zeros(self.batch_size, device=self.device)
        Q_s_[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()

        # Q_target=R+γ*maxQ(s')
        Q_target = reward_batch+(Q_s_ * self.gamma)

        # loss_fnc---(R+γ*maxQ(s'))-Q(s)
        # huber loss with delta=1
        loss = F.smooth_l1_loss(Q_s, Q_target.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def load_net(self, name):
        self.action_net = torch.load(name).cpu()

    def load_weight(self, name):
        self.eval_net.load_state_dict(torch.load(name))
        self.eval_net = self.eval_net.cpu()

    def act(self, state):
        with torch.no_grad():
            p = F.softmax(self.action_net.forward(state)).cpu().numpy()
            valid_moves = (state.cpu().numpy().reshape(
                3, 3, 3).argmax(axis=2).reshape(-1) == 0)
            p = valid_moves*p
            return p.argmax()
