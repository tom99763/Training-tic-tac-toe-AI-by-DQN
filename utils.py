import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

# transform to one hot code


def one_hot_board(state):
    player = state[1]
    if player == 'O':
        return np.eye(3)[list(state[0])].reshape(-1)
    if player == 'X':
        # permute for symmetry
        return np.eye(3)[list(state[0])][:, [0, 2, 1]].reshape(-1)


# state--27 features(one hot code)
# output---9 actions
class Net(nn.Module):
    def __init__(self, n_inputs=3 * 9, n_outputs=9):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs),
            nn.ReLU()
        )
        self.initialize()

    def initialize(self):
        for l in self.layer:
            try:
                torch.nn.init.normal_(l.weight, mean=0.0, std=0.1)
                torch.nn.init.constant_(l.bias, 0.01)
            except:
                pass

    def forward(self, x):
        x = self.layer(x)
        return x
    # choose best action

    def act(self, state):
        with torch.no_grad():
            return self.forward(state).max(1)[1].view(1, 1)
