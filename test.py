from gym_tictactoe.env import TicTacToeEnv
from utils import one_hot_board, Net
from algorithm import DQN
import torch

from gym_tictactoe.env import TicTacToeEnv
from utils import one_hot_board, Net
from algorithm import DQN
import torch


def run(player):
    done = False
    obs = env.reset()
    env.render()
    while not done:
        if obs[1] == player:
            action = int(input())
            obs_, _, done, info = env.step(action)
        else:
            obs = one_hot_board(obs)
            action = model.act(torch.tensor([obs], dtype=torch.float)).item()
            obs_, _, done, exp = env.step(action)
        obs = obs_

        env.render()


if __name__ == "__main__":
    env = TicTacToeEnv()
    model = DQN()
    model.load_net('net.pkl')
    print('choose O or X player')
    player=str(input(''))
    assert player=='O' or player=='X'
    run(player)
