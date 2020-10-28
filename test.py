from gym_tictactoe.env import TicTacToeEnv
from utils import one_hot_board, Net
from algorithm import DQN
import torch


def run():
    done = False
    obs = env.reset()
    env.render()
    while not done:
        if obs[1] == 'X':
            action = int(input())
            obs_, _, done, info = env.step(action)
        else:
            obs = one_hot_board(env)
            action = model.act(torch.tensor([obs], dtype=torch.float)).item()
            obs_, _, done, exp = env.step(action)
        obs = obs_

        env.render()


if __name__ == "__main__":
    env = TicTacToeEnv()
    model = DQN()
    model.load_net('net.pkl')
    run()
