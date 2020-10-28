from algorithm import DQN
from gym_tictactoe.env import TicTacToeEnv
import torch
import numpy as np
import random
from utils import Net,one_hot_board



def run():
    env = TicTacToeEnv()
    state = torch.tensor([one_hot_board(env.reset())], dtype=torch.float).to(model.device)
    for step in range(n_step):
        t = np.clip(step / eps_steps, 0, 1)
        eps = (1 - t) * eps_start + t * eps_end
        print('\r'+f'step----{step},epsilon----{eps}', flush=True, end='')
        action, was_random = model.select_action(state, eps)
        next_state, reward, done, _ = env.step(action.item())
        next_state=one_hot_board(next_state)
        
        if not done:
            next_state, _, done,  _ = env.step(
                model.select_dummy_action(next_state))
            next_state=one_hot_board(next_state)
            next_state = torch.tensor(
                [next_state], dtype=torch.float).to(model.device)

        if done:

            next_state = None

        model.Memory.memorize(state, action, next_state,
                              torch.tensor([reward], device=model.device))

        state = next_state

        model.learn()

        if done:
            state = torch.tensor(
                [one_hot_board(env.reset())], dtype=torch.float).to(model.device)

        if step % target_update == 0:
            model.target_update()


if __name__ == "__main__":
    # hyperparameters
    batch_size = 128
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.1
    eps_steps = 200000
    n_step = 500000
    target_update = (1e-2) * n_step

    # model
    model = DQN(batch_size=batch_size, n_step=n_step, gamma=gamma)

    run()
    torch.save(model.eval_net, 'tic_tac_toe_net.pkl')
