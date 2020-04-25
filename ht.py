import json
import logging

import numpy as np
from sklearn.model_selection import ParameterGrid
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from environment import StudentEnv
from main import _run_episode
from reporting import setup_logging


def main():
    setup_logging('ht.log', level=logging.WARNING)
    env = StudentEnv(3, 3, 3)
    grid = ParameterGrid({
        'gamma': [0.9, 0.95, 0.97, 0.99, 0.995],
        'n_steps': [8, 32, 128, 512, 2048],
        'timesteps': [10000, 50000, 100000, 200000, 300000],
        'l1': [32, 64, 128, 256],
        'l2': [32, 64, 128, 256],
        'l3': [32, 64, 128, 256],
    })
    for params in grid:
        model = PPO2(MlpPolicy, env, verbose=1,
                     gamma=params['gamma'],
                     n_steps=params['n_steps'],
                     policy_kwargs={'layers': [params['l1'], params['l2'], params['l3']]})
        model.learn(total_timesteps=params['timesteps'])

        episodes = []
        for ep in range(200):
            print(f'{ep+1}/200')
            ep_len = _run_episode(model, env, 20000)
            episodes.append(ep_len)

        logging.warning(
            json.dumps({
                **params,
                'mean_ep_len': np.mean(episodes),
                'std_ep_len': np.std(episodes),
            }))


if __name__ == '__main__':
    main()
