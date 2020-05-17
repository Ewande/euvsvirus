from pathlib import Path

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, TRPO


from environment import StudentEnv, StudentEnvBypass
from reporting import setup_logging
import numpy as np
import sys
import copy

USE_RANDOM = False
HEARING = False
VISUALS = not HEARING
# VISUALS = False
BIAS_HIGHEST_PROB = 0.6

env = StudentEnv(num_subjects=3)
# if Path('ppo2.zip').exists():
#     model = PPO2.load('ppo2')
if Path('ppo2_test.zip').exists():
    model = PPO2.load('ppo2_best')
else:
    model = PPO2(MlpPolicy, env, verbose=0, gamma=0.8, n_steps=128, policy_kwargs={'layers': [32, 32, 32]})
    model.learn(total_timesteps=50000)
    model.save('ppo2_new')

print(f'get size of: {sys.getsizeof(model)}')
if USE_RANDOM:
    setup_logging('application_random.log')
else:
    if HEARING:
        setup_logging('application_hearing.log', mode='w+')
    elif VISUALS:
        setup_logging('application_visuals.log', mode='w+')
    else:
        setup_logging('application_pp02_rand.log')

steps_list = []
for ep in range(100):
    obs = env.reset()
    if HEARING:
        excellence_skills = np.tile(
            np.concatenate(([np.random.normal(7, 0.5)], np.random.normal(1, 0.3, 2))), (3, 1))
        excellence_skills += np.random.normal(0, 0.2, size=(3, 3))
        excellence_skills = np.maximum(
            excellence_skills,
            np.full(shape=(3, 3), fill_value=0.1)
        )
        env.mean_skill_gains = excellence_skills
    elif VISUALS:
        excellence_skills = np.tile(
            np.concatenate((np.random.normal(1, 0.3, 2), [np.random.normal(7, 0.5)])), (3, 1))
        excellence_skills += np.random.normal(0, 0.2, size=(3, 3))
        excellence_skills = np.maximum(
            excellence_skills,
            np.full(shape=(3, 3), fill_value=0.1)
        )
        env.mean_skill_gains = excellence_skills
    env_list = []
    for i in range(2000):

        if USE_RANDOM:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs)
        if i <= 120:
            env_list.append(copy.deepcopy(env))
        else:
            env_list.pop(0)
            env_list.append(copy.deepcopy(env))
        obs, rewards, done, info = env.step(action)
        print(i)
        env.render()
        if done:
            steps_list.append(i)
            print(f'Mean skill gains: {env.mean_skill_gains}')
            break

    if VISUALS:
        bias = [[BIAS_HIGHEST_PROB, (1-BIAS_HIGHEST_PROB)/2, (1-BIAS_HIGHEST_PROB)/2],
                [(1-BIAS_HIGHEST_PROB)/2, BIAS_HIGHEST_PROB, (1-BIAS_HIGHEST_PROB)/2]]
    elif HEARING:
        bias = [[(1-BIAS_HIGHEST_PROB)/2, (1-BIAS_HIGHEST_PROB)/2, BIAS_HIGHEST_PROB],
                [(1-BIAS_HIGHEST_PROB)/2, BIAS_HIGHEST_PROB, (1-BIAS_HIGHEST_PROB)/2]]
    else:
        continue

    for b in bias:
        env_b = StudentEnvBypass(env_list[-100], b)
        obs = env_b.last_scores
        for i in range(2000):
            if USE_RANDOM:
                action = env_b.action_space.sample()
            else:
                action, _states = model.predict(obs)

            obs, rewards, done, info = env_b.step(action)
            print(env_b.step_num)
            env_b.render()
            if done:
                print(f'Mean skill gains: {env_b.mean_skill_gains}')
                break

print(f'number of steps in each episodes: {steps_list}')
