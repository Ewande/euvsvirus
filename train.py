from pathlib import Path

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

from environment import StudentEnv
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = StudentEnv(subjects_number=2)
if Path('ppo2.zip').exists():
    model = PPO2.load('ppo2')
else:
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    # model.save('ppo2')

obs = env.reset()
for i in range(20000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break
    print(i)


