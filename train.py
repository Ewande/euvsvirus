from pathlib import Path

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

from environment import StudentEnv

env = StudentEnv(subjects_number=2)
if Path('ppo2.zip').exists():
    model = PPO2.load('ppo2')
else:
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save('ppo2')

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break
    print(i)
    env.render()

