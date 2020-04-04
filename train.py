import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import environment
import tensorflow

# The algorithms require a vectorized environment to run
env = environment.StudentEnv(subjects_number=2)
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()