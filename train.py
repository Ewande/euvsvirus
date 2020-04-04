from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

from environment import StudentEnv

env = StudentEnv()

model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model  # remove to demonstrate saving and loading

model = TRPO.load("a2c_cartpole")

obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()
