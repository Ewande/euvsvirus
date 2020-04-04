import gym
from stable_baselines import PPO2
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.policies import MlpPolicy
import environment
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from keras import Input, Model
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# def build_model(state_size, num_actions):
#   input = Input(shape=(1, state_size))
#   x = Flatten()(input)
#   x = Dense(16, activation='relu')(x)
#   x = Dense(16, activation='relu')(x)
#   x = Dense(16, activation='relu')(x)
#   output = Dense(num_actions, activation='linear')(x)
#   model = Model(inputs=input, outputs=output)
#   print(model.summary())
#   return model
#
# env = environment.StudentEnv(subjects_number=2)
# ENV_NAME = 'StudentEnv'
# observation = env.reset()
# model = build_model(observation.shape, 36)
#
# memory = SequentialMemory(limit=50000, window_length=1)
#
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)
#
# dqn = DQNAgent(model=model, nb_actions=36, memory=memory, nb_steps_warmup=10,
#                target_model_update=1e-2, policy=policy)
#
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#
#
# def build_callbacks(env_name):
#   checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
#   log_filename = 'dqn_{}_log.json'.format(env_name)
#   callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
#   callbacks += [FileLogger(log_filename, interval=100)]
#   return callbacks
#
# callbacks = build_callbacks(ENV_NAME)
#
# dqn.fit(env, nb_steps=50000,
# visualize=False,
# verbose=2,
# callbacks=callbacks)

# env = environment.StudentEnv(subjects_number=2)
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

# The algorithms require a vectorized environment to run
env = environment.StudentEnv(subjects_number=2)
model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
      break
