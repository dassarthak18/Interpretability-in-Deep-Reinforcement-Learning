import gym
import numpy as np

from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.agents import SARSAAgent
from rl.policy import *
from rl.memory import EpisodeParameterMemory, SequentialMemory

# Default policy EpsGreedyQPolicy() (i = 1)
policies = [SoftmaxPolicy(), EpsGreedyQPolicy(), GreedyQPolicy(), BoltzmannQPolicy(), MaxBoltzmannQPolicy(), BoltzmannGumbelQPolicy()]

def cem(env,steps=100000):
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(env.action_space.n))
       model.add(Activation('softmax'))
       cem = CEMAgent(model=model, nb_actions=env.action_space.n, memory=EpisodeParameterMemory(limit=50000, window_length=1), batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
       cem.compile()
       train_history = cem.fit(env, nb_steps=steps, visualize=False, verbose=1)
       train_rewards = train_history.history['episode_reward']
       return model, cem, train_rewards

def dqn(env,steps=50000,i=1,tau=1):
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(env.action_space.n))
       model.add(Activation('linear'))
       if i == 3:
              policy = BoltzmannQPolicy(tau)
       else:
              policy = policies[i]
       dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=SequentialMemory(limit=50000, window_length=1), nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
       dqn.compile(Adam(learning_rate=1e-3), metrics=['mse'])
       train_history = dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)
       train_rewards = train_history.history['episode_reward']
       return model, dqn, train_rewards

def duel_dqn(env,steps=50000,i=1):
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(env.action_space.n))
       model.add(Activation('linear'))
       dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=SequentialMemory(limit=50000, window_length=1), nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policies[i])
       dqn.compile(Adam(learning_rate=1e-3), metrics=['mse'])
       train_history = dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)
       train_rewards = train_history.history['episode_reward']
       return model, dqn, train_rewards

def sarsa(env,steps=50000,i=1):
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(env.action_space.n))
       model.add(Activation('linear'))
       sarsa = SARSAAgent(model=model, nb_actions=env.action_space.n, nb_steps_warmup=100, policy=policies[i])
       sarsa.compile(Adam(learning_rate=1e-3), metrics=['mse'])
       train_history = sarsa.fit(env, nb_steps=steps, visualize=False, verbose=1)
       train_rewards = train_history.history['episode_reward']
       return model, sarsa, train_rewards

def plot_training(train_rewards):
       X = np.arange(len(train_rewards)) + 1
       Y = train_rewards
       plt.plot(X,Y,marker='o')
       plt.xlabel('Episode Number')
       plt.ylabel('Cumulative Reward')
       plt.title('Training History')
       return plt
