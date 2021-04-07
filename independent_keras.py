import itertools
import math
import time
import os
import numpy as np
import gym
import keras
import tensorflow.compat.v1 as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
from collections import deque
import pandas as pd


clear = lambda: os.system('clear')
tf.disable_v2_behavior()  # testing on tensorflow 1
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


def plot(values, values2, save):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.plot(values, label='reward')
    plt.plot(values2, label='avg reward')
    plt.legend(loc="upper left")
    plt.pause(0.001)
    if save:
        plt.savefig('pic')


class MoonlanderEnvManager:
    def __init__(self):
        self.env = gym.make('ma_gym:Combat-v0')
        self.env.reset()
        self.done = False

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space[0].n

    def state_size(self):
        return self.env.observation_space[0].shape[0]


class DQN:
    def __init__(self, input_dim, num_actions):
        self.activation = 'relu'
        self.optimizer = Adam(learning_rate=0.0001)
        self.input_dim = input_dim
        self.num_actions = num_actions

    def init_shallow_model(self):
        model = Sequential([
            Dense(128, input_dim=self.input_dim, activation="relu"),
            Dense(128, activation="relu"),
            Dense(self.num_actions, activation="linear")
        ])
        model.compile(loss="mse", optimizer=self.optimizer)

        return model

    def init_deep_model(self):
        model = Sequential([
            Dense(64, input_dim=self.input_dim, activation="relu"),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.num_actions, activation="linear")
        ])
        model.compile(loss="mse", optimizer=self.optimizer)

        return model


def _reshape_state_for_net(state):
    return np.reshape(state, (1, 8))


class Agent:
    def __init__(self, env, batch_size, gamma, memory_size, network, model, name):
        self.state_size = env.state_size()  # number of factors in the state; e.g: velocity, position, etc
        self.action_size = env.num_actions_available()
        self.batch_size = batch_size
        self.env = env
        self.current_step = 0
        self.replay_memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = 1

        if network == 'shallow' and not model:
            self.policy_net = DQN(self.state_size, self.action_size).init_shallow_model()
            self.target_net = DQN(self.state_size, self.action_size).init_shallow_model()
        elif model == 'deep' and not model:
            self.policy_net = DQN(self.state_size, self.action_size).init_deep_model()
            self.target_net = DQN(self.state_size, self.action_size).init_deep_model()
        else:
            self.policy_net = keras.models.load_model(f'models/{name}.h5')
            self.target_net = keras.models.load_model(f'models/{name}.h5')

        self.update_target_net()

    # add new experience to the replay memory
    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def update_target_net(self):
        return self.target_net.set_weights(self.policy_net.get_weights())

    def choose_action(self, state):
        if np.random.uniform(0.0, 1.0) < self.epsilon:  # exploration
            action = np.random.choice(self.action_size)
        else:
            state = np.reshape(state, [1, self.state_size])
            qhat = self.policy_net.predict(state) 
            action = np.argmax(qhat[0])
        return action

    def learn_normal_q(self):

        # take mini-batch from replay memory
        cur_batch_size = min(len(self.replay_memory), self.batch_size)
        mini_batch = random.sample(self.replay_memory, cur_batch_size)

        states = np.ndarray(shape=(cur_batch_size, self.state_size))  # replace 128 with cur_batch_size
        actions = np.ndarray(shape=(cur_batch_size, 1))
        rewards = np.ndarray(shape=(cur_batch_size, 1))
        next_states = np.ndarray(shape=(cur_batch_size, self.state_size))
        dones = np.ndarray(shape=(cur_batch_size, 1))

        idx = 0
        for exp in mini_batch:
            states[idx] = exp[0]
            actions[idx] = exp[1]
            rewards[idx] = exp[2]
            next_states[idx] = exp[3]
            dones[idx] = exp[4]
            idx += 1

        qhat_next = self.policy_net.predict(next_states)

        qhat_next = qhat_next * (np.ones(shape=dones.shape) - dones)

        qhat_next = np.max(qhat_next, axis=1)

        qhat = self.policy_net.predict(states)

        for i in range(cur_batch_size):
            a = actions[i, 0]
            qhat[i, int(a)] = rewards[i] + self.gamma * qhat_next[i]

        q_target = qhat

        self.policy_net.fit(states, q_target, epochs=1, verbose=0)

def epsilon_greedy(agent):
    agent.epsilon = max(0.1, 0.995 * agent.epsilon)

def save_data(experiment, rewards, aver_reward):
    name = str((experiment[0])) + '_' + str((experiment[1])) + '_' + str((experiment[2]))
    df = pd.DataFrame(list(zip(rewards, aver_reward)))
    df.columns = ['Rewards', 'Average reward']
    df.to_csv(f'data/{name}.csv')


def main():

    batch_size = [32, 64]
    gamma = 0.99
    target_update = [1, 10]
    network = ['shallow']
    memory_size = 1000000
    episodes = 5000

    cnt = 1
    for experiment in itertools.product(batch_size, target_update, network):
        name = str((experiment[0])) + '_' + str((experiment[1])) + '_' + str((experiment[2]) + 'double_q')
        em = MoonlanderEnvManager()
        agent = Agent(em, experiment[0], gamma, memory_size, experiment[2], False, name)
        time_start = time.clock()
        # run your code
        rewards = []
        aver_reward = []
        aver = deque(maxlen=100)

        for episode in range(episodes):
            state = em.reset()
            total_reward = 0
            done = [False for all in range(5)]

            while not all(done):
                actions = []
                for i in range(em.env.n_agents):
                    actions.append(agent.choose_action(state[i]))

                next_state, reward, done, info = em.step(actions)

                # em.render()

                total_reward += sum(reward)
                for i in range(em.env.n_agents):
                  agent.add_to_replay_memory(state[i], actions[i], reward[i], next_state[i], done[i])
                agent.learn_normal_q()

                state = next_state

            aver.append(total_reward)
            aver_reward.append(np.mean(aver))

            rewards.append(total_reward)
            # plot(rewards, aver_reward, episode == 499)

            # print info about experiments
            print(f'experiment: {cnt}/18 {name}\n'
                  f'Episode: {episode}    {round(episode/episodes, 1)*100}%\n'
                  f'Final Avg Reward: {round(np.mean(aver),1)}\n'
                  f'Computation Time: {round((time.clock() - time_start)/ 60, 1)} minutes')
            clear()

            # update target net
            if episode % experiment[1] == 0:
                agent.update_target_net()

            # set new epsilon    
            epsilon_greedy(agent)    
        cnt += 1
        save_data(experiment, rewards, aver_reward)
        # agent.policy_net.save(f'models/{name}.h5')


if __name__ == '__main__':
    main()