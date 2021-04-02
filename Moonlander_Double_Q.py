import math

import numpy as np
import torch
import gym
import tensorflow.compat.v1 as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
from collections import deque
import time
import pandas as pd


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
        self.env = gym.make('LunarLander-v2')
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
        return self.env.action_space.n

    def state_size(self):
        return self.env.observation_space.shape[0]


class DQN:
    def __init__(self, input_dim, num_actions):
        self.activation = 'relu'
        self.optimizer = Adam(learning_rate=0.0001)
        self.input_dim = input_dim
        self.num_actions = num_actions

    def init_model(self):
        model = Sequential([
            Dense(64, input_dim=self.input_dim, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.num_actions, activation="linear")
        ])
        model.compile(loss="mse", optimizer=self.optimizer)

        return model


def _reshape_state_for_net(state):
    return np.reshape(state, (1, 8))


class Agent:
    def __init__(self, env, batch_size, strategy, gamma, memory_size):
        self.state_size = env.state_size()  # number of factors in the state; e.g: velocity, position, etc
        self.action_size = env.num_actions_available()
        self.batch_size = batch_size
        self.env = env
        self.current_step = 0
        self.strategy = strategy
        self.replay_memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = 1

        # Build Policy Network
        self.policy_net = DQN(self.state_size, self.action_size).init_model()

        # Build Target Network
        self.target_net = DQN(self.state_size, self.action_size).init_model()

        self.update_target_net()

    # add new experience to the replay memory
    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def update_target_net(self):
        return self.target_net.set_weights(self.policy_net.get_weights())

    def choose_action(self, state):
        self.epsilon = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if np.random.uniform(0.0, 1.0) < self.epsilon:  # exploration
            action = np.random.choice(self.action_size)
        else:
            state = np.reshape(state, [1, self.state_size])
            qhat = self.policy_net.predict(state)  # output Q(s,a) for all a of current state
            action = np.argmax(qhat[0])  # because the output is m * n, so we need to consider the dimension [0]

        return action

    # update params in NN
    def learn_normal_q(self):

        # take a mini-batch from replay experience
        cur_batch_size = min(len(self.replay_memory), self.batch_size)
        mini_batch = random.sample(self.replay_memory, cur_batch_size)

        # batch data
        sample_states = np.ndarray(shape=(cur_batch_size, self.state_size))  # replace 128 with cur_batch_size
        sample_actions = np.ndarray(shape=(cur_batch_size, 1))
        sample_rewards = np.ndarray(shape=(cur_batch_size, 1))
        sample_next_states = np.ndarray(shape=(cur_batch_size, self.state_size))
        sample_dones = np.ndarray(shape=(cur_batch_size, 1))

        temp = 0
        for exp in mini_batch:
            sample_states[temp] = exp[0]
            sample_actions[temp] = exp[1]
            sample_rewards[temp] = exp[2]
            sample_next_states[temp] = exp[3]
            sample_dones[temp] = exp[4]
            temp += 1

        sample_qhat_next = self.target_net.predict(sample_next_states)

        # set all Q values terminal states to 0
        sample_qhat_next = sample_qhat_next * (np.ones(shape=sample_dones.shape) - sample_dones)
        # choose max action for each state
        sample_qhat_next = np.max(sample_qhat_next, axis=1)

        sample_qhat = self.policy_net.predict(sample_next_states)

        for i in range(cur_batch_size):
            a = sample_actions[i, 0]

            sample_qhat[i, int(a)] = sample_rewards[i] + self.gamma * sample_qhat_next[i]

        q_target = sample_qhat

        self.policy_net.fit(sample_states, q_target, epochs=1, verbose=0)

    # update params in NN
    def learn_double_q(self):

        # take a mini-batch from replay experience
        cur_batch_size = min(len(self.replay_memory), self.batch_size)
        minibatch = random.sample(self.replay_memory, cur_batch_size)
        minibatch_new_q_values = []

        for experience in minibatch:
            state, action, reward, next_state, done = experience
            state = _reshape_state_for_net(state)
            experience_new_q_values = self.policy_net.predict(state)[0]
            if done:
                q_update = reward
            else:
                next_state = _reshape_state_for_net(next_state)
                # using online network to SELECT action
                online_net_selected_action = np.argmax(self.policy_net.predict(next_state))
                # using target network to EVALUATE action
                target_net_evaluated_q_value = self.target_net.predict(next_state)[0][online_net_selected_action]
                q_update = reward + self.gamma * target_net_evaluated_q_value
            experience_new_q_values[action] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.array([e[0] for e in minibatch])
        minibatch_new_q_values = np.array(minibatch_new_q_values)
        self.policy_net.fit(minibatch_states, minibatch_new_q_values, verbose=False, epochs=1)


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)


def main():
    batch_size = 32
    gamma = 0.99
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 1
    memory_size = 1000000

    em = MoonlanderEnvManager()
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    agent = Agent(em, batch_size, strategy, gamma, memory_size)

    # load model
    # agent.brain_policy.set_weights(tf.keras.models.load_model('C:/Users/nhunh/.spyder-py3/Model1.h5').get_weights())

    rewards = []
    aver_reward = []
    aver = deque(maxlen=100)

    for episode in range(500):
        state = em.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = em.step(action)

            # env.render()

            total_reward += reward

            agent.add_to_replay_memory(state, action, reward, next_state, done)
            agent.learn_double_q()

            state = next_state

        aver.append(total_reward)
        aver_reward.append(np.mean(aver))

        rewards.append(total_reward)
        plot(rewards, aver_reward, episode == 499)

        # save data
        df = pd.DataFrame(rewards)
        df2 = pd.DataFrame(aver_reward)
        df.to_csv('data/rewards.csv')
        df2.to_csv('data/aver_rewards.csv')

        agent.update_target_net()
        # agent.epsilon = max(0.1, 0.995 * agent.epsilon)  # decaying exploration

        # update model_target after each episode
        # if episode % target_update == 0:

        print("Episode ", episode, total_reward)


if __name__ == '__main__':
    main()
