import os
import sys

import gym
import math
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot(reward, moving_avg_period, episode):
    clear = lambda: os.system('clear')
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward (all agents)')
    plt.plot(reward)
    # plt.pause(0.001)

    moving_avg = get_moving_average(moving_avg_period, reward)
    plt.plot(moving_avg)
    plt.pause(0.001)
    clear()
    print("Episode", len(reward), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])

    if episode == 5000:
        plt.savefig("5000")
    if episode == 10000:
        plt.savefig("10000")


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        # print(states[0])
        return policy_net.forward(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states, is_final):
        final_state_locations = is_final  # next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net.forward(non_final_states).max(dim=1)[0].detach()
        # print(type(values), values.shape, values)
        return values


'''
Extract tensors from experience tuple
'''
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'is_final'))


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    t5 = torch.cat(batch.is_final)

    return (t1, t2, t3, t4, t5)


'''
Environment class. Handles all important functions. 
'''


class CombatEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('ma_gym:Combat-v0')  # .unwrapped
        self.env.reset()
        self.done = [False for _ in range(self.env.n_agents)]

    def reset(self):
        self.done = [False for _ in range(self.env.n_agents)]
        return torch.tensor([self.env.reset()], device=self.device, requires_grad=True)

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return 10  # self.env.action_space[0].n

    def take_action(self, action):
        obs, reward, self.done, _ = self.env.step(action)
        return torch.tensor(reward, device=self.device), torch.tensor([obs], device=self.device)

    def get_single_state(self, agent):
        return format(self.env.observation_space[agent])

    def get_state(self, ):
        return self.env.observation_space

    def get_input_size(self):
        return self.env.observation_space[0].shape[0]


'''
Agent class.
'''


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax().to(self.device)  # exploit


'''
Epsilon greedy strategy, handles the exploration rate.
'''


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)


'''
Class that holds the memory of every [state, action, reward, next state] 
'''


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


'''
This class creates the deep neural network. It also handles the forward pass. 
All pytorch NNs require an implementation of forward. 
'''


class DQN(nn.Module):
    def __init__(self, observation_space_size):
        super().__init__()

        self.fc1 = nn.Linear(in_features=observation_space_size,
                             out_features=128)  # input size == size of observation_space
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        # self.fc3 = nn.Linear(in_features=32, out_features=20)
        self.out = nn.Linear(in_features=128, out_features=10)  # 10 output nodes since we have 10 actions

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        # t = F.relu(self.fc3(t))
        t = (self.out(t))
        return t


def train():
    batch_size = 64
    gamma = 0.99
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 1
    memory_size = 100000
    lr = 0.001
    num_episodes = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = CombatEnvManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    policy_net = DQN(em.get_input_size()).to(device)
    target_net = DQN(em.get_input_size()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
    total_reward = []
    win_count = 0
    for episode in range(num_episodes):
        state = em.reset()
        ep_reward = 0
        while not all(em.done):
            actions = []
            # Extract actions for all agents
            for a in range(em.env.n_agents):
                action = agent.select_action(state[0, a], policy_net)
                actions.append(action.item())
            reward, next_state = em.take_action(actions)
            ep_reward += sum(reward.data)
            is_final = torch.tensor(all(em.done)).unsqueeze(0)
            # push all experiences to replay memory
            for i in range(em.env.n_agents):
                state_n = state[0, i].unsqueeze(0)
                next_state_n = next_state[0, i].unsqueeze(0)
                reward_n = torch.unsqueeze(reward, 1)[i]
                action_n = torch.tensor(actions[i]).unsqueeze(0)
                # print(f' reward: {reward_n}\n state: {state_n}\n state next: {next_state_n}\n')
                memory.push(Experience(state_n, action_n, next_state_n, reward_n, is_final))
            state = next_state

            # if the batch size is large enough, extract batch and train network
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states, is_final = extract_tensors(experiences)
                # states.requires_grad = True
                # states.retain_grad()
                # print(states.requires_grad)

                current_q_values = QValues.get_current(policy_net, states, actions)
                current_q_values.retain_grad()
                next_q_values = QValues.get_next(target_net, next_states, is_final)
                # next_q_values.retain_grad()
                target_q_values = (next_q_values * gamma) + rewards
                # target_q_values.retain_grad()
                # print(current_q_values[0])

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                loss.retain_grad()
                optimizer.zero_grad()
                loss.backward()
                # print(f' gradient: {states.grad}')
                optimizer.step()

        total_reward.append(ep_reward)
        plot(total_reward, 200, episode)
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    em.close()


if __name__ == '__main__':
    train()
