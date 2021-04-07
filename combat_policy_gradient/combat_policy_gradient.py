import numpy as np
import gym 
import time
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_fields = ("o", "a", "r", "o_n", "mask", "R", "agent_health", "opp_health")
Batch = namedtuple("Batch", batch_fields)
def make_batch():
    return Batch(*([] for f in batch_fields))


class CombatEnvironmentManager:
    def __init__(self, dummy_env = False, one_hot = True, random = False):
        '''
        If dummy_env is true, modify all episodes to learn a simple
        relationship from observation to reward. This way we can verify
        that learning is working correctly
        '''
        self.batch = make_batch()
        self.dummy_env = dummy_env
        self.one_hot = one_hot
        self.random = random
    def clear_batch(self):
        self.batch = make_batch()
    def append_episode(self, episode):
        ''' Append an episode to the batch
        '''
        # Simply append each field
        for field in episode._fields:
            getattr(self.batch, field).append(
                getattr(episode, field)
            )
    def get_episode(self, policy):
        '''
        Run a single episode, return it as a batch of shape:
        (n_steps x n_agents)
        
        Already accumulate rewards and set masks based on done status
        '''
        # Will hold the new episode
        episode = make_batch()
        
        ### Run the episode ###
        env = gym.make('ma_gym:Combat-v0')
        obs_n = env.reset()
        if self.one_hot:
            obs_n = self.one_hot_obs(obs_n)
        n_agents = len(obs_n)
        dones = []
        opp_hp = []
        for i in range(40):
            obs_torch = torch.Tensor(obs_n)
            probs, _ = policy(obs_torch, None)
            action_dist = Categorical(probs)
            act_n = action_dist.sample().detach().numpy()
            if self.random:
                act_n = env.action_space.sample()
            next_obs_n, reward_n, done_n, info = env.step(act_n)
            if self.one_hot:
                next_obs_n = self.one_hot_obs(next_obs_n)
            # The done_n from the environment only indicates if all
            # agents are done. So we determine for each individual agent
            # if they are done based on the hp value.
            done_n = [info['health'][i] == 0 for i in range(n_agents)]
            episode.o.append(obs_n)
            episode.o_n.append(next_obs_n)
            episode.a.append(act_n)
            episode.r.append(reward_n)
            dones.append(done_n)
            opp_hp.append(sum(env.opp_health.values()))
            obs_n = next_obs_n
        env.close()
        
        ### Process the batch ###
        # Calculate the accumulated reward. This is identical for all agents
        
        # First, we must determine the win/loss condition
        # The agents have won if one of them is alive at the end of the episode
        player_alive = any(not d for d in done_n) 
        opponent_alive = (opp_hp[-1] > 0)
        
        win_bonus = 1.0 * (player_alive - opponent_alive)
        # Then, accumulate the opponent hp from end to start
        opp_hp_accumulated = np.cumsum(opp_hp[::-1])[::-1]
        future_reward = win_bonus - 0.1 * opp_hp_accumulated
        #print(accumulated_reward)
        R = np.tile(future_reward, (n_agents, 1)).T
        dones.insert(0, dones[0])
        mask = 1 - np.array(dones[:-1])
        
        # We need the future reward in the right shape
        episode = episode._replace(R = R, mask = mask)
        
        return episode

    def generate_batch(self, policy, num_episodes):
        ''' Get a batch of episodes sampled using the given policy
        
        '''
        ### Generate episodes ###
        for i in range(num_episodes):
            self.append_episode(self.get_episode(policy))
        
        torch_batch = self.get_pg_batch()
        self.clear_batch()
        return torch_batch
    def get_pg_batch(self):
        '''Get the stored batch as a set of masked tensors for use with
        on-policy methods
        Returns: a Batch() with as fields:
          torch.Tensor()s with shape (n_episodes x n_steps x n_agents x <field dim>)
        '''

        torch_batch = Batch(**{
                field: torch.tensor(data)
            for field, data in self.batch._asdict().items()})
        
        return torch_batch
    def one_hot_obs(self, obs):
        ''' Pre-process a (batch of) observations to make it adhere to the expected 
        representation.
        
        The ma-gym implementation environment has some undocumented deviates from the 
        original lua implementation, which gives a one-hot representation. 
        Input: torch batch with observations in float representation (last dim 150)
        Output: torch batch with observations in one-hot representation (last dim 280)
        '''
        obs = torch.Tensor(obs)
        is_agent = (obs[...,0:25] == 1).long()
        is_opponent = (obs[...,0:25] == -1).long()

        # Construct mask for empty locations. For those, the observation should be empty. 
        # Mask will have shape (... x 25)
        empty_mask = (is_agent + is_opponent)

        agent_id = (F.one_hot(obs[...,25:50].long(), num_classes=5) * empty_mask.unsqueeze(-1)).flatten(-2)

        # Health as integer 0...3
        health = obs[..., 50:75]

        health_2 = health >= 2
        health_3 = health >= 3

        cooldown = obs[..., 75:100].long()


        # Location

        loc_x_float = obs[..., 100:125] * 15
        agent_loc_x = loc_x_float[..., 2 * 5 + 2].long()
        x_one_hot = F.one_hot(agent_loc_x, num_classes = 15) 

        loc_y_float = obs[..., 125:150] * 15
        agent_loc_y = loc_y_float[..., 2 * 5 + 2].long()
        y_one_hot = F.one_hot(agent_loc_y, num_classes = 15) 

        one_hot_obs = torch.cat([
            is_agent, 
            is_opponent, 
            agent_id, 
            health_2, 
            health_3,
            cooldown,
            x_one_hot,
            y_one_hot
        ], -1).float()
        return one_hot_obs.detach().numpy()

class CommNetMLP(nn.Module):
    '''
    Skip connections: we need skip connections from h0 t
    '''
    def __init__(self, comm_steps, hidden_size = 50, obs_size = 280, actions = 10):
        super().__init__()
        self.comm_steps = comm_steps
        self.encoder = nn.Linear(in_features = obs_size, 
                             out_features=hidden_size)
        self.f0 = nn.Linear(in_features = hidden_size, 
                             out_features=hidden_size)
        self.f = nn.ModuleList()
        for i in range(comm_steps):
            self.f.append(nn.Linear(in_features = hidden_size * 2, 
                             out_features=hidden_size))
        # Add an extra output to the decoder for the baseline
        self.decoder = nn.Linear(in_features = hidden_size, out_features = actions + 1)
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, observations, mask):
        ''' Infer the probability distributions over actions, given observed state
        Observations
        
        TODO: Deal with masks, add communication
        '''
        
        h0 = torch.tanh(self.encoder(observations))
        
        # First layer is different, since it has no skip connection
        h = torch.tanh(self.f0(h0))
        
        for i in range(self.comm_steps):
            # TODO: Communicate
            
            # Add skip connection to the input encoding
            layer_input = torch.cat([h0, h], dim = -1)
            # Apply next communication step
            h = torch.tanh(self.f[i](layer_input))
        out = self.decoder(h)
        p_act = self.softmax(out[..., :-1])
        baseline = out[..., -1]
        return p_act, baseline

class CombatPolicyGradient:
    ''' Performs policy gradient on the Combat environment on some policy
    This class is also responsible for logging performance
    and for visualizing it. 
    Can perform multiple repeats on instantiations of the same policy 
    and aggregate statistics
    '''
    def __init__(self, policy_class, policy_parameters, reps, optimizer, optim_params, training_params, env_params):
        self.em = CombatEnvironmentManager(**env_params)
        # Create policies
        self.policies = [policy_class(**policy_parameters) 
                         for i in range(reps)]
        # Create optimizer for each policy
        self.optimizers = [optimizer(policy.parameters(), **optim_params) 
                           for policy in self.policies]
        
        self.training_params = training_params
        self.win_rate = [[] for i in range(reps)]
        self.agent_survival = [[] for i in range(reps)]
        self.policy_loss = [[] for i in range(reps)]
        self.epoch_duration = [[] for i in range(reps)]
        self.total_reward = [[] for i in range(reps)]
        self.baseline_loss = [[] for i in range(reps)]
        self.actions_dist = [[] for i in range(reps)]
        self.entropy = [[] for i in range(reps)]
    def train(self, epochs = 10, batch_size = 288, lr = 0.005):
        ''' Train the policies for a number of epochs
        Results are appended to the logs
        '''
        for e in tqdm(range(epochs)):
            for policy_idx, _ in enumerate(self.policies):
                self.train_epoch(policy_idx, **self.training_params)
    def train_epoch(self, policy_idx, steps = 100, 
                    baseline_weight = 0.03, batch_size = 288,
                   time_normalize = False, entropy_reg = 0.,
                   gradient_clip = 40
                   ):
        ''' Train a single epoch and add results to the logs
        '''
        policy = self.policies[policy_idx]
        optimizer = self.optimizers[policy_idx]
        # Generate a training batch
        #print('Making batch...')
        batch = self.em.generate_batch(policy, batch_size)
        
        #print('Training...')
        # Get frozen "target" baseline
        probs, fixed_baseline = policy(batch.o, batch.mask)
        #baseline = baseline.detach()
        fixed_baseline = fixed_baseline.detach()
        #print("Baseline mean: {}".format(torch.mean(fixed_baseline.abs())))
        
        if not time_normalize:
            norm_mask = batch.mask / torch.sum(batch.mask)
            #print("Unmasked samples: {} / {}".format(torch.sum(batch.mask), batch.mask.numel()))

        else:
            # Construct norm mask where each agent trajectory is given equal weight
            # Divide each agent's trajectory by the number of turns it was alive
            norm_mask = batch.mask / torch.sum(batch.mask, axis=-2, keepdim=True)
            # Fix divide-by-zero for agents that died immediately
            norm_mask[torch.isnan(norm_mask)] = 0
            # Normalize for batch size and number of agents
            norm_mask /= torch.sum(norm_mask)
        norm_mask = norm_mask.detach()
            
        #print("Total norm: {}".format(torch.sum(norm_mask * batch.mask)))
        # Log batch stats
    
        self.agent_survival[policy_idx].append(np.mean(batch.mask.detach().numpy()))
        self.total_reward[policy_idx].append(np.mean(batch.R[:,0].detach().numpy()))
        self.actions_dist[policy_idx].append(
            torch.bincount(batch.a.flatten(), weights = batch.mask.flatten(), minlength = 10).detach().numpy())
        batch_entropy = torch.sum(Categorical(probs).entropy() * norm_mask).detach().numpy()
        self.entropy[policy_idx].append(batch_entropy)
        
        for i in range(steps):
            #print("Step {}/{}".format(i, steps))
            optimizer.zero_grad()
            
            # Evaluate policy
            probs, baseline = policy(batch.o, batch.mask)
            m = Categorical(probs)
            
            # Policy gradient loss
            policy_loss = -m.log_prob(batch.a) * (batch.R)
            policy_loss = torch.sum(policy_loss * norm_mask)
            self.policy_loss[policy_idx].append(float(policy_loss.detach().numpy()))
            
            # Baseline loss
            baseline_loss = torch.sum(batch.mask * (batch.R - baseline)**2)
            baseline_loss = torch.sum(baseline_loss * norm_mask)
            self.baseline_loss[policy_idx].append(float(baseline_loss.detach().numpy()))
            
            # Entropy loss
            entropy_loss = m.log_prob(batch.a)
            entropy_loss = torch.sum(entropy_loss * norm_mask)
            
            # Combined loss
            loss = policy_loss 
            #loss += baseline_weight * baseline_loss 
            loss += entropy_reg * entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
            optimizer.step()
    def plot_training_stats(self):
        '''
        Plot the logged training stats
        '''
        for s in self.agent_survival:
            plt.plot(s)
        plt.title('Agent survival')
        plt.show()
        
        for r0 in self.total_reward:
            plt.plot(r0)
        plt.title('Accumulated reward')
        plt.show()

        for loss in self.policy_loss:
            plt.plot(loss)
        plt.title('Policy loss')
        plt.show()

        for loss in self.baseline_loss:
            plt.plot(loss)
        plt.title('Baseline loss')
        plt.show()

        for e in self.entropy:
            plt.plot(e)
        plt.title('Action entropy')
        plt.show()
        
        h = np.array([d[-1] / np.sum(d[-1]) for d in self.actions_dist])
        x = np.arange(10)
        w = 0.8 / len(self.policies)
        for i, h in enumerate(h):
            plt.bar(x - i * w , h, w)
        plt.xticks(x)
        plt.title('Action distribution')
        plt.show()

# In current form the code is best run interactively, e.g. in a 
# notebook.
# As this implementation did not work well enough to run systematic
# experiments, it is not designed to be executed as a separate script.

if __name__ == "__main__":
    policy_params = dict(
        comm_steps = 1,
        hidden_size = 50
    )

    optim_params = dict(
        lr = 0.001
    )

    training_params = dict(
        steps = 25,
        baseline_weight = 0.003,
        batch_size = 50,
        entropy_reg = 10.,
        time_normalize = False
    )

    env_params = dict(
        random = True
    )
        
    pg = CombatPolicyGradient(CommNetMLP, policy_params, 2, torch.optim.RMSprop, optim_params, training_params, env_params)
    pg.train(10)
    pg.plot_training_stats()