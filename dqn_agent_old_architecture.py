import sys
import gym
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import utils
import time
import torch.nn.functional as F


class DQNAgent:
    """
    Holds a model, an environment and a training session
    Can train or perform actions
    """

    def __init__(self, model, save_path, log_path, **kwargs):
        self.model = model
        self.env = gym.Env
        self.kwargs = kwargs
        self.save_path = save_path
        self.log_path = log_path
        self.all_lengths = []
        self.average_lengths = []
        self.all_rewards = []
        self.all_times = []
        self.log_buffer = []

    def train(self, epochs: int, trajectory_len: int, env_gen: utils.AsyncEnvGen, lr=1e-4,
              discount_gamma=0.99, scheduler_gamma=0.98, beta=1e-3, print_interval=1000, log_interval=1000,
              save_interval=10000, scheduler_interval=1000, no_per=False, no_cuda=False, **kwargs):
        """
        Trains the model
        :param epochs: int, number of epochs to run
        :param trajectory_len: int, maximal length of a single trajectory
        :param env_gen: AsyncEnvGen, generates environments asynchronicaly
        :param lr: float, learning rate
        :param discount_gamma: float, discount factor
        :param scheduler_gamma: float, LR decay factor
        :param beta: float, information gain factor
        :return:
        """
        if torch.cuda.is_available() and not no_cuda:
            device = torch.device('cuda')
            sys.stdout.write('Using CUDA\n')
        else:
            device = torch.device('cpu')
            sys.stdout.write('Using CPU\n')
        self.model.to(device)
        self.model.device = device
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

        for episode in range(epochs):
            ep_start_time = time.time()
            episode_rewards = []
            state, self.env = env_gen.get_reset_env()
            traj_log_probs, traj_values, traj_rewards = [], [], []

            experience = []

            for step in range(self.env.max_steps):

                with torch.no_grad():
                    q_vals = self.model.forward(state)
                    action, action_idx = self.env.on_policy(q_vals)
                    new_state, reward, done, info = self.env.step(action)
                    if not done:
                        new_q_vals = self.model.forward(new_state).detach()
                        new_q = self.env.off_policy(new_q_vals)
                    else:
                        new_q = self.get_zero_q().to(self.model.device)
                target = self.get_target(new_q, reward)
                delta = self.get_delta(q_vals, action_idx, target)
                experience.append((state, action_idx, reward, new_q, delta))
                state = new_state
                if step == self.env.max_steps - 1:
                    done = True
                traj_rewards.append(reward)
                episode_rewards.append(reward)
                if log_interval: # ie if to log at all
                    self.log_buffer.append(info.__repr__() + '\n')

                if done or ((step % trajectory_len == 0) and step != 0):
                    if not no_per: # IE use PER
                        sorted_exp = sorted(experience, key=lambda tup: tup[-1])
                        high_delta = sorted_exp[-len(experience)//2:]
                        low_delta = sorted_exp[:len(experience)//2]
                        per = high_delta + random.choices(low_delta, k=len(experience)//6)
                    else:
                        per = experience
                    random.shuffle(per)
                    for e in per:
                        # TODO: SWITCH TO BATCHS HERE (NOT ONE BY ONE)
                        state = e[0]
                        action_idx = e[1]
                        reward = e[2]
                        new_q = e[3]
                        prediction = self.predict(state, action_idx)
                        target = self.get_target(new_q, reward)
                        optimizer.zero_grad()
                        loss = F.mse_loss(prediction, target)
                        loss.backward()
                        optimizer.step()
                    if done:
                        self.all_times.append(time.time() - ep_start_time)
                        self.all_rewards.append(np.sum(episode_rewards))
                        self.all_lengths.append(step)
                        if np.mean(self.all_rewards[-100:]) >= 200:
                            print('='*10, 'episode {}, Last 100 episodes averaged 200 points '.format(episode), '='*10)
                            return
                        if (episode % print_interval == 0) and episode != 0:
                            utils.print_stats(self, episode, print_interval)
                        if (episode % scheduler_interval == 0) and (episode != 0):
                            scheduler.step()
                            sys.stdout.write('stepped scheduler, new lr: {:.5f}\n'.format(scheduler.get_last_lr()[0]))
                        if (episode % save_interval == 0) and (episode != 0):
                            utils.save_agent(self)
                        if log_interval and (episode % log_interval == 0) and (episode != 0):
                            utils.log(self)
                        break

        sys.stdout.write('-' * 10 + ' Finished training ' + '-' * 10 + '\n')
        utils.kill_process(env_gen)
        sys.stdout.write('Killed env gen process\n')
        utils.save_agent(self)
        if log_interval:
            utils.log(self)

    def act(self, state):
        self.model.eval()
        _, policy_dist = self.model.forward(state)
        dist = policy_dist.detach().squeeze(0)
        action = torch.multinomial(dist, 1).item()
        self.model.train()
        return action

    def get_delta(self, q_vals, action_idx, target):
        # TODO: replace with .view(1,-1,1)
        if self.env.action_type in [utils.ActionType.REGULAR, utils.ActionType.FIXED_LUNAR]:
            delta = abs(q_vals.detach().squeeze(0)[action_idx] - target) + 0.001
        elif self.env.action_type == utils.ActionType.DISCRETIZIED:
            delta = abs((torch.stack([q_vals.detach()[i][action_idx[i]]
                                  for i in range(action_idx.dim())], dim=1) - target)).sum() + 0.001
        else:
            raise NotImplementedError
        return delta

    def predict(self, state, action_idx):
        # TODO: replace with .view(1,-1,1)
        q_vals = self.model.forward(state)
        if self.env.action_type in [utils.ActionType.REGULAR, utils.ActionType.FIXED_LUNAR]:
            # prediction = q_vals.squeeze(0)[action_idx].unsqueeze(0).unsqueeze(0)
            prediction = q_vals.squeeze(0)[action_idx].view(1,1,1)
        elif self.env.action_type == utils.ActionType.DISCRETIZIED:
            # prediction = torch.stack([q_vals[i][action_idx[i]] for i in range(self.env.num_actions)]).unsqueeze(0)
            prediction = torch.stack([q_vals[i][action_idx[i]] for i in range(self.env.num_actions)])\
                .view(1, self.env.num_actions, 1)
        else:
            raise NotImplementedError
        return prediction

    def get_target(self, new_q, reward):
        # TODO: replace with .view(1,-1,1)
        if self.env.action_type in [utils.ActionType.REGULAR, utils.ActionType.FIXED_LUNAR]:
            # target = torch.FloatTensor([new_q + reward]).unsqueeze(0)
            target = (new_q + reward).view(1, 1, 1)
        elif self.env.action_type == utils.ActionType.DISCRETIZIED:
            target = (new_q + reward).view(1, self.env.num_actions, 1)
        else:
            raise NotImplementedError
        return target.to(self.model.device)

    def get_zero_q(self):
        # TODO: replace with .view(1,-1,1)
        if self.env.action_type in [utils.ActionType.REGULAR, utils.ActionType.FIXED_LUNAR]:
            zeros = torch.zeros(1, dtype=torch.float32)
        elif self.env.action_type == utils.ActionType.DISCRETIZIED:
            zeros = torch.zeros(self.env.num_actions, dtype=torch.float32)
        else:
            raise NotImplementedError
        return zeros




