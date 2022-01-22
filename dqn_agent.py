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
from torch.utils.data import DataLoader
from torch.nn.modules.rnn import LSTM


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
        self.is_lstm = any([isinstance(module, LSTM) for module in model.modules()])

    def train(self, epochs: int, trajectory_len: int, env_gen: utils.AsyncEnvGen, lr=1e-4,
              discount_gamma=0.99, scheduler_gamma=0.98, beta=1e-3, print_interval=1000, log_interval=1000,
              save_interval=10000, scheduler_interval=1000, no_per=False, no_cuda=False, epsilon=0,
              epsilon_decay=0.997, eval_interval=0, stop_trick_at=0, batch_size=64, epsilon_min=0.01,
              epsilon_bounded=False, **kwargs):
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
        steps_count = 0

        for episode in range(epochs):
            ep_start_time = time.time()
            episode_rewards, traj_rewards = [], []
            state, self.env = env_gen.get_reset_env()
            if stop_trick_at and episode >= stop_trick_at:
                self.env.cone_trick = False
                self.env.move_trick = False
                if episode == stop_trick_at:
                    sys.stdout.write('Stopped using trick\n')

            experience = []

            for step in range(self.env.max_steps):
                steps_count += 1
                with torch.no_grad():
                    q_vals = self.model.forward(state)
                    action, action_idx = self.env.on_policy(q_vals, epsilon, eps_bounded=epsilon_bounded)
                    new_state, reward, done, info = self.env.step(action)
                    new_q_vals = self.model.forward(new_state).detach()
                    new_q = self.env.off_policy(new_q_vals)
                target = (reward + discount_gamma * new_q).view(1, -1, 1)
                delta = self.get_delta(q_vals, action_idx, target)
                experience.append((state, action_idx, reward, new_state, done, delta))
                state = (new_state[0].copy(), new_state[1].copy())
                if step == self.env.max_steps - 1:
                    done = True
                traj_rewards.append(reward)
                episode_rewards.append(reward)
                if log_interval: # ie if to log at all
                    self.log_buffer.append(info.__repr__() + '\n')

                if done or ((step % trajectory_len == 0) and step != 0):
                    dataset = utils.PERDataLoader(experience, use_per=(not no_per))
                    if self.is_lstm:
                        # No shuffeling for LSTM!
                        dataloader = DataLoader(dataset, batch_size=min(len(dataset), batch_size), shuffle=False)
                        self.model.reset_hidden(batch_size=batch_size)
                    else:
                        dataloader = DataLoader(dataset, batch_size=min(len(dataset), batch_size), shuffle=True)
                    for states, action_idxs, rewards, new_states, dones in dataloader:
                        with torch.no_grad():
                            new_qs = self.model(new_states)
                            off_policy = self.env.off_policy(new_qs)
                            off_policy *= (1 - dones.int().to(device)) # Q=0 where action leads to end of episode
                        targets = (rewards.to(device) + discount_gamma * off_policy).view(-1, 1)
                        predictions = self.model(states)
                        targets_full = predictions.detach().clone()
                        targets_full.scatter_(-1, action_idxs.squeeze(-1).to(device), targets.float())
                        optimizer.zero_grad()
                        loss = F.mse_loss(predictions, targets_full.float())
                        if self.is_lstm:
                            loss.backward(retain_graph=True)
                            self.model.reset_hidden(batch_size=batch_size)
                        else:
                            loss.backward()
                        optimizer.step()
                    if done:
                        if self.is_lstm:
                            self.model.reset_hidden()
                        self.all_times.append(time.time() - ep_start_time)
                        self.all_rewards.append(np.sum(episode_rewards))
                        self.all_lengths.append(step)
                        if np.mean(self.all_rewards[-100:]) >= 200:
                            sys.stdout.write('{0} episode {1}, Last 100 train episodes averaged 200 points {0}\n'
                                             .format('*' * 10, episode))
                            utils.save_agent(self)
                            return
                        if epsilon > epsilon_min:
                            epsilon *= epsilon_decay
                        if np.mean(self.all_rewards[-100:]) >= 200:
                            print('='*10, 'episode {}, Last 100 episodes averaged 200 points '.format(episode), '='*10)
                            return
                        if (episode % print_interval == 0) and episode != 0:
                            utils.print_stats(self, episode, print_interval, steps_count)
                            steps_count = 0
                        if (episode % scheduler_interval == 0) and (episode != 0):
                            scheduler.step()
                            sys.stdout.write('stepped scheduler, new lr: {:.5f}\n'.format(scheduler.get_last_lr()[0]))
                        if (episode % save_interval == 0) and (episode != 0):
                            utils.save_agent(self)
                        if log_interval and (episode % log_interval == 0) and (episode != 0):
                            utils.log(self)
                        if eval_interval:
                            if ((episode % eval_interval)  == 0 ) and (episode != 0):
                                _, all_episode_rewards, completed_sokoban_levels = utils.evaluate(self, 100, render=False)
                                utils.print_eval(all_episode_rewards, completed_sokoban_levels)
                                if np.mean(all_episode_rewards) >= 200:
                                    sys.stdout.write('{0} episode {1}, Last 100 eval episodes averaged 200 points {0}\n'
                                                     .format('*' * 10, episode))
                                    utils.save_agent(self)
                                    return
                        break

        sys.stdout.write('-' * 10 + ' Finished training ' + '-' * 10 + '\n')
        utils.kill_process(env_gen)
        sys.stdout.write('Killed env gen process\n')
        utils.save_agent(self)
        if log_interval:
            utils.log(self)

    def act(self, state):
        q_vals = self.model.forward(state)
        action, _ = self.env.on_policy(q_vals, eps=0, is_eval=True)
        return action

    def get_delta(self, q_vals, action_idx, target):
        delta = abs(q_vals.detach().squeeze(0)[action_idx] - target) + 0.001
        return delta

    def predict(self, state, action_idx):
        q_vals = self.model.forward(state)
        if self.env.action_type in [utils.ActionType.REGULAR, utils.ActionType.FIXED_LUNAR]:
            prediction = q_vals.gather(-1, action_idx.squeeze(-1)).view(1, -1, 1)
        elif self.env.action_type == utils.ActionType.DISCRETIZIED:
            prediction = q_vals.gather(-1,action_idx.squeeze(0))
        else:
            raise NotImplementedError
        return prediction

    def get_zero_q(self):
        if self.env.action_type in [utils.ActionType.REGULAR, utils.ActionType.FIXED_LUNAR]:
            zeros = torch.zeros(1, dtype=torch.float32).squeeze(0)
        elif self.env.action_type == utils.ActionType.DISCRETIZIED:
            zeros = torch.zeros(self.env.num_actions, dtype=torch.float32).squeeze(0)
        else:
            raise NotImplementedError
        return zeros




