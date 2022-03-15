import sys
import gym
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import utils
import time


class A2CAgent:
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
        self.traj_lengths = []

    def train(self, epochs: int, trajectory_len: int, env_gen: utils.AsyncEnvGen, lr=1e-4,
              discount_gamma=0.99, scheduler_gamma=0.98, beta=1e-3, print_interval=1000, log_interval=1000,
              save_interval=10000, scheduler_interval=1000, clip_gradient=False, stop_trick_at=0, no_cuda=False,
              **kwargs):
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
        sys.stdout.flush()
        self.model.to(device)
        self.model.device = device
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
        tricks_used = 0
        steps_count = 0

        for episode in range(epochs):
            ep_start_time = time.time()
            episode_rewards = []
            state, self.env = env_gen.get_reset_env()
            traj_log_probs, traj_values, traj_rewards = [], [], []
            traj_entropy_term = torch.zeros(1).to(device)
            if stop_trick_at and episode == stop_trick_at:
                self.env.cone_trick = False
                self.env.move_trick = False
                sys.stdout.write('Stopped using trick\n')

            for step in range(self.env.max_steps):
                steps_count += 1
                value, policy_dist = self.model.forward(state)
                value = value.detach().item()
                action, log_prob, entropy = self.env.process_action(policy_dist.detach().squeeze(0),
                                                                    policy_dist.squeeze(0))
                new_state, reward, done, info = self.env.step(action)
                if step == self.env.max_steps - 1:
                    done = True
                if info['used_trick']:
                    tricks_used += 1
                traj_rewards.append(reward)
                episode_rewards.append(reward)
                traj_values.append(value)
                traj_log_probs.append(log_prob)
                traj_entropy_term += entropy
                state = new_state
                if log_interval: # ie if to log at all
                    self.log_buffer.append(info.__repr__() + '\n')

                if done or ((step % trajectory_len == 0) and step != 0):
                    self.traj_lengths.append((step % trajectory_len) + 1)
                    q_val, _ = self.model.forward(new_state) # this is actually last_val and not yet a q value
                    q_val = q_val.detach().item()
                    q_vals = torch.zeros(len(traj_values)).to(device)
                    for t in reversed(range(len(traj_rewards))):
                        q_val = traj_rewards[t] + discount_gamma * q_val
                        q_vals[t] = q_val
                    traj_values = torch.FloatTensor(traj_values).to(device)
                    traj_log_probs = torch.stack(traj_log_probs, dim=traj_log_probs[0].dim()) # for more than one action dim will be 1
                    advantage = q_vals - traj_values
                    actor_loss = (-traj_log_probs * advantage).mean()
                    critic_loss = 0.5 * advantage.pow(2).mean()
                    ac_loss = (actor_loss + critic_loss + beta * traj_entropy_term)
                    optimizer.zero_grad()
                    ac_loss.backward()
                    if clip_gradient:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    optimizer.step()
                    traj_log_probs, traj_values, traj_rewards = [], [], []
                    traj_entropy_term = torch.zeros(1).to(device)

                    if done:
                        self.all_times.append(time.time() - ep_start_time)
                        self.all_rewards.append(np.sum(episode_rewards))
                        self.all_lengths.append(step)
                        if np.mean(self.all_rewards[-100:]) >= 200:
                            print('='*10, 'episode {}, Last 100 episodes averaged 200 points '.format(episode), '='*10)
                            return
                        if (episode % print_interval == 0) and episode != 0:
                            utils.print_stats(self, episode, print_interval, tricks_used, steps_count)
                            tricks_used = 0
                            steps_count = 0
                        if (episode % scheduler_interval == 0) and (episode != 0):
                            scheduler.step()
                            sys.stdout.write('stepped scheduler, new lr: {:.5f}\n'.format(scheduler.get_last_lr()[0]))
                            sys.stdout.flush()
                        if (episode % save_interval == 0) and (episode != 0):
                            utils.save_agent(self)
                        if log_interval and (episode % log_interval == 0) and (episode != 0):
                            utils.log(self)
                        break

        sys.stdout.write('-' * 10 + ' Finished training ' + '-' * 10 + '\n')
        sys.stdout.flush()
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
