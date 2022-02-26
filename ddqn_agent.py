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

# REPLAY_INIT_LEN = 100 # 50000
# REPLAY_MAX_LEN = 500
# EPS_STEP = 0.9 / 1e6
# EPS_STEP = 0.0005
# UPDATE_TARGET_INTERVAL = 100
# BACKPROP_INTERVAL =  32

class DDQNAgent:
    """
    Double DQN Agent as per https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf,
    https://arxiv.org/pdf/1509.06461.pdf
    Holds a model, an environment and a training session
    Can train or perform actions
    """

    def __init__(self, model, save_path, log_path, replay_max_len=int(1e6), **kwargs):
        self.model = model
        self.target_model = utils.clone_model(self.model)
        self.env = gym.Env
        self.kwargs = kwargs
        self.save_path = save_path
        self.log_path = log_path
        self.all_lengths = []
        self.average_lengths = []
        self.all_rewards = []
        self.all_times = []
        self.log_buffer = []
        self.episode_rewards = []
        self.is_lstm = any([isinstance(module, LSTM) for module in model.modules()])
        self.replay_buffer = utils.PERDataSet(max_len=replay_max_len)

    def train(self, epochs: int, trajectory_len: int, env_gen: utils.AsyncEnvGen, lr=1e-4,
              discount_gamma=0.99, scheduler_gamma=0.98, beta=1e-3, print_interval=1000, log_interval=1000,
              save_interval=10000, scheduler_interval=1000, no_per=False, epsilon=0,
              epsilon_decay=0.997, eval_interval=0, stop_trick_at=0, batch_size=32, epsilon_min=0.01,
              epsilon_bounded=False, device=torch.device('cpu'), update_target_interval=10000,
              backprop_interval=32, replay_init_len=50000, final_exp_time = int(1e6), clip_loss=False, **kwargs):
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
        self.model.to(device)
        self.target_model.to(device)
        self.model.device = device
        self.model.eval()
        self.target_model.eval()
        # self.target_model.eval()
        eps_step = (epsilon - epsilon_min) / final_exp_time

        # Init replay buffer with 50K random examples
        print('Initializing replay buffer')
        state, self.env = env_gen.get_reset_env()
        frames = 0
        with torch.no_grad():
            while frames < replay_init_len:
                frames += 1
                done, state = self.train_step(state, epsilon, epsilon_bounded, discount_gamma)
                if done:
                    state, self.env = env_gen.get_reset_env()
        print('Replay buffer initialized')

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
        steps_count = 0

        for episode in range(epochs):
            ep_start_time = time.time()
            self.episode_rewards = []
            state, self.env = env_gen.get_reset_env()

            for step in range(self.env.max_steps):
                steps_count += 1
                done, state = self.train_step(state, epsilon, epsilon_bounded, discount_gamma)
                if epsilon > epsilon_min:
                    epsilon -= eps_step
                # if not (steps_count % update_target_interval):
                #     self.target_model.load_state_dict(self.model.state_dict())
                if done or ((not steps_count % backprop_interval) and steps_count != 0):
                    if not no_per:
                        self.replay_buffer.per()
                    states, action_idxs, rewards, new_states, dones = self.replay_buffer.sample(batch_size)
                    with torch.no_grad():
                        new_qs = self.target_model(new_states)
                    off_policy = self.env.off_policy(new_qs).to(device)
                    targets = (rewards.to(device) + discount_gamma * off_policy * (1 - dones.int().to(device))).view(-1, 1)
                    predictions = self.model(states).to(device)
                    predictions = torch.gather(predictions, -1, action_idxs.squeeze(-1).to(device))
                    self.model.train()
                    optimizer.zero_grad()
                    loss = F.mse_loss(predictions, targets.float())
                    if clip_loss:  # This doesn't work well
                        loss = torch.clip(loss, -1, 1)
                    loss.backward()
                    optimizer.step()
                    self.model.eval()

                if not (steps_count % update_target_interval) and (steps_count != 0):
                    self.target_model.load_state_dict(self.model.state_dict())

                if done:
                    break

            if done:
                self.all_times.append(time.time() - ep_start_time)
                self.all_rewards.append(np.sum(self.episode_rewards))
                self.all_lengths.append(step)
                if np.mean(self.all_rewards[-100:]) >= 200:
                    sys.stdout.write('{0} episode {1}, Last 100 train episodes averaged 200 points {0}\n'
                                     .format('*' * 10, episode))
                    utils.save_agent(self)
                    return
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

    def train_step(self, state, epsilon, epsilon_bounded, discount_gamma):
        """
        Steps the agent one move
        Notice that delta is computed with the target model that is updated on the fly
        This means that there will be 'stale' delta values in the buffer, but this should balance out
        As the buffer has a limited capacity and 'very stale' values will be washed out with time
        These stale targets are to be used with PER only and not with the backprop itself
        To enhance this wash-out (over randomness) lower the maximal buffer's length
        """
        with torch.no_grad():
            q_vals = self.model.forward(state)
            action, action_idx = self.env.on_policy(q_vals, epsilon, eps_bounded=epsilon_bounded)
            new_state, reward, done, info = self.env.step(action)
            new_q_vals = self.target_model.forward(new_state).detach()
            new_q = self.env.off_policy(new_q_vals)
            target = (reward + discount_gamma * new_q).view(1, -1, 1)
            delta = self.get_delta(q_vals, action_idx, target)
            self.replay_buffer.append((state, action_idx, reward, new_state, done, delta))
            self.episode_rewards.append(reward)
        return done, new_state

