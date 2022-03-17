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

timing = {
            'reset_env': 0,
            'train_step': 0,
            'per': 0,
            'sample': 0,
            'compute_targets': 0,
            'compute_predictions': 0,
            'backprop': 0,
            'update_target': 0,
            'model_forward': 0,
            'on_policy': 0,
            'step': 0,
            'target_model_forward': 0,
            'off_policy': 0,
            'get_delta': 0,
            'replay_buffer_append': 0
}
time_stuff = False

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

    def train(self, epochs: int, trajectory_len: int, env_wrapper: utils.EnvWrapper, lr=1e-4,
              discount_gamma=0.99, scheduler_gamma=0.98, beta=1e-3, print_interval=1000, log_interval=1000,
              save_interval=10000, scheduler_interval=1000, no_per=False, epsilon=0,
              epsilon_decay=0.997, eval_interval=0, stop_trick_at=0, batch_size=32, epsilon_min=0.01,
              epsilon_bounded=False, device=torch.device('cpu'), update_target_interval=10000,
              backprop_interval=32, replay_init_len=50000, final_exp_time = int(1e6), clip_loss=False,
              no_cuda=False, **kwargs):
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
        self.env = env_wrapper
        # self.target_model.eval()
        eps_step = (epsilon - epsilon_min) / final_exp_time

        if torch.cuda.is_available() and not no_cuda:
            sys.stdout.write('Using CUDA {}\n'.format(device))
        else:
            sys.stdout.write('Using CPU\n')

        # Init replay buffer with 50K random examples
        sys.stdout.write('Initializing replay buffer\n')
        sys.stdout.flush()
        state = self.env.reset()
        frames = 0
        with torch.no_grad():
            done = False
            while frames < replay_init_len:
                frames += 1
                if done:
                    state = self.env.reset()
                    done = False
                else:
                    done, state = self.train_step(state, epsilon, epsilon_bounded, discount_gamma)
        sys.stdout.write('Replay buffer initialized\n')
        sys.stdout.flush()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
        steps_count = 0

        global time_stuff
        time_stuff = True

        for episode in range(epochs):
            ep_start_time = time.time()
            self.episode_rewards = []

            start_time = time.time()
            state = self.env.reset()
            timing['reset_env'] += time.time() - start_time

            for step in range(self.env.max_steps):
                steps_count += 1

                start_time = time.time()
                done, state = self.train_step(state, epsilon, epsilon_bounded, discount_gamma)
                timing['train_step'] += time.time() - start_time

                if epsilon > epsilon_min:
                    epsilon -= eps_step
                if (not (steps_count % backprop_interval)) and (steps_count != 0):
                    if not no_per:

                        start_time = time.time()
                        self.replay_buffer.per()
                        timing['per'] += time.time() - start_time

                    start_time = time.time()
                    states, action_idxs, rewards, new_states, dones = self.replay_buffer.sample(batch_size)
                    timing['sample'] += time.time() - start_time

                    with torch.no_grad():
                        new_qs = self.target_model(new_states)
                    off_policy = self.env.off_policy(new_qs).to(device)

                    start_time = time.time()
                    targets = (rewards.to(device) + discount_gamma * off_policy * (1 - dones.int().to(device))).view(-1, 1)
                    timing['compute_targets'] += time.time() - start_time

                    start_time = time.time()
                    predictions = self.model(states).to(device)
                    predictions = torch.gather(predictions, -1, action_idxs.squeeze(-1).to(device))
                    timing['compute_predictions'] += time.time() - start_time

                    start_time = time.time()
                    self.model.train()
                    optimizer.zero_grad()
                    loss = F.mse_loss(predictions, targets.float())
                    if clip_loss:  # This doesn't work well
                        loss = torch.clip(loss, -1, 1)
                    loss.backward()
                    optimizer.step()
                    self.model.eval()
                    timing['backprop'] += time.time() - start_time

                if not (steps_count % update_target_interval) and (steps_count != 0):

                    start_time = time.time()
                    self.target_model.load_state_dict(self.model.state_dict())
                    timing['update_target'] += time.time() - start_time

                if done:
                    break

            # Either done or max steps per episode reached
            self.all_times.append(time.time() - ep_start_time)
            self.all_rewards.append(np.sum(self.episode_rewards))
            self.all_lengths.append(step)
            if (episode % print_interval == 0) and (episode != 0):
                utils.print_stats(self, episode, print_interval, step)
                print(timing)
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
            sys.stdout.flush()

        sys.stdout.write('-' * 10 + ' Finished training ' + '-' * 10 + '\n')
        sys.stdout.flush()
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

            start_time = time.time()
            q_vals = self.model.forward(state)
            if time_stuff: timing['model_forward'] += time.time() - start_time

            start_time = time.time()
            action, action_idx = self.env.on_policy(q_vals, epsilon, eps_bounded=epsilon_bounded)
            if time_stuff: timing['on_policy'] += time.time() - start_time

            start_time = time.time()
            new_state, reward, done, info = self.env.step(action)
            if time_stuff: timing['step'] += time.time() - start_time

            start_time = time.time()
            new_q_vals = self.target_model.forward(new_state).detach()
            if time_stuff: timing['target_model_forward'] += time.time() - start_time

            start_time = time.time()
            new_q = self.env.off_policy(new_q_vals)
            if time_stuff: timing['off_policy'] += time.time() - start_time

            start_time = time.time()
            target = (reward + discount_gamma * new_q).view(1, -1, 1)
            delta = self.get_delta(q_vals, action_idx, target)
            if time_stuff: timing['get_delta'] += time.time() - start_time

            start_time = time.time()
            self.replay_buffer.append((state, action_idx, reward, new_state, done, delta))
            if time_stuff: timing['replay_buffer_append'] += time.time() - start_time

            self.episode_rewards.append(reward)
        return done, new_state

