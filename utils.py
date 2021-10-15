import gym
from enum import Enum
import time
import multiprocessing as mp
import torch
import math
from torch.distributions import Categorical, Normal
import numpy as np
import sys
import pickle
import gym_sokoban # Don't remove this
import gym
import random
import os
import torch.nn.functional as F
try:
    from gym.wrappers import Monitor
except ModuleNotFoundError as e:
    sys.stdout.write('Cannot import Monitor module, rendering won\'t be possible: {}\nContinuing..\n'.format(e))


class ObsType(Enum):
    VIDEO_ONLY = 1
    VIDEO_NO_CLUE = 2
    VIDEO_MONO = 3
    VIDEO_STEREO = 4


class ActionType(Enum):
    PUSH_TWICE_WAIT = 1
    FREE = 2
    NO_WAIT = 3


class MoveType(Enum):
    UP = 1
    LEFT = 2
    RIGHT = 3
    BUTTON = 4
    NONE = 9


class PERDataLoader(torch.utils.data.DataLoader):
    def __init__(self, experience, use_per, low_ratio=6):
        super(PERDataLoader, self).__init__(experience)
        self.use_per = True
        self.low_ratio = low_ratio
        if use_per:
            sorted_exp = sorted(experience, key=lambda tup: tup[-1])
            high_delta = sorted_exp[-len(experience) // 2:]
            low_delta = sorted_exp[:len(experience) // 2]
            per = high_delta + random.choices(low_delta, k=len(experience) // low_ratio)
            self.exp = per
        else:
            self.exp = experience

    def __getitem__(self, idx):
        # returns state, action_idx, reward, new_state, done
        exp = self.exp[idx]
        return exp[0], exp[1], exp[2], exp[3], exp[4]

    def __len__(self):
        return len(self.exp)


def kill_process(p):
    if p.is_alive():
        p.q.cancel_join_thread()
        p.kill()
        p.join(1)


def init_weights(model):
    for name, layer in model._modules.items():
        if hasattr(layer, '__iter__'):
            init_weights(layer)
        elif isinstance(layer, torch.nn.Module):
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
            elif isinstance(layer, torch.nn.modules.conv.Conv1d) or isinstance(layer, torch.nn.modules.conv.Conv2d):
                layer.weight.data.fill_(0.01)
                layer.bias.data.fill_(0.01)


def save_agent(agent):
    with open(agent.save_path, 'wb') as f:
        pickle.dump(agent, f)
    sys.stdout.write('Saved agent to {}\n'.format(agent.save_path))


def log(agent):
    with open(agent.log_path, 'a') as f:
        _ = f.writelines(agent.log_buffer)
        agent.log_buffer = []
    sys.stdout.write('Logged info to {}\n'.format(agent.log_path))


def print_stats(agent, episode, print_interval, steps_count=0):
    sys.stdout.write(
        "eps: {}, stats for last {} eps:\tavg eps reward: {:.3f}\t\tavg eps step reward: {:.3f}\t\t"
        "avg eps length: {:.3f}\t avg time: {:.3f}\n"
            .format(episode, print_interval, np.mean(agent.all_rewards[-print_interval:]),
                    np.sum(agent.all_rewards[-print_interval:]) / steps_count,
                    np.mean(agent.all_lengths[-print_interval:]) + 1,
                    np.mean(agent.all_times[-print_interval:])))


def print_eval(all_episode_rewards, completed_levels):
    sys.stdout.write('{0} Evaluation {0}'.format('*' * 10))
    sys.stdout.write('\nEvaluation on last 100 episodes:\tmean: {:.3f}\tmin: {:.3f}\t\tmax: {:.3f}\t\t'
                     '%completed levels (sokoban only): {:.3f}\n'.format(np.mean(all_episode_rewards),
                     np.min(all_episode_rewards), np.max(all_episode_rewards), completed_levels / 100))


def evaluate(agent, num_episodes=1, render=True):
    agent.model.eval()
    all_rewards = []
    all_episode_rewards = []
    completed_sokoban_levels = 0
    for epispode in range(num_episodes):
        if render:
            sys.stdout.write('Saving render video to {}\n'.format(os.path.join(os.getcwd(), 'video')))
            agent.env.env = Monitor(agent.env.env, './video', force=True)
        episode_rewards = []
        obs = agent.env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = agent.act(obs)
                obs, reward, done, info = agent.env.step(action, is_eval=True)
            if 'all_boxes_on_target' in info.keys():
                if info['all_boxes_on_target']:
                    completed_sokoban_levels += 1
            all_rewards.append(reward)
            episode_rewards.append(reward)
        all_episode_rewards.append(np.sum(episode_rewards))
    if render:
        agent.env.env.close()
    agent.model.train()
    return all_rewards, all_episode_rewards, completed_sokoban_levels


class EnvWrapper:
    """
    Wrapps a Sokoban gym environment s.t. we can use the room_state property instead of regular state
    """

    def __init__(self, env_name, obs_type=ObsType.VIDEO_ONLY, action_type=ActionType.PUSH_TWICE_WAIT,
                 max_steps=5000, compression_rate=4, **kwargs):
        """
        Wraps a gym environment s.t. you can control it's input and output
        :param env_name: str, The environments name
        :param obs_type: ObsType, type of output for environment's observations
        :param compression_rate: video compression rate
        :param args: Any args you want to pass to make()
        :param kwargs: Any kwargs you want to pass to make()
        """
        self.obs_type = obs_type
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.env.max_steps = max_steps
        self.max_steps = max_steps
        self.action_type = action_type
        self.compression_rate = compression_rate
        if obs_type == ObsType.VIDEO_ONLY:
            screen_size = self.env.observation_space[:, :, 0][::4, ::4].shape
            self.obs_size = screen_size[0] * screen_size[1]
        elif obs_type == ObsType.ROOM_STATE_VECTOR:
            self.obs_size = self.env.room_state.shape[0] ** 2
        elif obs_type == ObsType.VIDEO_NO_CLUE:
            raise NotImplementedError
        elif obs_type == ObsType.VIDEO_MONO:
            raise NotImplementedError
        elif obs_type == ObsType.VIDEO_STEREO:
            raise NotImplementedError
        if action_type == ActionType.REGULAR:
            self.num_actions = self.env.action_space.n
        elif action_type == ActionType.PUSH_ONLY:
            self.num_actions = 4
        elif action_type == ActionType.PUSH_PULL:
            self.num_actions = 8
        elif action_type == ActionType.GAUSSIAN:
            self.num_actions = self.env.action_space.shape[0]
        elif action_type == ActionType.DISCRETIZIED:
            self.num_actions = self.env.action_space.shape[0]
            self.num_discrete = kwargs['num_discrete']
            # specially taylored to this game
            if self.env_name == 'LunarLanderContinuous-v2':
                assert self.num_discrete % 2
                low_main = 0
                high_main = 1
                low_sides = -1
                middle_sides = 0.5
                high_sides = 1
                self.discrete_array = torch.cat([torch.tensor([-1]),
                                     torch.flip(torch.linspace(high_main, low_main, self.num_discrete - 1), dims=[0])])
                a = torch.linspace(low_sides, -middle_sides, self.num_discrete // 2)
                b = torch.cat([torch.tensor([0]), torch.flip(
                                     torch.linspace(high_sides, middle_sides, self.num_discrete // 2), dims=[0])])
                self.split_discrete_array = torch.cat((a, b))
            else:
                low = self.env.action_space.low[0].item()
                high = self.env.action_space.high[0].item()
                self.discrete_array = torch.arange(low, high, (high - low) / self.num_discrete)
                a = torch.arange(low, low / 2, (-low / 2) / (self.num_discrete // 2))
                b = torch.arange(high / 2, high, (high / 2) / (self.num_discrete // 2))
                self.split_discrete_array = torch.cat((a, b)) # in Lunar lander -0.5 to 0.5 is NOP for L\R engines
        elif action_type == ActionType.FIXED_LUNAR:
            self.num_actions = len(FIXED_ACTIONS)

    def reset(self):
        obs = self.env.reset()
        return self.process_obs(obs)

    def step(self, action, is_eval=False):
        if self.action_type in [ActionType.REGULAR, ActionType.FIXED_LUNAR]:
            pass # No change if action type is regular
        elif self.action_type == ActionType.PUSH_ONLY:
            # maps from 0-3 to 1-4 since 0 is NOP
            action += 1
        elif self.action_type == ActionType.PUSH_PULL:
            # maps from 0-7 to [1,2,3,4,9,10,11,12]
            action += 1
            if action >= 5:
                action += 4
        elif self.action_type == ActionType.GAUSSIAN:
            action = action.cpu().numpy()
        elif self.action_type == ActionType.DISCRETIZIED:
            action = action.flatten().numpy()
        obs, reward, done, info = self.env.step(action)
        obs = self.process_obs(obs)
        info['used_trick'] = False
        if self.cone_trick and not is_eval:
            x_pos = obs[0]
            y_pos = obs[1]
            alpha = math.atan2(y_pos, abs(x_pos))
            if (alpha < math.pi / 4) and (y_pos > 1/3):
                reward -= self.trick_fine
                done = True
                info['used_trick'] = True
        if self.move_trick and not is_eval:
            # Don't penalize for regular steps when using move trick
            if reward == -0.1:
                reward = 0
            room = self.env.room_state
            player_pos = self.env.player_position
            is_valid = is_valid_command(room, player_pos, MoveType(action))
            if not is_valid:
                reward -= self.trick_fine
                done = True
                info['used_trick'] = True
        return obs, reward, done, info

    def process_obs(self, obs):
        if (self.obs_type == ObsType.REGULAR) or (self.obs_type == ObsType.BOX2D):
            return obs
        elif self.obs_type == ObsType.ROOM_STATE_VECTOR:
            return self.env.room_state.flatten()
        elif self.obs_type == ObsType.ROOM_STATE_MATRIX:
            return self.env.room_state

    def process_action(self, dist, policy_dist, is_eval=False):
        if self.action_type == ActionType.GAUSSIAN:
            if is_eval:
                raise NotImplementedError
            detached_mu = dist[0]
            detached_sigma = dist[1]
            attached_mu = policy_dist[0]
            attached_sigma = policy_dist[1]
            action = reparametrize(attached_mu, attached_sigma).squeeze(0).squeeze(0)
            action_dist = Normal(detached_mu, detached_sigma)
            log_prob = torch.log(torch.sigmoid((torch.abs(action - detached_mu)) / detached_sigma))
            entropy = action_dist.entropy().detach().sum()
            action = action.detach()
        elif self.action_type == ActionType.DISCRETIZIED:
            if is_eval:
                action_idx = torch.argmax(dist, 1).view(-1,1)
            else:
                action_idx = torch.multinomial(dist, 1)
            if self.env_name == 'LunarLanderContinuous-v2':
                action = torch.stack((self.discrete_array[action_idx[0]], self.split_discrete_array[action_idx[1]]))
            else:
                action = self.discrete_array[action_idx]
            log_prob = torch.log(torch.gather(policy_dist, 1, action_idx).squeeze(1))
            entropy = Categorical(probs=dist).entropy().sum()
        elif self.action_type == ActionType.FIXED_LUNAR:
            if is_eval:
                action_idx = torch.argmax(dist, -1).item()
            else:
                action_idx = torch.multinomial(dist, 1).item()
            action = FIXED_ACTIONS[action_idx]
            log_prob = torch.log(policy_dist[action_idx])
            entropy = Categorical(probs=dist).entropy()
        else:
            if is_eval:
                action = torch.argmax(dist, -1).item()
            else:
                action = torch.multinomial(dist, 1).item()
            log_prob = torch.log(policy_dist[action])
            entropy = Categorical(probs=dist).entropy()
        return action, log_prob, entropy

    def on_policy(self, q_vals, eps=0, is_eval=False):
        """
        Returns on policy (epsilon soft or greedy) action for a DQN net
        Returns epsilon soft by default. If eps is specified will return epsilon greedy
        with the given eps value.
        :param q_vals: Tensor - q values per action
        :return: Int - action to take
        """
        if is_eval:
            # in evaluation take the best action you can do..
            action_idx = torch.argmax(q_vals, axis=-1).view(q_vals.shape[0], -1).to(q_vals.device)
        elif eps: # epsilon greedy option
            if np.random.rand() <= eps:
                action_idx = torch.randint(0, q_vals.shape[-1], (q_vals.shape[0], 1)).to(q_vals.device)
            else:
                action_idx = torch.argmax(q_vals, axis=-1).view(q_vals.shape[0], -1).to(q_vals.device)
        else: # epsilon soft option
            activated = F.softmax(q_vals, dim=1)
            action_idx = torch.multinomial(activated, 1)

        if self.action_type == ActionType.REGULAR:
            action = action_idx.item()
        elif self.action_type == ActionType.DISCRETIZIED:
            if self.env_name == 'LunarLanderContinuous-v2': # use special discrete arrays
                action = torch.stack((self.discrete_array[action_idx[0]], self.split_discrete_array[action_idx[1]]))
            else:
                action = self.discrete_array[action_idx]
        elif self.action_type == ActionType.FIXED_LUNAR:
            action = FIXED_ACTIONS[action_idx]
        else:
            raise NotImplementedError
        return action, action_idx

    def off_policy(self, q_vals):
        """
        Returns off policy (max q value) value for a DQN net
        :param q_vals: Tensor - q values per action
        :return: Int - action to take
        """
        if self.action_type in [ActionType.REGULAR, ActionType.FIXED_LUNAR]:
            if q_vals.dim() > 1:
                q_val, _ = q_vals.max(dim=-1)
            else:
                q_val = q_vals.max()
        elif self.action_type == ActionType.DISCRETIZIED:
            q_val, _ = q_vals.max(dim=1) # this is actually q_vals
        else:
            raise NotImplementedError
        return q_val


class AsyncEnvGen(mp.Process):
    """
    Creates and manages gym environments a-synchroneuosly
    This is used to save time on env.reset() command while playing a game
    """
    def __init__(self, envs, sleep_interval):
        super(AsyncEnvGen, self).__init__()
        self.envs = envs
        self.q = mp.Queue(len(self.envs) - 1)
        self._kill = mp.Event()
        self.env_idx = 0
        self.sleep_interval = sleep_interval

    def run(self):
        while not self._kill.is_set():
            if not self.q.full():
                state = self.envs[self.env_idx].reset()
                self.q.put((state, self.envs[self.env_idx]))
                self.env_idx += 1
                if self.env_idx == len(self.envs):
                    self.env_idx = 0
            elif self.sleep_interval != 0:
                time.sleep(self.sleep_interval)
        self.q.close()
        self.q.cancel_join_thread()

    def get_reset_env(self):
        if self.is_alive():
            return self.q.get()
        else:
            state = self.envs[0].reset()
            return state, self.envs[0]

    def kill(self):
        self._kill.set()
