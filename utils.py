from enum import Enum
import time
import multiprocessing as mp
import torch
from torch.distributions import Categorical
import numpy as np
import sys
import pickle
# import gym_sokoban # Don't remove this
import gym
import retro
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
    ACT_WAIT = 1
    FREE = 2
    NO_WAIT = 3


class MoveType(Enum):
    UP = 0
    LEFT = 1
    RIGHT = 2
    BUTTON = 3
    NONE = 4


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


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        combos: ordered list of lists of valid button combinations
    """
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy().astype(np.int8)

    def reverse_action(self, action):
        raise NotImplementedError


def kill_process(p):
    if p.is_alive():
        p.q.cancel_join_thread()
        p.kill()
        p.join(1)


def get_health_score(tens, ones):
    """
    Fixes the wired way variables are saved in Skeleton plus's memory
    """
    return (((tens - 47) // 5) * 10) + ((ones - 47) // 5)


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
    completed_levels = 0
    for epispode in range(num_episodes):
        if render:
            sys.stdout.write('Saving render video to {}\n'.format(os.path.join(os.getcwd(), 'video')))
            agent.env.env = Monitor(agent.env.env, './video', force=True)
        episode_rewards = []
        obs = agent.env.reset()
        done = False
        while not done:
            with torch.no_grad():
                level = agent.env.score // 10
                action = agent.act(obs)
                obs, reward, done, info = agent.env.step(action, is_eval=True)
            if (agent.env.score // 10) > level:
                    completed_levels += 1
            all_rewards.append(reward)
            episode_rewards.append(reward)
        all_episode_rewards.append(np.sum(episode_rewards))
    if render:
        agent.env.env.close()
    agent.model.train()
    return all_rewards, all_episode_rewards, completed_levels


class EnvWrapper:
    """
    Wrapps a Sokoban gym environment s.t. we can use the room_state property instead of regular state
    """

    def __init__(self, env_name, obs_type=ObsType.VIDEO_ONLY, action_type=ActionType.ACT_WAIT,
                 max_steps=5000, compression_rate=4, kill_hp_ratio=0.05, debug=False,
                 time_penalty=0.005, **kwargs):
        """
        Wraps a gym environment s.t. you can control it's input and output
        :param env_name: str, The environments name
        :param obs_type: ObsType, type of output for environment's observations
        :param compression_rate: video compression rate
        :param args: Any args you want to pass to make()
        :param kill_hp_ratio: float. way to compute the cost of each hp in the reward as a function of
         how many kill I expect. Example 5 kills / 100 hp = 0.05
        :param kwargs: Any kwargs you want to pass to make()
        """
        self.obs_type = obs_type
        self.env = retro.make(game=env_name, inttype=retro.data.Integrations.ALL)  # 'skeleton_plus'
        self.env_name = env_name
        self.env.max_steps = max_steps
        self.max_steps = max_steps
        self.action_type = action_type
        self.compression_rate = compression_rate
        self.discretisizer = Discretizer(self.env, [['UP'], ['LEFT'], ['RIGHT'], ['BUTTON'], [None]])
        self.health = 99
        self.score = 0
        self.kill_hp_ratio = kill_hp_ratio
        self.debug = debug
        self.time_penalty = time_penalty
        if obs_type == ObsType.VIDEO_ONLY:
            obs_shape = self.env.observation_space.shape
            self.obs_size = (int(np.ceil(obs_shape[0] / self.compression_rate))) * \
                            (int(np.ceil(obs_shape[1] / self.compression_rate)))
        elif obs_type == ObsType.VIDEO_NO_CLUE:
            raise NotImplementedError
        elif obs_type == ObsType.VIDEO_MONO:
            raise NotImplementedError
        elif obs_type == ObsType.VIDEO_STEREO:
            raise NotImplementedError
        if action_type == ActionType.ACT_WAIT:
            self.num_actions = len(MoveType)
            # self.env.action_space.n
        elif action_type == ActionType.FREE:
            raise NotImplementedError
        elif action_type == ActionType.NO_WAIT:
            raise NotImplementedError

    def reset(self):
        obs = self.env.reset()
        self.health = 99
        self.score = 0
        return self.process_obs(obs)

    def step(self, action, is_eval=False):
        if self.action_type == ActionType.ACT_WAIT:
            _, reward1, done1, _ = self.env.step(self.discretisizer.action(action))
            obs, reward2, done2, info = self.env.step(self.discretisizer.action(MoveType.NONE.value))
            reward = reward1 + reward2
            done = done1 | done2
        else:
            raise NotImplementedError
        obs = self.process_obs(obs)
        if self.env_name == 'skeleton_plus':
            health = get_health_score(info['health_tens'], info['health_ones'])
            score = get_health_score(info['score_tens'], info['score_ones'])
            # Trying to fix the negative health_delta bug
            if done:
                health_delta = 10
            else:
                health_delta = self.health - health
            score_delta = score - self.score
            # TODO: removes these asserts once we know the program runs smoothly
            assert health_delta >= 0
            assert score_delta >= 0
            self.health = health
            self.score = score
            reward = score_delta - (self.kill_hp_ratio * health_delta) - self.time_penalty
            if health < 10:
                done = True
            # done =
            if self.debug:
                print('health: {}\tscore: {}\treward: {}\taction: {}'.format(health, score, reward, action))
        return obs, reward, done, info

    def process_obs(self, obs):
        if self.obs_type == ObsType.VIDEO_ONLY:
            obs = obs[:, :, 0][::self.compression_rate, ::self.compression_rate].flatten().astype(np.bool8)
        elif self.obs_type == ObsType.VIDEO_NO_CLUE:
            raise NotImplementedError
        elif self.obs_type == ObsType.VIDEO_MONO:
            raise NotImplementedError
        elif self.obs_type == ObsType.VIDEO_STEREO:
            raise NotImplementedError
        return obs

    @staticmethod
    def process_action(dist, policy_dist, is_eval=False):
        if is_eval:
            action = torch.argmax(dist, -1).item()
        else:
            action = torch.multinomial(dist, 1).item()
        log_prob = torch.log(policy_dist[action])
        entropy = Categorical(probs=dist).entropy()
        return action, log_prob, entropy

    @staticmethod
    def on_policy(q_vals, eps=0, is_eval=False):
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
        action = action_idx.item()
        return action, action_idx

    def off_policy(self, q_vals):
        """
        Returns off policy (max q value) value for a DQN net
        :param q_vals: Tensor - q values per action
        :return: Int - action to take
        """
        if q_vals.dim() > 1:
            q_val, _ = q_vals.max(dim=-1)
        else:
            q_val = q_vals.max()
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
