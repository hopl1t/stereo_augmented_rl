from enum import Enum, auto
import time
import multiprocessing as mp
import torch
from skimage.measure import block_reduce
from torch.distributions import Categorical
import numpy as np
import librosa
import sys
import pickle
# import gym_sokoban # Don't remove this
from copy import deepcopy
import gym
import retro
import random
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
# try:
#     from gym.wrappers import Monitor
# except ModuleNotFoundError as e:
#     sys.stdout.write('Cannot import Monitor module, rendering won\'t be possible: {}\nContinuing..\n'.format(e))
#     sys.stdout.flush()

EPS = 1e-20
OBS_BUFFER_LEN = 4
SAMPLE_RATE = 32000
AUDIO_BUFFER_SHAPE = (524, 2)

class ObsType(Enum):
    GYM = auto()
    VIDEO_ONLY = auto()
    VIDEO_NO_CLUE = auto()
    VNC_BUFFER_MONO = auto()
    VNC_BUFFER_STEREO = auto()
    VNC_MAX_MONO = auto()
    VNC_MAX_STEREO = auto()
    VNC_FFT_MONO = auto()
    VNC_FFT_STEREO = auto()
    VNC_MEL_MONO = auto()
    VNC_MEL_STEREO = auto()
    VIDEO_CONV = auto()
    # TODO: Add RNN with the best of the naive types, and sound only with the best of the naive types


class ActionType(Enum):
    GYM = auto()
    ACT_WAIT = auto()
    FREE = auto()
    NO_WAIT = auto()


class MoveType(Enum):
    UP = 0
    LEFT = 1
    RIGHT = 2
    BUTTON = 3
    NONE = 4


MAX_VOL = 2**14  # 16384
AUDIO_BUFFER_SIZE = 524
SPECTOGRAM_SIZE = 64


class PERDataSet():
    def __init__(self, max_len=1000000, min_len=500):
        self.exp = []
        self.max_len = max_len
        self.min_len = min_len

    def append(self, experience):
        self.exp.append(experience)
        self.exp = self.exp[-self.max_len:] ## CHECK IF THIS IS MEFASTEN

    def per(self, low_ratio=6):
        """
        Removes experiences with low distance between expected and actual q-values ('delta')
        """
        if len(self) >= self.min_len:
            self.exp = sorted(self.exp, key=lambda tup: tup[-1])
            high_delta = self.exp[-len(self.exp) // 2:]
            low_delta = self.exp[:len(self.exp) // 2]
            self.exp = high_delta + random.choices(low_delta, k=len(self.exp) // low_ratio)

    def sample(self, batch_size):
        samples = random.sample(self.exp, batch_size)
        states_vid, states_aud, action_idxs, rewards, new_states_vid, new_states_aud, dones = [], [], [], [], [], [], []
        for sample in samples:
            states_vid.append(sample[0][0])
            states_aud.append(sample[0][1])
            action_idxs.append(sample[1])
            rewards.append(sample[2])
            new_states_vid.append(sample[3][0])
            new_states_aud.append(sample[3][1])
            dones.append(sample[4])
        states = [torch.tensor(np.array(states_vid)), torch.tensor(np.array(states_aud))]
        new_states = [torch.tensor(np.array(new_states_vid)), torch.tensor(np.array(new_states_aud))]
        action_idxs = torch.tensor(action_idxs).unsqueeze(-1).unsqueeze(-1)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)
        return states, action_idxs, rewards, new_states, dones

    def __getitem__(self, idx):
        # returns state, action_idx, reward, new_state, done
        exp = self.exp[idx]
        # return exp[0], exp[1], exp[2], exp[3], exp[4]
        return tuple(exp)

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
        if isinstance(agent.env.env, retro.retro_env.RetroEnv):
            # In retro one cannot save the env and discretisizer, so we need to recreate them
            tmp_env = agent.env.env
            tmp_discretisizer = agent.env.discretisizer
            agent.env.env = None
            agent.env.discretisizer = None
            pickle.dump(agent, f)
            agent.env.env = tmp_env
            agent.env.discretisizer = tmp_discretisizer
        else:
            pickle.dump(agent, f)
    sys.stdout.write('Saved agent to {}\n'.format(agent.save_path))
    sys.stdout.flush()


def log(agent, what_to_log=None):
    if not what_to_log:
        what_to_log = agent.log_buffer
    with open(agent.log_path, 'a') as f:
        _ = f.writelines(what_to_log)
        agent.log_buffer = []
    sys.stdout.write('Logged info to {}\n'.format(agent.log_path))
    sys.stdout.flush()


def print_stats(agent, episode, print_interval, steps_count=0):
    message = "eps: {0}, stats for last {1} eps:\tavg eps reward: {2:.3f}\t\tavg eps step reward: {3:.3f}\t\t" \
              "avg episode score: {4:.3f}\t\tavg eps length: {5:.3f}\t avg time: {6:.3f}\n"\
        .format(episode, print_interval, np.mean(agent.all_rewards[-print_interval:]),
                np.sum(agent.all_rewards[-print_interval:]) / steps_count, np.mean(agent.all_scores[-print_interval:]),
                np.mean(agent.all_lengths[-print_interval:]) + 1, np.mean(agent.all_times[-print_interval:]))
    sys.stdout.write(message)
    sys.stdout.flush()
    log(agent, message)


def print_eval(all_episode_rewards, completed_levels):
    sys.stdout.write('{0} Evaluation {0}'.format('*' * 10))
    sys.stdout.write('\nEvaluation on last 100 episodes:\tmean: {:.3f}\tmin: {:.3f}\t\tmax: {:.3f}\t\t'
                     '%completed levels (sokoban only): {:.3f}\n'.format(np.mean(all_episode_rewards),
                     np.min(all_episode_rewards), np.max(all_episode_rewards), completed_levels / 100))
    sys.stdout.flush()


def evaluate(agent, num_episodes=1, render=True):
    agent.model.eval()
    all_rewards = []
    all_episode_rewards = []
    completed_levels = 0
    for epispode in range(num_episodes):
        if render:
            sys.stdout.write('Saving render video to {}\n'.format(os.path.join(os.getcwd(), 'video')))
            sys.stdout.flush()
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


def clone_model(model):
    """
    Clones a model
    """
    return deepcopy(model)


def shape_reward_func(score, time_from_last_score):
    """
    Run this in Desmos and get an idea on how this shaping looks like
    time <= 250: reards = 5
    time = 700: reward = 1.5
    time = 1500: reward = 0.5
    time >= 10k: reward = 0.1
    I expect the agent to learn to kill in ~700 steps
    """
    if time_from_last_score <= 250:
        multiplier = 5
    elif time_from_last_score <= 4200:
        multiplier = 10000 / (time_from_last_score * np.log2(time_from_last_score))
    else:
        multiplier = 0.2
    return score * multiplier


class EnvWrapper:
    """
    Wrapps a Sokoban gym environment s.t. we can use the room_state property instead of regular state
    """

    def __init__(self, env_name, obs_type=ObsType.VIDEO_ONLY, action_type=ActionType.ACT_WAIT,
                 max_steps=5000, compression_rate=4, kill_hp_ratio=0.05, debug=False,
                 time_penalty=0.005, use_history=False, audio_pooling=4, mel_bands=40, shape_reward=False,  **kwargs):
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
        if env_name == 'skeleton_plus':
            self.env = retro.make(game=env_name, inttype=retro.data.Integrations.ALL)  # 'skeleton_plus'
        elif env_name == 'CartPole-v1':
            self.env = gym.make(env_name)
        else:
            raise NotImplementedError
        self.env_name = env_name
        self.env.max_steps = max_steps
        self.max_steps = max_steps
        self.action_type = action_type
        self.compression_rate = compression_rate
        self.audio_poooling = audio_pooling
        self.mel_bands = mel_bands
        if env_name == 'skeleton_plus':
            self.discretisizer = Discretizer(self.env, [['UP'], ['LEFT'], ['RIGHT'], ['BUTTON'], [None]])
        self.health = 99
        self.score = 0
        # The rewards takes into account both the killed skeletons and the hp lost. hp penalties are muliplied by the kill_hp_ratio
        self.shape_reward = shape_reward
        self.kill_hp_ratio = kill_hp_ratio
        self.debug = debug
        self.time_penalty = time_penalty
        self.frames_to_skip = kwargs.get('frames_to_skip', 1)
        self.use_history = use_history
        self.obs_buffer = []
        self.steps_since_last_score = 0
        obs_shape = self.env.observation_space.shape
        if self.obs_type != ObsType.GYM:
            compressed_y_shape = (int(np.ceil(obs_shape[0] / self.compression_rate)))
            compressed_x_shape = (int(np.ceil(obs_shape[1] / self.compression_rate)))
        # self.obs shape: ((y_dim, x_dim), audio)
        if obs_type == ObsType.VIDEO_ONLY or obs_type == ObsType.VIDEO_NO_CLUE:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), 0)
        elif obs_type == ObsType.VIDEO_CONV:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), 0)
        elif obs_type == ObsType.VNC_BUFFER_MONO:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), AUDIO_BUFFER_SIZE)
        elif obs_type == ObsType.VNC_BUFFER_STEREO:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), AUDIO_BUFFER_SIZE * 2)
        elif obs_type == ObsType.VNC_MAX_MONO:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), 1)
        elif obs_type == ObsType.VNC_MAX_STEREO:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), 2)
        elif obs_type == ObsType.VNC_FFT_MONO:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), (AUDIO_BUFFER_SHAPE[0] // 2 + 1) // audio_pooling)
        elif obs_type == ObsType.VNC_FFT_STEREO:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), 2 * ((AUDIO_BUFFER_SHAPE[0] // 2 + 1) // audio_pooling))
        elif obs_type == ObsType.VNC_MEL_MONO:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), mel_bands)
        elif obs_type == ObsType.VNC_MEL_STEREO:
            self.obs_shape = ((compressed_y_shape, compressed_x_shape), 2 * mel_bands)
        elif obs_type == ObsType.GYM:
            self.obs_shape = ((obs_shape[0], 1), 1)

        if use_history:
            self.obs_shape = ((self.obs_shape[0][0]*OBS_BUFFER_LEN, self.obs_shape[0][1]), self.obs_shape[1])

        if action_type == ActionType.ACT_WAIT:
            self.num_actions = len(MoveType)
            # self.env.action_space.n
        elif action_type == ActionType.FREE:
            raise NotImplementedError
        elif action_type == ActionType.NO_WAIT:
            raise NotImplementedError
        elif action_type == ActionType.GYM:
            self.num_actions = self.env.action_space.n

    def reset(self):
        obs = self.env.reset()
        self.health = 99
        self.score = 0
        return self.process_obs(obs)

    def step(self, action, is_eval=False):
        if self.action_type == ActionType.ACT_WAIT:
            obs, reward, done, info = self.env.step(self.discretisizer.action(action))
            for i in range(self.frames_to_skip):
                if not done:
                    obs, reward_, done, info = self.env.step(self.discretisizer.action(MoveType.NONE.value))
                    reward += reward_
                else:
                    break
        elif self.action_type == ActionType.GYM:
            obs, reward, done, info = self.env.step(action)
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
            assert score_delta >= 0
            if health_delta < 0:
                # This is a bug in the game
                if self.debug:
                    sys.stdout.write('health_delta < 0, info: {}\nskipping...'.format(info))
                    sys.stdout.flush()
                    reward = 0
                    done = True
            else:
                self.health = health
                self.score = score
                # we shape the reward to encourage fast killings of the skeletons - the faster the kill the greater the reward
                if self.shape_reward:
                    if score_delta == 0:
                        self.steps_since_last_score += 1
                    else:  # score_delta > 0 ==> we've killed a skeleton
                        score_delta = shape_reward_func(score, self.steps_since_last_score)
                        self.steps_since_last_score = 0
                # The rewards takes into account both the killed skeletons and the hp lost. hp penalties are muliplied by the kill_hp_ratio
                reward = score_delta - (self.kill_hp_ratio * health_delta) - self.time_penalty
            if health < 10:
                done = True
            if self.debug:
                sys.stdout.write('health: {}\tscore: {}\treward: {}\taction: {}\n'.format(health, score, reward, action))
                sys.stdout.flush()
        return obs, reward, done, info

    @staticmethod
    def obfuscate_clue(obs):
        """
        Obfuscates the clue from the observation
        :param obs: np.array, observation
        :return: np.array, observation without clue
        """
        mask = np.ones((9, 8, 3), dtype=np.uint8) * 255
        obs[178:187, 3:11] = mask  # lower left clue
        obs[3:12, 3:11] = mask  # upper left clue
        obs[3: 12, 150: 158] = mask  # upper right clue
        obs[178: 187, 150: 158] = mask  # lower right clue
        obs[90:99, 150:158] = mask # middle right clue
        obs[90:99, 3:11] = mask # middle left clue
        obs[90:99, 76:84] = mask # middle middle clue
        return obs

    def compress_obs(self, obs):
        """
        Compresses the observation
        :param obs: np.array, observation
        :return: np.array, compressed observation
        """
        # return obs[:, :, 0][::self.compression_rate, ::self.compression_rate].flatten().astype(np.bool8)
        # return obs.sum(axis=2, dtype=np.uint8)[::self.compression_rate, ::self.compression_rate].flatten().clip(max=1)
        return obs.sum(axis=2, dtype=np.uint8)[::self.compression_rate, ::self.compression_rate].clip(max=1)

    @staticmethod
    def normalize_sound(audio):
        """
        Maps audio to be in a scale from -1 to 1
        :param audio: np.array, stereo audio
        :return: np.array, normalized audio
        """
        return np.interp(audio, [0, MAX_VOL], [-1, 1])

    @staticmethod
    def dB(y):
        "Calculate the log ratio of y / max(y) in decibel."
        y = np.abs(y) + EPS
        y /= y.max()
        return 20 * np.log10(y)

    def get_spectrogram(self, buffer, spectrogram_type='fft', pooling=1):
        samples_per_window = len(buffer)
        frequencies = (samples_per_window // 2) + 1
        if spectrogram_type.lower() == 'fft':
            spectrum = np.abs(np.fft.fft(buffer, axis=0)[:frequencies:-1])  # the slicing is to let go of the 0 and the negative parts
        elif spectrogram_type.lower() == 'fft_db':
            spectrum = np.abs(np.fft.fft(buffer, axis=0)[:frequencies:-1])
            spectrum = self.dB(spectrum)
        elif spectrogram_type.lower() == 'mel':
            # Buffer is transposed because in stereo libosa expects a shape of (#num_channels, #num_samples)
            # librosa returns two bins because it uses stft (forces 2 windows) so we mean them. dim=-1 works for both mono and stereo
            # for now I don't use pooling on MEL as it already doest the dimensionality reduction
            mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=buffer.T.astype(np.float16), sr=SAMPLE_RATE, n_fft=samples_per_window, n_mels=40), axis=-1)
            return librosa.power_to_db(mel_spectrogram).flatten()
        if spectrum.ndim == 1:
            pool_mask = (pooling,)
        else:
            pool_mask = (pooling, 1)
        return block_reduce(spectrum, pool_mask, np.max).flatten()

    def process_obs(self, obs):
        if self.obs_type == ObsType.VIDEO_ONLY:
            obs = (self.compress_obs(obs), np.zeros(SPECTOGRAM_SIZE))
        elif self.obs_type == ObsType.VIDEO_NO_CLUE:
            obs = self.obfuscate_clue(obs)
            obs = (self.compress_obs(obs), np.zeros(SPECTOGRAM_SIZE))
        elif self.obs_type == ObsType.VIDEO_CONV:
            obs = self.obfuscate_clue(obs)
            obs = (self.compress_obs(obs), np.zeros(SPECTOGRAM_SIZE))
        elif self.obs_type == ObsType.VNC_BUFFER_MONO:
            stereo = self.env.em.get_audio()
            mono = stereo.sum(axis=1).astype(np.int16) / 2
            mono = self.normalize_sound(mono)
            obs = (self.compress_obs(obs), mono)
        elif self.obs_type == ObsType.VNC_BUFFER_STEREO:
            stereo = self.env.em.get_audio()
            # obs = (self.compress_obs(obs), stereo.flatten())
            obs = (self.compress_obs(obs), stereo)
        elif self.obs_type == ObsType.VNC_MAX_MONO:
            stereo = self.env.em.get_audio()
            max_mono = stereo.max().astype(np.int16)
            mono = self.normalize_sound(max_mono)
            obs = (self.compress_obs(obs), np.array(mono))
        elif self.obs_type == ObsType.VNC_MAX_STEREO:
            stereo = self.env.em.get_audio()
            max_stereo = stereo.max(axis=0).astype(np.int16)
            max_stereo = self.normalize_sound(max_stereo)
            obs = (self.compress_obs(obs), max_stereo)
        elif self.obs_type == ObsType.VNC_FFT_MONO:
            stereo = self.env.em.get_audio()
            mono = np.mean(stereo, axis=-1)
            obs = (self.compress_obs(obs), self.get_spectrogram(mono, spectrogram_type='fft_db', pooling=self.audio_poooling))
        elif self.obs_type == ObsType.VNC_FFT_STEREO:
            stereo = self.env.em.get_audio()
            obs = (self.compress_obs(obs), self.get_spectrogram(stereo, spectrogram_type='fft_db', pooling=self.audio_poooling))
        elif self.obs_type == ObsType.VNC_MEL_MONO:
            stereo = self.env.em.get_audio()
            mono = np.mean(stereo, axis=-1)
            obs = (self.compress_obs(obs), self.get_spectrogram(mono, spectrogram_type='mel'))
        elif self.obs_type == ObsType.VNC_MEL_STEREO:
            stereo = self.env.em.get_audio()
            obs = (self.compress_obs(obs), self.get_spectrogram(stereo, spectrogram_type='mel'))
        elif self.obs_type == ObsType.GYM:
            obs = (obs, np.zeros(SPECTOGRAM_SIZE))

        if self.use_history:  # using OBS_BUFFER_LEN last observations as input as per https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
            if len(self.obs_buffer) == 0:  # first run (after reset)
                for i in range(OBS_BUFFER_LEN):
                    self.obs_buffer.append(obs)
            else:
                self.obs_buffer.append(obs)
                self.obs_buffer = self.obs_buffer[1:]
            if len(self.obs_buffer) > OBS_BUFFER_LEN:  # this is just a sanity check
                raise BufferError('obs buffer is too long')
            vid_obs = np.concatenate([tup[0] for tup in self.obs_buffer], axis=0)
            aud_obs = np.concatenate([tup[1] for tup in self.obs_buffer], axis=0)
            obs = (vid_obs, aud_obs)

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
    def on_policy(q_vals, eps=0, is_eval=False, eps_bounded=False):
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
        elif eps_bounded:  # combination of soft and greedy
            if np.random.rand() <= eps:  # this is the greedy policy
                action_idx = torch.randint(0, q_vals.shape[-1], (q_vals.shape[0], 1)).to(q_vals.device)
            else:  # this is the soft policy
                activated = F.softmax(q_vals, dim=1)
                action_idx = torch.multinomial(activated, 1)
        elif eps:  # epsilon greedy option
            if np.random.rand() <= eps:
                action_idx = torch.randint(0, q_vals.shape[-1], (q_vals.shape[0], 1)).to(q_vals.device)
            else:
                action_idx = torch.argmax(q_vals, axis=-1).view(q_vals.shape[0], -1).to(q_vals.device)
        else:  # epsilon soft option
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
            env = self.envs[0]
            state = env.reset()
            return state, env

    def kill(self):
        self._kill.set()


class RetroEnvProcess(mp.Process):
    def __init__(self, args):
        super(RetroEnvProcess, self).__init__()
        self._kill = mp.Event()
        self.is_ready = False
        self.state = None
        self.env = None
        self.args = args
        self.sleep_interval = args.async_sleep_interval

    def run(self):
        while not self._kill.is_set():
            if not self.is_ready:
                self.env = EnvWrapper(self.args.env, ObsType[self.args.obs_type], ActionType[self.args.action_type],
                                      self.args.max_len, num_discrete=self.args.num_discrete, debug=self.args.debug,
                                      time_penalty=self.args.time_penalty)
                self.state = self.env.reset()
                self.is_ready = True
            if self.sleep_interval != 0:
                time.sleep(self.sleep_interval)

    def get_state_env(self):
        while not self.is_ready:
            time.sleep(self.sleep_interval)
        return self.state, self.env

    def reset(self):
        if self.env:
            self.env.close()
        self.is_ready = False

    def kill(self):
        self._kill.set()


class AsyncRetroEnvGen:
    """
    Creates and manages retro environments a-synchroneuosly
    This is used to save time on env.reset() command while playing a game
    This is a special case for retro lib since it's not possible to have two envs together
    """
    def __init__(self, args):
        self.args = args
        self.num_envs = args.num_envs
        self.envs = [RetroEnvProcess(args) for _ in range(args.num_envs)]
        self.env_idx = 0
        self.sleep_interval = args.async_sleep_interval

    def get_reset_env(self):
        self.envs[self.env_idx].reset()
        self.env_idx += 1
        if self.env_idx == self.num_envs:
            self.env_idx = 0
        state, env = self.envs[self.env_idx].get_state_env()
        return state, env

    def start(self):
        for env in self.envs:
            env.start()

    def kill(self):
        for env in self.envs:
            env.kill()
