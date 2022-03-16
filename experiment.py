import gym
# from gym import logger as gymlogger
from gym.wrappers import Monitor
# gymlogger.set_level(40) # error only
import numpy as np
import random
import math
import glob
import io
import base64
import sys
import os
import time
import pandas as pd
import pickle
import torch
from models import ConvDQN, SimpleMultiModalActorCritic
from ddqn_agent import DDQNAgent
from a2c_agent import A2CAgent
import utils
from datetime import datetime

# device = torch.device('cuda:3')
device = torch.device('cuda')
# device = torch.device('cpu')

if torch.cuda.is_available() and (device == torch.device('cuda')):
    sys.stdout.write('Using CUDA {}\n'.format(device))
else:
    sys.stdout.write('Using CPU\n')

timestamp = datetime.now().strftime('%y%m%d%H%m%s')
# SAVE_DIR = '/home/nir/stereo_augmented_rl/saved_agents/tuning_{}'.format(timestamp)
# SAVE_DIR = '/users/nirweingarten/Desktop/tuning_{}'.format(timestamp)
SAVE_DIR = '/content/drive/MyDrive/RL_research/skel_plus/saved_models/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
EPOCHS = 1000
TRAJ_LEN = 50000
FINAL_EXP_TIME = 1000000
BETA = 1e-3
GAMMA = 0.99
PRINT_INTERVAL = 10
LOG_INTERVAL = 0
SCHED_GAMMA = 0.95
SCHED_INTERVAL = 100000
# REPLAY_INIT_LEN = 100
REPLAY_INIT_LEN = 50000

LR = 0.00025
HIDDEN_SIZE = 256
FRAMES_TO_SKIP = 4
USE_HISTORY = False
EPSILON = 0
OBS_TYPES = ['VNC_MAX_MONO', 'VNC_MAX_STEREO', 'VNC_FFT_MONO', 'VNC_FFT_STEREO', 'VNC_MEL_MONO', 'VNC_MEL_STEREO']

for obs_type in OBS_TYPES:
    agent_name = 'experiment_{}'.format(obs_type)
    save_path = os.path.join(SAVE_DIR, agent_name)
    print('#'*20, agent_name, '#'*20)
    start = time.time()
    env = utils.EnvWrapper('skeleton_plus', utils.ObsType[obs_type],
        utils.ActionType['ACT_WAIT'], max_steps=50000, num_discrete=10, debug=False, time_penalty=0.0,
        frames_to_skip=FRAMES_TO_SKIP, use_history=USE_HISTORY)
    model = SimpleMultiModalActorCritic(env.obs_shape, env.num_actions, device=device, hidden_size=HIDDEN_SIZE)
    log_path = 'logs/{}.log'.format(agent_name)
    agent = A2CAgent(model, save_path, log_path)
    agent.train(epochs=EPOCHS, trajectory_len=TRAJ_LEN, env_wrapper=env,
            lr=LR, discount_gamma=GAMMA, scheduler_gamma=SCHED_GAMMA, beta=BETA,
            print_interval=PRINT_INTERVAL, log_interval=LOG_INTERVAL, scheduler_interval=SCHED_INTERVAL,
            epsilon=EPSILON, batch_size=32, epsilon_min=0.1, device=device, replay_init_len=REPLAY_INIT_LEN,
            final_exp_time=FINAL_EXP_TIME)
    end = time.time()
    utils.save_agent(agent)
    env.env.close()
    print('----- training took {:.3f} minutes -----\n'.format((end - start)/60))
