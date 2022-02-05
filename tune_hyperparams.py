import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) # error only
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
from models import ConvDQN
from dqn_agent import DQNAgent
import utils
from datetime import datetime

device = torch.device('cuda:3')
timestamp = datetime.now().strftime('%y%m%d%H%m%s')
SAVE_DIR = '/home/nir/stereo_augmented_rl/saved_agents/tuning_{}'.format(timestamp)
os.mkdir(SAVE_DIR)
EPOCHS = 1500
BETA = 1e-3
GAMMA = 0.99
PRINT_INTERVAL = 100
LOG_INTERVAL = 0
SCHED_GAMMA = 0.95
SCHED_INTERVAL = 500

lrs = [0.0001, 0.00005]
hidden_sizes = [512, 256, 128, 64]
traj_lens = [1000, 5000, 50000]
eps_policies = ['eps_soft', 'eps_greedy'] #, 'eps_bounded']
for lr in lrs:
    for hidden_size in hidden_sizes:
        for traj_len in traj_lens:
            for idx, eps_policy in enumerate(eps_policies):
                agent_name = 'dqn_{}_{}_{}_{}'.format(lr, hidden_size, traj_len, eps_policy)
                save_path = os.path.join(SAVE_DIR, agent_name)
                print('#'*20, agent_name, '#'*20)
                start = time.time()
                envs = [utils.EnvWrapper('skeleton_plus', utils.ObsType.VIDEO_CONV,
                    utils.ActionType['ACT_WAIT'], 50000, num_discrete=10, debug=False, time_penalty=0.0)]
                env_gen = utils.AsyncEnvGen(envs, 1)
                model = ConvDQN(envs[0].obs_shape, envs[0].num_actions, device=device, hidden_size=hidden_size)
                log_path = './log.log'
                agent = DQNAgent(model, save_path, log_path)
                try:
                    agent.train(epochs=EPOCHS, trajectory_len=traj_len, env_gen=env_gen,
                            lr=lr, discount_gamma=GAMMA, scheduler_gamma=SCHED_GAMMA, beta=BETA,
                            print_interval=PRINT_INTERVAL, log_interval=LOG_INTERVAL, scheduler_interval=SCHED_INTERVAL)
                except Exception as e:
                    envs[0].env.close()
                    raise e
                finally:
                    end = time.time()
                    utils.save_agent(agent)
                    utils.kill_process(env_gen)
                    envs[0].env.close()
                    print('----- training took {:.3f} minutes -----\n'.format((end - start)/60))
