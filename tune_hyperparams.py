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
from ddqn_agent import DDQNAgent
import utils
from datetime import datetime

device = torch.device('cuda:3')
# device = torch.device('cpu')
timestamp = datetime.now().strftime('%y%m%d%H%m%s')
SAVE_DIR = '/home/nir/stereo_augmented_rl/saved_agents/tuning_{}'.format(timestamp)
# SAVE_DIR = '/users/nirweingarten/Desktop/tuning_{}'.format(timestamp)
os.mkdir(SAVE_DIR)
EPOCHS = 5000
TRAJ_LEN = 50000
FINAL_EXP_TIME = 5000
BETA = 1e-3
GAMMA = 0.99
PRINT_INTERVAL = 100
LOG_INTERVAL = 0
SCHED_GAMMA = 0.95
SCHED_INTERVAL = 100000
# REPLAY_INIT_LEN = 100
REPLAY_INIT_LEN = 50000

lrs = [0.00025, 0.001]
hidden_sizes = [512, 256, 128]
update_intervals = [10000, 5000, 1000]
backprop_intervals = [1, 10, 32]
frames_to_skip = [4, 1]
use_histories = [False, True]
epsilons = [0, 1]
no_pers = [False, True]
for lr in lrs:
    for hidden_size in hidden_sizes:
        for update_interval in update_intervals:
            for backprop_interval in backprop_intervals:
                for use_history in use_histories:
                    for frame_to_skip in frames_to_skip:
                        for epsilon in epsilons:
                            for no_per in no_pers:
                                agent_name = 'ddqn_{}_{}_{}_{}_{}_{}_{}_{}'.format(lr, hidden_size, frame_to_skip, use_history, update_interval, backprop_interval, epsilon, no_per)
                                save_path = os.path.join(SAVE_DIR, agent_name)
                                print('#'*20, agent_name, '#'*20)
                                start = time.time()
                                envs = [utils.EnvWrapper('skeleton_plus', utils.ObsType.VIDEO_CONV,
                                    utils.ActionType['ACT_WAIT'], 50000, num_discrete=10, debug=False, time_penalty=0.0,
                                    frames_to_skip=frame_to_skip, use_history=use_history)]
                                env_gen = utils.AsyncEnvGen(envs, 1)
                                model = ConvDQN(envs[0].obs_shape, envs[0].num_actions, device=device, hidden_size=hidden_size)
                                log_path = './log.log'
                                agent = DDQNAgent(model, save_path, log_path)
                                try:
                                    agent.train(epochs=EPOCHS, trajectory_len=TRAJ_LEN, env_gen=env_gen,
                                            lr=lr, discount_gamma=GAMMA, scheduler_gamma=SCHED_GAMMA, beta=BETA,
                                            print_interval=PRINT_INTERVAL, log_interval=LOG_INTERVAL, scheduler_interval=SCHED_INTERVAL,
                                            no_per=no_per, epsilon=epsilon, batch_size=32, epsilon_min=0.1, device=device, replay_init_len=REPLAY_INIT_LEN,
                                            update_target_interval=update_interval, backprop_interval=backprop_interval, final_exp_time=FINAL_EXP_TIME)
                                except Exception as e:
                                    envs[0].env.close()
                                    raise e
                                finally:
                                    end = time.time()
                                    utils.save_agent(agent)
                                    utils.kill_process(env_gen)
                                    envs[0].env.close()
                                    print('----- training took {:.3f} minutes -----\n'.format((end - start)/60))
