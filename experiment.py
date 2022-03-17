import sys
import os
import time
import multiprocessing as mp
import torch
from models import MultiModalCONVLSTMActorCritic, CONVLSTMActorCritic
from a2c_agent import A2CAgent
import utils
from datetime import datetime

device = torch.device('cuda:3')
# device = torch.device('cuda')
# device = torch.device('cpu')

if torch.cuda.is_available() and (device == torch.device('cuda')):
    sys.stdout.write('Using CUDA {}\n'.format(device))
else:
    sys.stdout.write('Using CPU\n')

timestamp = datetime.now().strftime('%y%m%d%H%m%s')
SAVE_DIR = '/home/nir/stereo_augmented_rl/saved_agents/experiment_{}'.format(timestamp)
# SAVE_DIR = '/users/nirweingarten/Desktop/tuning_{}'.format(timestamp)
# SAVE_DIR = '/content/drive/MyDrive/RL_research/skel_plus/saved_models/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
EPOCHS = 5000
TRAJ_LEN = 50000
FINAL_EXP_TIME = 1000000
BETA = 1e-3
GAMMA = 0.99
PRINT_INTERVAL = 10
LOG_INTERVAL = 0
SCHED_GAMMA = 0.95
SCHED_INTERVAL = 100000
MAX_STEPS = 1000000
REPLAY_INIT_LEN = 50000
LR = 0.00025
HIDDEN_SIZE = 512
FRAMES_TO_SKIP = 1
USE_HISTORY = False
EPSILON = 0
MULTIMODAL_OBS_TYPES = ['VNC_MAX_MONO', 'VNC_MAX_STEREO', 'VNC_FFT_MONO', 'VNC_FFT_STEREO', 'VNC_MEL_MONO', 'VNC_MEL_STEREO']
VIDEO_ONLY_OBS_TYPES = ['VIDEO_ONLY', 'VIDEO_NO_CLUE']


class ExperimentWorker(mp.Process):
    """
    A multiprocess worker that runs one type of agent 3 times
    """
    def __init__(self, obs_type, save_dir, num_runs=3):
        super(ExperimentWorker, self).__init__()
        self.obs_type = obs_type
        self.save_dir = save_dir
        self.num_runs = num_runs
        self._kill = mp.Event()
        self.dir_path = os.path.join(save_dir, obs_type)
        if not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)
        self.env = utils.EnvWrapper('skeleton_plus', utils.ObsType[obs_type],
                                    utils.ActionType['ACT_WAIT'], max_steps=MAX_STEPS, num_discrete=10, debug=False,
                                    time_penalty=0.0, frames_to_skip=FRAMES_TO_SKIP, use_history=USE_HISTORY)

    def run(self):
        for run in range(self.num_runs):
            if self._kill.is_set():
                break
            agent_name = 'run_{}'.format(run)
            save_path = os.path.join(self.save_dir, agent_name)
            log_path = 'logs/{}.log'.format(agent_name)
            if obs_type in MULTIMODAL_OBS_TYPES:
                model = MultiModalCONVLSTMActorCritic(self.env.obs_shape, self.env.num_actions, device=device, hidden_size=HIDDEN_SIZE)
            else:
                model = CONVLSTMActorCritic(self.env.obs_shape, self.env.num_actions, device=device, hidden_size=HIDDEN_SIZE)

            sys.stdout.write("{} {} run {} {}".format("#" * 20, self.obs_type, run, "#" * 20))
            sys.stdout.flush()
            agent = A2CAgent(model, save_path, log_path)
            start = time.time()
            agent.train(epochs=EPOCHS, trajectory_len=TRAJ_LEN, env_wrapper=self.env,
                    lr=LR, discount_gamma=GAMMA, scheduler_gamma=SCHED_GAMMA, beta=BETA,
                    print_interval=PRINT_INTERVAL, log_interval=LOG_INTERVAL, scheduler_interval=SCHED_INTERVAL,
                    epsilon=EPSILON, batch_size=32, epsilon_min=0.1, device=device, replay_init_len=REPLAY_INIT_LEN,
                    final_exp_time=FINAL_EXP_TIME)
            utils.save_agent(agent)
            end = time.time()
            sys.stdout.write('----- training took {:.3f} minutes -----\n'.format((end - start)/60))
            sys.stdout.flush()
        self.env.env.close()
        sys.stdout.write('----- Worker finished: {} -----\n'.format(self.obs_type))
        sys.stdout.flush()

    def kill(self):
        self._kill.set()
        self.env.env.close()

for obs_type in MULTIMODAL_OBS_TYPES + VIDEO_ONLY_OBS_TYPES:
    worker = ExperimentWorker(obs_type, SAVE_DIR)
    worker.start()

