import sys
import os
import time
import torch
from models import MultiModalCONVLSTMActorCritic, CONVLSTMActorCritic
from a2c_agent import A2CAgent
import utils
from datetime import datetime
import argparse
import pickle
import retro
from utils import Discretizer

DEVICE = torch.device('cuda')
# DEVICE = torch.device('cuda')
# DEVICE = torch.device('cpu')

timestamp = datetime.now().strftime('%y%m%d%H%m%s')
SAVE_DIR = '/home/ubuntu/stereo_augmented_rl/saved_agents/experiment_{}'.format(timestamp)
# SAVE_DIR = '/users/nirweingarten/Desktop/tuning_{}'.format(timestamp)
# SAVE_DIR = '/content/drive/MyDrive/RL_research/skel_plus/saved_models/experiment_{}'.format(timestamp)

EPOCHS = 1500
TRAJ_LEN = 50000
FINAL_EXP_TIME = 1000000
BETA = 1e-3
GAMMA = 0.99125
PRINT_INTERVAL = 10
LOG_INTERVAL = 0
SCHED_GAMMA = 0.93
SCHED_INTERVAL = 100
MAX_STEPS = 1000000
REPLAY_INIT_LEN = 50000
SAVE_INTERVAL = 100
LR = 0.00015
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 1
FRAMES_TO_SKIP = 1
USE_HISTORY = False
EPSILON = 0
MULTIMODAL_OBS_TYPES = ['VNC_MAX_MONO', 'VNC_MAX_STEREO', 'VNC_FFT_MONO', 'VNC_FFT_STEREO', 'VNC_MEL_MONO', 'VNC_MEL_STEREO']
VIDEO_ONLY_OBS_TYPES = ['VIDEO_ONLY', 'VIDEO_NO_CLUE']
TIME_PENALTY = 0.0
SHAPE_REWARD = False


class ExperimentWorker():
    """
    A multiprocess worker that runs one type of agent 3 times
    """
    def __init__(self, obs_type, save_dir, num_runs=1):
        self.obs_type = obs_type
        self.save_dir = save_dir
        self.num_runs = num_runs
        self.save_dir = os.path.join(save_dir, obs_type)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.env = utils.EnvWrapper('skeleton_plus', utils.ObsType[obs_type],
                                    utils.ActionType['ACT_WAIT'], max_steps=MAX_STEPS, num_discrete=10, debug=False,
                                    time_penalty=TIME_PENALTY, frames_to_skip=FRAMES_TO_SKIP, use_history=USE_HISTORY, shape_reward=SHAPE_REWARD)

    def run(self, device):
        for run in range(self.num_runs):
            agent_name = 'run_{}'.format(run)
            save_path = os.path.join(self.save_dir, agent_name)
            log_path = 'logs/{}.log'.format(agent_name)
            if self.obs_type in MULTIMODAL_OBS_TYPES:
                model = MultiModalCONVLSTMActorCritic(self.env.obs_shape, self.env.num_actions, device=device, hidden_size=HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS)
                sys.stdout.write('Chose multi-modal model\n')
                sys.stdout.flush()
            else:
                model = CONVLSTMActorCritic(self.env.obs_shape, self.env.num_actions, device=device, hidden_size=HIDDEN_SIZE, num_lstm_layers=NUM_LSTM_LAYERS)
                sys.stdout.write('Chose single-modal model\n')
                sys.stdout.flush()

            sys.stdout.write("{} {} run {} {}\n".format("#" * 20, self.obs_type, run, "#" * 20))
            sys.stdout.flush()
            agent = A2CAgent(model, save_path, log_path)
            start = time.time()
            agent.train(epochs=EPOCHS, trajectory_len=TRAJ_LEN, env_wrapper=self.env,
                    lr=LR, discount_gamma=GAMMA, scheduler_gamma=SCHED_GAMMA, beta=BETA,
                    print_interval=PRINT_INTERVAL, log_interval=LOG_INTERVAL, scheduler_interval=SCHED_INTERVAL,
                    epsilon=EPSILON, batch_size=32, epsilon_min=0.1, device=device, replay_init_len=REPLAY_INIT_LEN,
                    final_exp_time=FINAL_EXP_TIME, save_interval=SAVE_INTERVAL)
            utils.save_agent(agent)
            end = time.time()
            sys.stdout.write('----- training took {:.3f} minutes -----\n'.format((end - start)/60))
            sys.stdout.flush()
        self.env.env.close()
        sys.stdout.write('----- Worker finished: {} -----\n'.format(self.obs_type))
        sys.stdout.flush()


def main(raw_args):
    parser = argparse.ArgumentParser(description='Creates or load an agent and then trains it')
    parser.add_argument(
        '-load', type=str, nargs='?', help='Weather or not to load an existing agent from the specified path.\n'
                                           'In case of loading all other arguments are ignored', default='')
    parser.add_argument(
        '-device', type=str, nargs='?', help='Device type: cpu, cuda, cuda:0, cuda:1,...', default='cuda')
    parser.add_argument(
        '-epochs', type=int, nargs='?', help='How many epochs to train', default=0)
    parser.add_argument(
        '-save_path', type=str, nargs='?', help='new save path for loaded agent. Defaults to old path', default='')
    parser.add_argument(
        '-obs_type', type=str, nargs='?', help='obs type for new experiment', default='')
    args = parser.parse_args(raw_args)

    if torch.cuda.is_available() and ('cuda' in args.device):
        sys.stdout.write('Using CUDA {}\n'.format(args.device))
    else:
        sys.stdout.write('Using CPU\n')
    sys.stdout.flush()

    # import ipdb
    # ipdb.set_trace()

    if args.load:
        assert os.path.isfile(args.load)
        with open(args.load, 'rb') as f:
            agent = pickle.load(f)
            # In retro one cannot save the env and discretisizer, so we need to recreate them
            # agent.env.env = retro.retro_env.RetroEnv
            agent.env.env = retro.make(game='skeleton_plus', inttype=retro.data.Integrations.ALL)
            agent.env.discretisizer = Discretizer(agent.env.env, [['UP'], ['LEFT'], ['RIGHT'], ['BUTTON'], [None]])
            if args.save_path:
                assert os.path.isdir(os.path.dirname(args.save_path))
                agent.save_path = args.save_path
            sys.stdout.write(f'agent trained for {len(agent.all_rewards)} episodes\nFinal episode scored {agent.all_rewards[-1]}\nObs type: {agent.env.obs_type}\n')
            sys.stdout.flush()
            agent.train(epochs=args.epochs, trajectory_len=TRAJ_LEN, env_wrapper=agent.env, lr=LR, discount_gamma=GAMMA,
                        scheduler_gamma=SCHED_GAMMA, beta=BETA, print_interval=PRINT_INTERVAL,
                        log_interval=LOG_INTERVAL, save_interval=SAVE_INTERVAL, scheduler_interval=SCHED_INTERVAL,
                        clip_gradient=False, eval_interval=0, device=torch.device(args.device))
    else:
        if not os.path.exists(SAVE_DIR):
            try:
                os.makedirs(SAVE_DIR)
            except FileExistsError:
                sys.stdout.write('Directory already exists\n')
                sys.stdout.flush()

        worker = ExperimentWorker(args.obs_type, SAVE_DIR)
        worker.run(torch.device(args.device))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
