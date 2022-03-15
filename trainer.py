import argparse
import sys
import os
import models
import utils
import pickle
from datetime import datetime
import retro
import torch
from utils import Discretizer
from a2c_agent import A2CAgent # Don't remove this
from dqn_agent import DQNAgent # Don't remove this
from ddqn_agent import DDQNAgent # Don't remove this


def main(raw_args):
    parser = argparse.ArgumentParser(description='Creates or load an agent and then trains it')
    parser.add_argument(
        '-load', type=str, nargs='?', help='Weather or not to load an existing agent from the specified path.\n'
                                           'In case of loading all other arguments are ignored')
    parser.add_argument(
        '-async_env', action='store_true', help='Flag. If present use async environment generation, else don\'t',
        default=False)
    parser.add_argument(
        '-env', type=str, nargs='?', help='desired gym environment, example: "Pong-v0"')
    parser.add_argument(
        '-model', type=str, nargs='?', help='Model class name from models.py to use, example: "ModelClassName"')
    parser.add_argument(
        '-agent', type=str, nargs='?', help='Type of agent to use, either A2CAgent or DQNAgent. defaults to A2C',
        default='A2CAgent')
    parser.add_argument(
        '-save_dir', type=str, nargs='?', help='Path to save and load the agent. Defaults to ./saved_agents',
        default='./saved_agents')
    parser.add_argument(
        '-log_dir', type=str, nargs='?', help='Path to save log files. Defaults to ./logs', default='./logs')
    parser.add_argument('-obs_type', type=str, nargs='?',
                        help='Type of observation to use - either VIDEO_ONLY, VIDEO_NO_CLUE, VIDEO_MONO or VIDEO_STEREO',
                        default='VIDEO_ONLY')
    parser.add_argument('-action_type', type=str, nargs='?',
                        help='Type of action to use - wither ACT_WAIT, FREE, NO_WAIT', default='ACT_WAIT')
    parser.add_argument('-epochs', type=int, nargs='?', help='Num epochs (episodes) to train', default=5000)
    parser.add_argument('-time_penalty', type=float, nargs='?', help='penalty for each turn', default=0.0)
    parser.add_argument('-trajectory_len', type=int, nargs='?', help='Maximal length of single trajectory', default=100)  # this used to be 5000
    parser.add_argument('-lr', type=float, nargs='?', help='Learning rate', default=5e-3)
    parser.add_argument('-discount_gamma', type=float, nargs='?', help='Discount factor', default=0.99)
    parser.add_argument('-scheduler_gamma', type=float, nargs='?', help='Scheduling factor', default=0.95)
    parser.add_argument('-scheduler_interval', type=int, nargs='?', help='Interval to step scheduler', default=1000)
    parser.add_argument('-frames_to_skip', type=int, nargs='?', help='Frames to skip between agent interactions'
                                                                     '1 is the default for ACT_WAIT action type',
                        default=1)
    parser.add_argument('-beta', type=float, nargs='?', help='Info loss factor', default=1e-3)
    parser.add_argument('-epsilon', type=float, nargs='?', help='Epsilon for epsilon greedy policy for DQN only.'
                                                            ' Default is 0 and then epsilon-soft is used', default=0)
    parser.add_argument('-epsilon_decay', type=float, nargs='?', help='Epsilon decay for epsilon greedy policy (DQN only). '
                                                                  'Default is 0.997', default=0.997)
    parser.add_argument('-epsilon_min', type=float, nargs='?', help='Minimal epsilon for esp-greedy dqn. '
                                                                    'Default is 0.1', default=0.1)
    parser.add_argument('-epsilon_bounded', action='store_true', help='Flag. If stated uses epsilon soft and greedy '
                                                                      'together such that a completly random policy is '
                                                                      'executed in a probability that equals the '
                                                                      'epsilon_min param. Otherwise an epsilon soft '
                                                                      'action is chosen', default=False)
    parser.add_argument('-std_bias', type=float, nargs='?', help='std bias for softplus rectification if using gaussian'
                        , default=5)
    parser.add_argument('-print_interval', type=int, nargs='?', help='Print stats to screen evey x steps', default=1000)
    parser.add_argument('-log_interval', type=int, nargs='?', help='Log stats to file evey x steps. '
                                                                   'Set 0 for no logs at all', default=1000)
    parser.add_argument('-max_len', type=int, nargs='?', help='Maximal steps for a single episode', default=50000)  # this used to be 5000
    parser.add_argument('-hidden_size', type=int, nargs='?', help='Size of largest hidden layer', default=512)
    parser.add_argument('-save_interval', type=int, nargs='?', help='Save every x episodes', default=10000)
    parser.add_argument('-batch_size', type=int, nargs='?', help='Batch size for PER', default=64)
    parser.add_argument('-lstm_layers', type=int, nargs='?', help='number of lstm layers if used', default=2)
    parser.add_argument('-eval_interval', type=int, nargs='?', help='Evaluate model every x steps.'
                                                                    ' 0 is don\'t eval during training', default=0)
    parser.add_argument('-compression_rate', type=int, nargs='?', help='Video compression rate for Skeleton+. '
                                                                       'Defaults to 4', default=4)
    parser.add_argument('-kill_hp_ratio', type=float, nargs='?', help='For computing hp factor in reward for Skeleton+ '
                                                                      'Defaults to 0.05', default=0.05)
    parser.add_argument('-async_sleep_interval', type=float, nargs='?', help='How long should the env gen thread sleep',
                        default=1e-2)
    parser.add_argument('-num_envs', type=int, nargs='?', help='Number of async envs to use if using async_env.'
                                                               ' default 2', default=2)
    parser.add_argument('-no_cuda', action='store_true', help='Flag. If specified do not use cuda', default=False)
    parser.add_argument(
        '-no_PER', action='store_true', help='Flag. If specified disables the use of PER in DQN agents', default=False)
    parser.add_argument('-debug', action='store_true', help='Flag. If specified prints debug data', default=False)
    parser.add_argument(
        '-clip_gradient', action='store_true', help='Flag. If specified the gradient is clipped during training to '
                                                    'prevent exploding gradient', default=False)
    parser.add_argument('-num_discrete', type=int, nargs='?', help='How many discrete actions to generate for a cont.'
                                                                   ' setting using discrete action space', default=10)
    parser.add_argument('-replay_init_len', type=int, nargs='?', help='Initialization size of replay buffer for ddqn',
                        default=50000)
    parser.add_argument('-replay_max_len', type=int, nargs='?', help='Max size of replay buffer for ddqn',
                        default=int(1e6))
    parser.add_argument('-update_target_interval', type=int, nargs='?', help='How often to update target network',
                        default=10000)
    parser.add_argument('-backprop_interval', type=int, nargs='?', help='How often to backprop', default=1)
    parser.add_argument('-final_exp_time', type=int, nargs='?', help='Number of frames until eps is minimal',
                        default=int(1e6))
    parser.add_argument('-clip_loss', action='store_true', help='Flag. How often to backprop', default=False)
    parser.add_argument('-use_history', action='store_true', help='Flag. Use last 4 observations as input',
                        default=False)
    args = parser.parse_args(raw_args)
    assert os.path.isdir(args.save_dir)
    assert os.path.isdir(args.log_dir)
    assert ~args.no_PER^('lstm' in args.model.lower())  # can't have PER and LSTM together

    env = utils.EnvWrapper(args.env, utils.ObsType[args.obs_type], utils.ActionType[args.action_type],
            args.max_len, num_discrete=args.num_discrete, debug=args.debug, time_penalty=args.time_penalty,
                             frames_to_skip=args.frames_to_skip, use_history=args.use_history)
    obs_shape = env.obs_shape
    num_actions = env.num_actions

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device('cuda')
        sys.stdout.write('Using CUDA\n')
    else:
        device = torch.device('cpu')
        sys.stdout.write('Using CPU\n')

    if args.load:
        with open(args.load, 'rb') as f:
            agent = pickle.load(f)
            # In retro one cannot save the env and discretisizer, so we need to recreate them
            agent.env.env = retro.retro_env.RetroEnv
            agent.env.discretisizer = Discretizer(envs[0].env, [['UP'], ['LEFT'], ['RIGHT'], ['BUTTON'], [None]])
    else:
        model = getattr(models, args.model)(obs_shape, num_actions, hidden_size=args.hidden_size,
                                            num_discrete=args.num_discrete, std_bias=args.std_bias, device=device,
                                            num_lstm_layers=args.lstm_layers)
        timestamp = datetime.now().strftime('%y%m%d%H%m')
        save_path = os.path.join(args.save_dir, '{0}_{1}_{2}.pkl'.format(args.model, args.env, timestamp))
        log_path = os.path.join(args.log_dir, '{0}_{1}_{2}.log'.format(args.model, args.env, timestamp))
        agent = getattr(sys.modules[__name__], args.agent)(model, save_path, log_path,
                                                           replay_max_len=args.replay_max_len)

    try:
        agent.train(args.epochs, args.trajectory_len, env, args.lr,
                    args.discount_gamma, args.scheduler_gamma, args.beta,
                    args.print_interval, args.log_interval, scheduler_interval=args.scheduler_interval,
                    clip_gradient=args.clip_gradient, no_per=args.no_PER, no_cuda=args.no_cuda,
                    save_interval=args.save_interval, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
                    eval_interval=args.eval_interval, batch_size=args.batch_size, epsilon_min=args.epsilon_min,
                    epsilon_bounded=args.epsilon_bounded, device=device, replay_init_len=args.replay_init_len,
                     update_target_interval=args.update_target_interval, backprop_interval=args.backprop_interval,
                    final_exp_time=args.final_exp_time, clip_loss=args.clip_loss)
    except Exception as e:
        raise e


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
