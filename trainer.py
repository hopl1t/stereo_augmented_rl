import argparse
import sys
import os
import models
import utils
import pickle
from datetime import datetime
from a2c_agent import A2CAgent
from dqn_agent import DQNAgent


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
                        help='Type of observation to use - either REGULAR, ROOM_STATE_VECTOR or ROOM_STATE_MATRIX',
                        default='REGULAR')
    parser.add_argument('-action_type', type=str, nargs='?',
                        help='Type of action to use - wither REGULAR, PUSH_ONLY, PUSH_PULL', default='REGULAR')
    parser.add_argument('-epochs', type=int, nargs='?', help='Num epochs (episodes) to train', default=3000)
    parser.add_argument('-trajectory_len', type=int, nargs='?', help='Maximal length of single trajectory', default=300)
    parser.add_argument('-lr', type=float, nargs='?', help='Learning rate', default=3e-4)
    parser.add_argument('-trick_fine', type=float, nargs='?', help='Default fine for one of the tricks', default=300)
    parser.add_argument('-discount_gamma', type=float, nargs='?', help='Discount factor', default=0.99)
    parser.add_argument('-scheduler_gamma', type=float, nargs='?', help='Scheduling factor', default=0.999)
    parser.add_argument('-scheduler_interval', type=int, nargs='?', help='Interval to step scheduler', default=1000)
    parser.add_argument('-beta', type=float, nargs='?', help='Info loss factor', default=1e-3)
    parser.add_argument('-epsilon', type=float, nargs='?', help='Epsilon for epsilon greedy policy for DQN only.'
                                                            ' Default is 0 and then epsilon-soft is used', default=0)
    parser.add_argument('-epsilon_decay', type=float, nargs='?', help='Epsilon decay for epsilon greedy policy (DQN only). '
                                                                  'Default is 0.997', default=0.997)
    parser.add_argument('-std_bias', type=float, nargs='?', help='std bias for softplus rectification if using gaussian'
                        , default=5)
    parser.add_argument('-print_interval', type=int, nargs='?', help='Print stats to screen evey x steps', default=1000)
    parser.add_argument('-log_interval', type=int, nargs='?', help='Log stats to file evey x steps. '
                                                                   'Set 0 for no logs at all', default=1000)
    parser.add_argument('-max_len', type=int, nargs='?', help='Maximal steps for a single episode', default=5000)
    parser.add_argument('-hidden_size', type=int, nargs='?', help='Size of largest hidden layer', default=512)
    parser.add_argument('-save_interval', type=int, nargs='?', help='Save every x episodes', default=10000)
    parser.add_argument('-batch_size', type=int, nargs='?', help='Batch size for PER', default=64)
    parser.add_argument('-eval_interval', type=int, nargs='?', help='Evaluate model every x steps.'
                                                                    ' 0 is don\'t eval during training', default=0)
    parser.add_argument('-stop_trick_at', type=int, nargs='?', help='Stop the trick after this epoch.'
                                                                    '0 is don\'t stop', default=0)
    parser.add_argument('-async_sleep_interval', type=float, nargs='?', help='How long should the env gen thread sleep',
                        default=1e-2)
    parser.add_argument('-num_envs', type=int, nargs='?', help='Number of async envs to use if using async_env.'
                                                               ' default 2', default=2)
    parser.add_argument('-no_cuda', action='store_true', help='Flag. If specified do not use cuda',default=False)
    parser.add_argument(
        '-cone_trick', action='store_true', help='Flag. If specified the cone trick for Lunar Lander is used',
        default=False)
    parser.add_argument(
        '-move_trick', action='store_true', help='Flag. If specified the move trick for Sokoban is used', default=False)
    parser.add_argument(
        '-no_PER', action='store_true', help='Flag. If specified disables the use of PER in DQN agents', default=False)
    parser.add_argument(
        '-clip_gradient', action='store_true', help='Flag. If specified the gradient is clipped during training to '
                                                    'prevent exploding gradient', default=False)
    parser.add_argument('-num_discrete', type=int, nargs='?', help='How many discrete actions to generate for a cont.'
                                                                   ' setting using discrete action space', default=10)

    args = parser.parse_args(raw_args)
    assert os.path.isdir(args.save_dir)
    assert os.path.isdir(args.log_dir)
    envs = [utils.EnvWrapper(args.env, utils.ObsType[args.obs_type], utils.ActionType[args.action_type],
            args.max_len, num_discrete=args.num_discrete, cone_trick=args.cone_trick, move_trick=args.move_trick
                             , trick_fine=args.trick_fine) for _ in range(args.num_envs)]
    env_gen = utils.AsyncEnvGen(envs, args.async_sleep_interval)
    if args.load:
        with open(args.load, 'rb') as f:
            agent = pickle.load(f)
    else:
        model = getattr(models, args.model)(envs[0].obs_size, envs[0].num_actions, hidden_size=args.hidden_size,
                                            num_discrete=args.num_discrete, std_bias=args.std_bias)
        timestamp = datetime.now().strftime('%y%m%d%H%m')
        save_path = os.path.join(args.save_dir, '{0}_{1}_{2}.pkl'.format(args.model, args.env, timestamp))
        log_path = os.path.join(args.log_dir, '{0}_{1}_{2}.log'.format(args.model, args.env, timestamp))
        agent = getattr(sys.modules[__name__], args.agent)(model, save_path, log_path)

    try:
        if args.async_env:
            env_gen.start()
            sys.stdout.write('Started async env_gen process..\n')
        agent.train(args.epochs, args.trajectory_len, env_gen, args.lr,
                    args.discount_gamma, args.scheduler_gamma, args.beta,
                    args.print_interval, args.log_interval, scheduler_interval=args.scheduler_interval,
                    clip_gradient=args.clip_gradient, no_per=args.no_PER, stop_trick_at=args.stop_trick_at,
                    no_cuda=args.no_cuda, save_interval=args.save_interval, epsilon=args.epsilon,
                    epsilon_decay=args.epsilon_decay, eval_interval=args.eval_interval, batch_size=args.batch_size)
    except Exception as e:
        raise e
    finally:
        utils.kill_process(env_gen)
        if env_gen.is_alive():
            env_gen.terminate()
        sys.stdout.write('Killed env gen process\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
