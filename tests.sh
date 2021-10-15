#!/bin/bash
echo started training...
python agent.py -env LunarLanderContinuous-v2 -model DiscreteActorCritic -epochs 40000 -obs_type BOX2D -action_type DISCRETIZIED -beta 0.0005 -print_interval 100 -save_dir /users/nirweingarten/Desktop/idc/rl/project/git/saved_agents -log_interval 0 -trajectory_len 500 -lr 0.001 -num_discrete 100
echo finished 1
python agent.py -env LunarLanderContinuous-v2 -model DiscreteActorCritic -epochs 40000 -obs_type BOX2D -action_type DISCRETIZIED -beta 0.0005 -print_interval 100 -save_dir /users/nirweingarten/Desktop/idc/rl/project/git/saved_agents -log_interval 0 -trajectory_len 500 -lr 0.001 -num_discrete 1000
echo finished 2
python agent.py -env LunarLanderContinuous-v2 -model DiscreteActorCritic -epochs 40000 -obs_type BOX2D -action_type DISCRETIZIED -beta 0.0005 -print_interval 100 -save_dir /users/nirweingarten/Desktop/idc/rl/project/git/saved_agents -log_interval 0 -trajectory_len 500 -lr 0.001 -num_discrete 10
echo finished 3
python agent.py -env LunarLanderContinuous-v2 -model DiscreteActorCritic -epochs 40000 -obs_type BOX2D -action_type DISCRETIZIED -beta 0.0005 -print_interval 100 -save_dir /users/nirweingarten/Desktop/idc/rl/project/git/saved_agents -log_interval 0 -trajectory_len 500 -lr 0.001 -num_discrete 500
echo finished 4
python agent.py -env LunarLanderContinuous-v2 -model GaussianActorCritic -epochs 40000 -obs_type BOX2D -action_type GAUSSIAN -print_interval 100 -save_dir /users/nirweingarten/Desktop/idc/rl/project/git/saved_agents -log_interval 0 -trajectory_len 500 -lr 0.0001
echo finished 5
echo done.