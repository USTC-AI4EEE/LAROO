import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC_ensemble
from torch.utils.tensorboard import SummaryWriter
from utils import robust_loss,func_kurtosis,cumulative_discount,get_qbias_online_data
from TD3_BC_ensemble import TD3_BC_ensemble_robust
# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, exploration_policy, env_name, seed, mean, std, num_nets, t, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		episode_timesteps = 0
		while not done:
			episode_timesteps += 1
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.ensemble_eval_select_action(state)
			
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"ENumber: {num_nets} Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score, avg_reward
	

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC_ensemble")               # Policy name
	parser.add_argument("--env", default="hopper-random-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=2e5, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", action="store_true")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--offdataset", action="store_true")
	parser.add_argument("--cuda_id",default=0,type=int)
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--start_timesteps", default=3e3, type=int)
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=False)
	parser.add_argument('--utd',type=int,default=1)
	# ablation for heavy distribution of Q-error 
	parser.add_argument('--abl_heavydis',type=int,default=0)

	# my paramter
	parser.add_argument('--q_target_mode',type=str,default='min')
	parser.add_argument('--action_mode',type=str,default='ave')
	parser.add_argument('--robust_func',type=str,default='mse')
	parser.add_argument('--alpha_mee',type=float,default=0.01)
	args = parser.parse_args()

	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	# if not os.path.exists("./results"):
	# 	os.makedirs("./results")

	if args.load_model and not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env)
	load_model = args.load_model
	load_offline_data = args.offdataset
	
	mask_prob = 0.9
	num_nets = 5


	Utd = args.utd
	device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
	writer = SummaryWriter(f"online_enoto/{str(args.policy).upper() + '_mode_'+ str(args.action_mode) + '_Rfun_' + str(args.robust_func) + '_UTD_' + str(Utd) +'_Mee_' + str(args.alpha_mee)+ '_' + str(args.env) + '_loadmodel_' + str(load_model) + '_offlinedata_' + str(load_offline_data) + '_alpha_' + str(args.alpha) + '_seed_' + str(args.seed)}/")
	
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha,
		"num_nets": num_nets,
		"device": device,
		"q_target_mode": args.q_target_mode,
		"action_mode": args.action_mode,
		"robust_func": args.robust_func,
		"alpha_mee":args.alpha_mee,
		"env_name":args.env,
	}

	# Initialize policy	
	policy = TD3_BC_ensemble_robust(**kwargs)
	exploration_policy = TD3_BC_ensemble_robust(**kwargs)

	if load_model == True:
		load_path = './results'
		policy.load_policy(seed=args.seed,name=f'LAROO_{args.env}',load_path=load_path)
		print('='*30)
		print('model load done..., file name is ', f'LAROO_{args.env}', '\n', '='*30)

	if load_model == True:
		load_path = './results'
		exploration_policy.load_policy(seed=args.seed,name=f'LAROO_{args.env}',load_path=load_path)
		print('='*30)
		print('exploration model load done..., file name is ', f'LAROO_{args.env}', '\n', '='*30)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, mask_prob, device, max_size=int(1e6))
	
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1

	# We provide the model of the fine-tuned agent after 100k steps
	d4rl_score, avg_reward = eval_policy(policy, exploration_policy, args.env, args.seed, mean, std, num_nets, 0)
	print(d4rl_score)
	

