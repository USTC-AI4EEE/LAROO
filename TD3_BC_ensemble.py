import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import robust_loss,func_kurtosis
import os
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1



MIN_BIAS = -200
MAX_BIAS = 200
scratch_policy = ['hopper-random-v2','walker2d-random-v2']

class TD3_BC_ensemble_robust(object):

	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		num_nets=1,
		device=None,
		action_mode = 'ave',
		q_target_mode = 'min',
		robust_func='huber',
		alpha_mee=0.01,
		env_name='hopper-random-v2'
	):

		self.device = device
		self.num_nets = num_nets
		assert self.num_nets == 5, "num_bets is not 5"
		self.L_actor, self.L_actor_target, self.L_critic, self.L_critic_target = [], [], [], []
		for _ in range(self.num_nets):
			self.L_actor.append(Actor(state_dim, action_dim, max_action).to(self.device))
			self.L_actor_target.append(Actor(state_dim, action_dim, max_action).to(self.device))
			self.L_critic.append(Critic(state_dim, action_dim).to(self.device))
			self.L_critic_target.append(Critic(state_dim, action_dim).to(self.device))
		self.L_actor_optimizer, self.L_critic_optimizer = [], []
		for en_index in range(self.num_nets):
			self.L_actor_optimizer.append(torch.optim.Adam(self.L_actor[en_index].parameters(), lr=3e-4))
			self.L_critic_optimizer.append(torch.optim.Adam(self.L_critic[en_index].parameters(), lr=3e-4))

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0
		self.action_mode = action_mode
		self.robust_func = robust_func
		self.q_target_mode = q_target_mode
		self.scale = 1.0
		self.alpha_mee = alpha_mee
		self.env_name = env_name
		
	def calculate_scale(self,q,target_q):
		batch_size = int(target_q.shape[0])
		# adaptive estimate the scale parameter, take the mean u
		with torch.no_grad():
			bias_1 = target_q.detach() - q.detach() 
			qbias_clip = torch.clamp(bias_1,MIN_BIAS,MAX_BIAS) 
			variance = torch.var(qbias_clip,dim=0,unbiased=False)
			bias_mean = torch.mean(qbias_clip, dim=0)
			fourth_moment = torch.mean((qbias_clip - bias_mean) ** 4, dim=0)
			kurt = fourth_moment / (variance ** 2) - 3
			estimate_variance = variance/(kurt/batch_size + (batch_size + 1)/(batch_size - 1))
			deviation = torch.sqrt(estimate_variance/2).item() #
		return deviation

	def ensemble_eval_select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		a = None
		for en_index in range(self.num_nets):
			_a = self.L_actor[en_index](state).cpu().data.numpy().flatten()
			if en_index == 0:
				a = _a
			else:
				a += _a
		a = a / self.num_nets
		# a = self.L_actor[0](state).cpu().data.numpy().flatten()
		return a


	
	def estimation_q(self,state,action,ac_index=0):
		if self.action_mode == 'allmin':
			qvalue_list = []
			for en_index in range(self.num_nets):
				# Compute TD3 actor losse
				current_Q1, current_Q2 = self.L_critic[en_index](state, action)
				qvalue_list.append(torch.min(current_Q1, current_Q2))
			q_a_tilda_cat = torch.cat(qvalue_list, 1)
			# ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
			ave_q = torch.min(q_a_tilda_cat, dim=1, keepdim=True)[0]
			

		elif self.action_mode == 'ave':
			qvalue_list = []
			for en_index in range(self.num_nets):
				# Compute TD3 actor losse
				current_Q1, current_Q2 = self.L_critic[en_index](state, action)
				qvalue_list.append(torch.min(current_Q1, current_Q2))
			q_a_tilda_cat = torch.cat(qvalue_list, 1)
			ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
			# ave_q = torch.min(q_a_tilda_cat, dim=1, keepdim=True)[0]
			

		elif self.action_mode == 'sample_min':
			qvalue_list = []
			sample_idxs = np.random.choice(self.num_nets, 2, replace=False)
			for sample_idx in sample_idxs:
				current_Q1, current_Q2 = self.L_critic[sample_idx](state, action)
				qvalue_list.append(torch.min(current_Q1, current_Q2))
			q_a_tilda_cat = torch.cat(qvalue_list, 1)
			min_q, min_indices = torch.min(q_a_tilda_cat, dim=1, keepdim=True)
			ave_q = min_q

		elif self.action_mode == 'corresponding' or self.action_mode == 'cor':
			ave_q = self.L_critic[ac_index].Q1(state, action)
		
		return ave_q
         

	def train(self, replay_buffer, offline_replay_buffer, batch_size=256, t=None, Utd=None):
		self.total_it += 1
		
		# Sample replay buffer 
		online_batch_size = batch_size * Utd
		offline_batch_size = batch_size * Utd

		online_state, online_action, online_next_state, online_reward, online_not_done = replay_buffer.sample(online_batch_size)
		offline_state, offline_action, offline_next_state, offline_reward, offline_not_done = offline_replay_buffer.sample(offline_batch_size)
		
		for i in range(Utd):
			state = torch.concat([online_state[batch_size*i:batch_size*(i+1)], offline_state[batch_size*i:batch_size*(i+1)]])
			action = torch.concat([online_action[batch_size*i:batch_size*(i+1)], offline_action[batch_size*i:batch_size*(i+1)]])
			next_state = torch.concat([online_next_state[batch_size*i:batch_size*(i+1)], offline_next_state[batch_size*i:batch_size*(i+1)]])
			reward = torch.concat([online_reward[batch_size*i:batch_size*(i+1)], offline_reward[batch_size*i:batch_size*(i+1)]])
			not_done = torch.concat([online_not_done[batch_size*i:batch_size*(i+1)], offline_not_done[batch_size*i:batch_size*(i+1)]])
			
			for en_index in range(self.num_nets):
				with torch.no_grad():
					# Select action according to policy and add clipped noise
					noise = (
						torch.randn_like(action) * self.policy_noise
					).clamp(-self.noise_clip, self.noise_clip)
					
					next_action = (
						self.L_actor_target[en_index](next_state) + noise
					).clamp(-self.max_action, self.max_action)

					# Compute the target Q value
					target_Q1, target_Q2 = self.L_critic_target[en_index](next_state, next_action)
					target_Q = torch.min(target_Q1, target_Q2)
					target_Q = reward + not_done * self.discount * target_Q

				# Get current Q estimates
				current_Q1, current_Q2 = self.L_critic[en_index](state, action)
				
				# calculate the scale
				self.scale = self.calculate_scale(torch.min(current_Q1,current_Q2),target_Q)

				# Compute critic loss
				# critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
				critic_loss = robust_loss(self.robust_func,current_Q1 - target_Q, 3.0).mean() + \
					robust_loss(self.robust_func,current_Q2 - target_Q, 3.0).mean()
				
				# Optimize the critic
				self.L_critic_optimizer[en_index].zero_grad()
				critic_loss.backward()
				self.L_critic_optimizer[en_index].step()

				# Update the frozen target models
				for param, target_param in zip(self.L_critic[en_index].parameters(), self.L_critic_target[en_index].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
			# actor UTD 
			for en_index in range(self.num_nets):
				# Compute TD3 actor losse
				# actor_loss = -self.L_critic[en_index].Q1(state, self.L_actor[en_index](state)).mean()
				q = self.estimation_q(state, self.L_actor[en_index](state),en_index)
				actor_loss = -q.mean()
				# Optimize the actor 
				self.L_actor_optimizer[en_index].zero_grad()
				actor_loss.backward()
				self.L_actor_optimizer[en_index].step()

				for param, target_param in zip(self.L_actor[en_index].parameters(), self.L_actor_target[en_index].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
		loss_dict = {
			'loss/Q1':current_Q1.mean().detach().cpu().numpy(),
			'loss/actor':actor_loss.item(),
			'loss/critic':critic_loss.item(),
		}
		return loss_dict


	def load(self, policy_file, file_name):
		if self.env_name in scratch_policy:
			print('model load done...')
		else:
			for en_index in range(self.num_nets):
				self.L_critic[en_index].load_state_dict(torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_critic", map_location=self.device))
				self.L_critic_optimizer[en_index].load_state_dict(torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_critic_optimizer", map_location=self.device))
				self.L_critic_target[en_index] = copy.deepcopy(self.L_critic[en_index])

				self.L_actor[en_index].load_state_dict(torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_actor", map_location=self.device))
				self.L_actor_optimizer[en_index].load_state_dict(torch.load(f"{policy_file}/{file_name}_agent_{str(en_index)}" + "_actor_optimizer", map_location=self.device))
				self.L_actor_target[en_index] = copy.deepcopy(self.L_actor[en_index])
				print('model ', en_index, ' load done...')


	def save_policy(self,seed,name,save_path : str):
		checkpoint = dict()
		
		for en_index in range(self.num_nets):
			checkpoint.update({
				f'actor_state_{en_index}': self.L_actor[en_index].state_dict()
			})
			checkpoint.update({
				f'critic_state_{en_index}':self.L_critic[en_index].state_dict(),
			})
			checkpoint.update({
				f'actor_optimizer_state_{en_index}': self.L_actor_optimizer[en_index].state_dict()
			})
			checkpoint.update({
				f'critic_optimizer_state_{en_index}':self.L_critic_optimizer[en_index].state_dict(),
			})

		fpath = os.path.join(save_path, "{}_seed{}.pth".format(name, str(seed)))

		torch.save(checkpoint, fpath)

	# load from pretrained TD3BC
	def load_policy(self,seed,name,load_path : str,):
			
		fpath = os.path.join(load_path, "{}_seed{}.pth".format(name, str(seed)))
		# checkpoint = torch.load(fpath)
		checkpoint = torch.load(fpath, map_location=self.device)
		for en_index in range(self.num_nets):
			self.L_critic[en_index].load_state_dict(checkpoint[f"critic_state_{en_index}"])
			self.L_critic_optimizer[en_index].load_state_dict(checkpoint[f"critic_optimizer_state_{en_index}"])
			self.L_critic_target[en_index] = copy.deepcopy(self.L_critic[en_index])

			self.L_actor[en_index].load_state_dict(checkpoint[f"actor_state_{en_index}"])
			self.L_actor_optimizer[en_index].load_state_dict(checkpoint[f"actor_optimizer_state_{en_index}"])
			self.L_actor_target[en_index] = copy.deepcopy(self.L_actor[en_index])


				
		
		


