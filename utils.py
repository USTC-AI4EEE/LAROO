import numpy as np
import torch


def cumulative_discount(list_data):
    # 倒序list 相加,reward 在data数据中的第0位
    q_list = np.zeros(len(list_data))
    discount = 0.99
    q_list[-1] = list_data[-1]
    for i in range(len(list_data)-1):
        q_list[len(list_data) - i - 2] += q_list[len(list_data)-i-1] * discount + list_data[len(list_data)- i - 2]

    return q_list

def func_kurtosis(numpy_data):
    all_data = numpy_data.flatten()
    mean_q = numpy_data.mean()
    up,down = 0.0,0.0
    count = len(all_data)
    for item in all_data:
        up +=(item - mean_q)**4
        down+=(item - mean_q)**2

    kurtosis = (up/count)/((down/count)**2)
    return kurtosis

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, mask_prob, device, max_size=int(2e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		print('capacity: ', max_size, 'state capacity: ', self.state.shape)
		self.device = device

		self.mask_prob = mask_prob

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		# print(self.size, self.ptr)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		bootstrap_mask = np.array(np.random.binomial(1, self.mask_prob, len(dataset['observations'])),  dtype = bool)
		self.state = dataset['observations'][bootstrap_mask]
		self.action = dataset['actions'][bootstrap_mask]
		self.reward = dataset['rewards'][bootstrap_mask].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'][bootstrap_mask].reshape(-1,1)
		self.next_state = dataset['next_observations'][bootstrap_mask]
		
		self.size = self.state.shape[0]
		self.ptr = self.state.shape[0]
		print('convert buffer size: ', self.size)


	def initialize_with_dataset(self, dataset):
		dataset_size = len(dataset['observations'])
		num_samples = dataset_size
		indices = np.arange(num_samples)
		self.state[:num_samples] = dataset['observations'][indices]
		self.action[:num_samples] = dataset['actions'][indices]
		self.next_state[:num_samples] = dataset['next_observations'][indices]
		self.reward[:num_samples] = dataset['rewards'].reshape(-1,1)[indices]
		self.not_done[:num_samples] = 1. - dataset['terminals'].reshape(-1,1)[indices]
		self.size = num_samples
		self.ptr = num_samples
		print('convert buffer size: ', self.size, ' ptr: ', self.ptr)


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std
	
	def sample_all(self):
		return {
            "observations": self.state[:self.size].copy(),
            "actions": self.action[:self.size].copy(),
            "next_observations": self.next_state[:self.size].copy(),
            "terminals": 1.0 - self.not_done[:self.size].copy(),
            "rewards": self.reward[:self.size].copy()
        }



class Online_ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		print('capacity: ', max_size, 'state capacity: ', self.state.shape)
		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]
		self.ptr = self.state.shape[0]
		print('convert buffer size: ', self.size)


	def initialize_with_dataset(self, dataset):
		dataset_size = len(dataset['observations'])
		num_samples = dataset_size
		indices = np.arange(num_samples)
		self.state[:num_samples] = dataset['observations'][indices]
		self.action[:num_samples] = dataset['actions'][indices]
		self.next_state[:num_samples] = dataset['next_observations'][indices]
		self.reward[:num_samples] = dataset['rewards'].reshape(-1,1)[indices]
		self.not_done[:num_samples] = 1. - dataset['terminals'].reshape(-1,1)[indices]
		self.size = num_samples
		self.ptr = num_samples
		print('convert buffer size: ', self.size, ' ptr: ', self.ptr)


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std
	
	def sample_all(self):
		return {
            "observations": self.state[:self.size].copy(),
            "actions": self.action[:self.size].copy(),
            "next_observations": self.next_state[:self.size].copy(),
            "terminals": 1.0 - self.not_done[:self.size].copy(),
            "rewards": self.reward[:self.size].copy()
        }
	

def tukeys_biweight_loss(residuals, sigma=5.0):
    """
    计算 Tukey's Biweight 损失函数

    参数:
    y_true : Tensor
        实际值，形状为 [batch_size, 1]
    y_pred : Tensor
        预测值，形状为 [batch_size, 1]
    sigma : float
        调整参数，默认为 4.685

    返回:
    loss : Tensor
        Tukey's Biweight 损失
    """
    # residuals = y_true - y_pred
    sigma = torch.tensor(sigma).to(residuals.device) #注意，在一个torch.where函数运行时，两个条件应该是tensor且在同一device
    loss = torch.where(
            torch.abs(residuals) < sigma,
            (sigma**2 / 6) * (1 - (1 - (residuals / sigma)**2)**3),
            (sigma**2 / 6)
        )
    return loss

# # 示例使用
# y_true = torch.tensor([[3.0], [-0.5], [2.0], [7.0]])
# y_pred = torch.tensor([[2.5], [0.0], [2.0], [8.0]])

# loss = tukeys_biweight_loss(y_true, y_pred)
# print("Tukey's Biweight Loss:", loss.item())

def cauchy_loss(residuals,sigma=1.0):
    #residuals = y_true - y_pred
    loss = torch.log(1 + sigma * residuals**2)
    return loss

def huber_loss(diff, sigma = 1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss

def clip_loss(diff,lower = -10, upper = 20):
	clipped_diff = diff - diff.detach() + torch.clamp(diff, lower, upper).detach()
	loss = (clipped_diff.pow(2))
	return loss

def robust_loss(robust_func,diff,sigma=1.0):
	'''
	static robust regression
		diff: y_pred - y_true
		sigma : float,调整参数
	'''
	if robust_func == 'huber':
		loss = huber_loss(diff,sigma=sigma)
	elif robust_func == 'cauchy':
		loss = cauchy_loss(diff,sigma=sigma)
	elif robust_func == 'tukeys_biweight':
		loss = tukeys_biweight_loss(diff)
	elif robust_func == 'clip':
		loss = clip_loss(diff)
	else:
		loss = diff.pow(2)
	return loss


def get_qbias_online_data(dataset,device,policy,npy_path):
	true_value, all_estimate, estimated_q = [], [], []
	device = torch.device(device)
	# split_into_trajectories
	trajs = [[]]
	for i in range(len(dataset["observations"])):
		trajs[-1].append(dataset["rewards"][i])
		if dataset["terminals"][i] == 1.0 and i + 1 < len(dataset["observations"]):
			trajs.append([])

		# estimate Q
		torch_obs = torch.as_tensor(dataset["observations"][i].reshape(1,-1), device=device, dtype=torch.float32)
		torch_action = torch.as_tensor(dataset["actions"][i].reshape(1,-1), device=device, dtype=torch.float32)
		with torch.no_grad():
			q_esti = policy.estimation_q(torch_obs,torch_action).item()
		all_estimate.append(q_esti)
		estimated_q.append(q_esti)

	# calculate true Q for every trajectory
	for traj in trajs[:-1]:
		true_value.append(cumulative_discount(traj))

	true_value_flatten = np.concatenate([np.array(sublist) for sublist in true_value])
	pair_num = len(true_value_flatten)
	estimated_q_flatten = np.array(all_estimate[:pair_num])
	q_bias_flatten = estimated_q_flatten - true_value_flatten
	bias_kurtosis = func_kurtosis(q_bias_flatten)

	norm_qbias_std = np.std(q_bias_flatten/np.mean(true_value_flatten))
	if npy_path:
		np.save(npy_path,{
		"true_q": true_value_flatten,
		"q_bias": q_bias_flatten,
		"estimate_q": estimated_q_flatten,
		"bias_kurtosis":bias_kurtosis,
	})
	return {
		"true_q": np.mean(true_value_flatten),
		"estimate_q":  np.mean(estimated_q_flatten),
		"q_bias_mean": np.mean(q_bias_flatten),
		"q_bias_std" : np.std(q_bias_flatten),
		"norm_q_bias_mean" : np.mean(q_bias_flatten)/np.mean(true_value_flatten),
		"norm_q_bias_std" : norm_qbias_std,
		"bias_kurtosis": bias_kurtosis,
	}   
	# compute statistics of qbias
	# return {
	# 	"eval/true_q": np.mean(true_value_flatten),
	# 	"eval/estimate_q":  np.mean(estimated_q_flatten),
	# 	"eval/q_bias_mean": np.mean(q_bias_flatten),
	# 	"eval/q_bias_std" : np.std(q_bias_flatten),
	# 	"eval/norm_q_bias_mean" : np.mean(q_bias_flatten)/np.mean(true_value_flatten),
	# 	"eval/norm_q_bias_std" : norm_qbias_std,
	# 	"bias_kurtosis": bias_kurtosis,
	# }   


class ReplayBuffer_LAPO(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)

        self.storage = dict()
        self.storage['state'] = np.zeros((max_size, state_dim))
        self.storage['action'] = np.zeros((max_size, action_dim))
        self.storage['next_state'] = np.zeros((max_size, state_dim))
        self.storage['reward'] = np.zeros((max_size, 1))
        self.storage['not_done'] = np.zeros((max_size, 1))

        self.min_r, self.max_r = 0, 0

        self.action_mean = None
        self.action_std = None
        self.state_mean = None
        self.state_std = None

    def add(self, state, action, next_state, reward, done):
        self.storage['state'][self.ptr] = state.copy()
        self.storage['action'][self.ptr] = action.copy()
        self.storage['next_state'][self.ptr] = next_state.copy()

        self.storage['reward'][self.ptr] = reward
        self.storage['not_done'][self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.storage['state'][ind]).to(self.device),
            torch.FloatTensor(self.storage['action'][ind]).to(self.device),
            torch.FloatTensor(self.storage['next_state'][ind]).to(self.device),
            torch.FloatTensor(self.storage['reward'][ind]).to(self.device),
            torch.FloatTensor(self.storage['not_done'][ind]).to(self.device),
        )

    def save(self, filename):
        np.save("./buffers/" + filename + ".npy", self.storage)

    def normalize_state(self, state):
        return (state - self.state_mean)/(self.state_std+0.000001)

    def unnormalize_state(self, state):
        return state * (self.state_std+0.000001) + self.state_mean

    def normalize_action(self, action):
        return (action - self.action_mean)/(self.action_std+0.000001)

    def unnormalize_action(self, action):
        return action * (self.action_std+0.000001) + self.action_mean

    def renormalize(self):
        self.storage['state'] = self.unnormalize_state(self.storage['state'])
        self.storage['next_state'] = self.unnormalize_state(self.storage['next_state'])
        self.storage['action'] = self.unnormalize_action(self.storage['action'])

        self.action_mean = np.mean(self.storage['action'][:self.size], axis=0)
        self.action_std = np.std(self.storage['action'][:self.size], axis=0)
        self.state_mean = np.mean(self.storage['state'][:self.size], axis=0)
        self.state_std = np.std(self.storage['state'][:self.size], axis=0)        

        self.storage['state'] = self.normalize_state(self.storage['state'])
        self.storage['next_state'] = self.normalize_state(self.storage['next_state'])
        self.storage['action'] = self.normalize_action(self.storage['action'])

        self.min_r = self.storage['reward'].min()
        self.max_r = self.storage['reward'].max()

    def load(self, data):
        assert('next_observations' in data.keys())

        for i in range(data['observations'].shape[0]):
            self.add(data['observations'][i], data['actions'][i], data['next_observations'][i],
                     data['rewards'][i], data['terminals'][i])
                     
        self.action_mean = np.mean(self.storage['action'][:self.size], axis=0)
        self.action_std = np.std(self.storage['action'][:self.size], axis=0)
        self.state_mean = np.mean(self.storage['state'][:self.size], axis=0)
        self.state_std = np.std(self.storage['state'][:self.size], axis=0)

        self.storage['state'] = self.normalize_state(self.storage['state'])
        self.storage['next_state'] = self.normalize_state(self.storage['next_state'])
        self.storage['action'] = self.normalize_action(self.storage['action'])
