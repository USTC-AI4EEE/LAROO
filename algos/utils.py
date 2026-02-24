"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class ReplayBuffer(object):
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

def ori_loss(diff,sigma1=3.0,sigma2=1.0):
    loss = (sigma1 * torch.exp(-torch.abs(diff)/sigma1) + torch.abs(diff) - sigma1) / sigma2
    return loss

def ori_huber_loss(diff,sigma1=3.0,sigma2=1.0):
    diff = torch.abs(diff)
    cond = diff < sigma1
    loss = torch.where(cond, diff**2/(2 * sigma1 * sigma2), (diff - 0.5 * sigma1)/sigma2)
    return loss

def robust_loss(robust_func,diff,sigma=1.0,*kwargs):
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
    elif robust_func == 'ori':
        loss = ori_loss(diff,sigma1=sigma)
    elif robust_func == 'ori_huber':
        loss = ori_huber_loss(diff,sigma1=sigma)
    else:
        loss = (diff.pow(2))
    return loss

def func_kurtosis(numpy_data):
    all_data = numpy_data.flatten()
    mean_q = numpy_data.mean()
    up,down = 0.0,0.0
    count = len(all_data)
    if count == 0:
        raise ValueError("Count 为零，无法计算峰度。")
    for item in all_data:
        up +=(item - mean_q)**4
        down+=(item - mean_q)**2

    kurtosis = (up/count)/((down/count)**2)
    return kurtosis

def cumulative_discount(list_data):
    # 倒序list 相加,reward 在data数据中的第0位
    q_list = np.zeros(len(list_data))
    discount = 0.99
    q_list[-1] = list_data[-1][0]
    for i in range(len(list_data)-1):
        q_list[len(list_data) - i - 2] += q_list[len(list_data)-i-1] * discount + list_data[len(list_data)- i - 2][0]

    return q_list
