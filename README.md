# [ICLR 2026] Tackling Heavy-Tailed Q-Value Bias in Offline-to-Online Reinforcement Learning with Laplace-Robust Modeling

\> ***\*Authors:\**** Ruibo Guo, Lei Liu*, Rui Yang, Junjie Shen, Guoping Wu, Jie Wang, Bin Li



## 1. Abstract

Offline-to-online reinforcement learning (O2O RL) aims to improve the performance of offline pretrained agents through online fine-tuning. Existing O2O RL methods have achieved advances in mitigating the overestimation of Q-value biases (i.e., biases of cumulative rewards), improving the performance. However, in this paper, we are the first to reveal that Q-value biases of these methods often follow a heavy-tailed distribution during online fine-tuning. Such biases induce high estimation variance and hinder performance improvement. To address this challenge, we propose a Laplace-based robust offline-to-online RL (LAROO) approach. LAROO introduces a parameterized Laplace-distributed noise and transfers the heavy-tailed nature of Q-value biases into this noise, alleviating heavy tailedness of biases for training stability and performance improvement. Specifically, (1) since Laplace distribution is well-suited for modeling heavy-tailed data, LAROO introduces a parameterized Laplace-distributed noise that can adaptively capture heavy tailedness of any data. (2) By combining estimated Q-values with the noise to approximate true Q-values, LAROO transfers the heavy-tailed nature of biases into the noise, reducing estimation variance. (3) LAROO employs conservative ensemble-based estimates to re-center Q-value biases, shifting their mean towards zero. Based on (2) and (3), LAROO promotes heavy-tailed Q-value biases into a standardized form, improving training stability and performance. Extensive experiments demonstrate that LAROO achieves significant performance improvement, outperforming several state-of-the-art O2O RL baselines.



## 2. Installation

```
# install MuJoCo for Linux
mkdir -p ~/.mujoco/mujoco210 
wget https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz -O mujoco210-macos-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/mujoco210 
pip install -U 'mujoco-py<2.2,>=2.1'

# install D4RL
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

# install LAROO
conda create --name LAROO python=3.7 -y
conda activate LAROO
pip install -e .

```



## 3. Getting Started

### Train

#### MuJoco-v2

You  should first pretrain agents and save the models in the directory './models', and then online fine-tune pretrained agents.

```
# offline pretraining
python -u offline_main.py --seed 1 --env_name walker2d-medium-v2 --save_model 
# online fine-tuning
python -u finetune_main.py --env walker2d-medium-v2 --load_model
```

#### Antmaze

```
#offline pretraining
python -u offline_lapo.py --seed 1 --env_name antmaze-umaze-v2 
#online fine-tuning
python -u finetune_lapo.py --seed 1 --env antmaze-umaze-v2 --model_dir yourdirs
```

#### 

## 4. Citation

```
@inproceedings{guotackling,
  title={Tackling Heavy-Tailed Q-Value Bias in Offline-to-Online Reinforcement Learning with Laplace-Robust Modeling},
  author={Guo, Ruibo and Yang, Rui and Liu, Lei and Shen, Junjie and Wu, Guoping and Wang, Jie and Li, Bin},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
