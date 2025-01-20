# LAROO: A Laplace-Based Robust Approach for Q-Value Estimation in Offline-to-Online Reinforcement Learning

## Code
Please note that we only provide codes including our algorithms and testing code, but not training code. If our paper could be accepted, we promise to provide the training code.
We provide testing code for validating our experimental performance. 

## Environment Installation
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

## Performance Validation
We provide testing models for validating our experimental performance. Checkpoints can be downloaded from the supplementary material. Put them in the folder './results' and run 'test.py'. [Recommended]

```
# Note download checkpoints and unzip into './results' folder
python -u test.py --env walker2d-medium-v2 --load_model 
```

### Other code
We provide our algorithms in 'TD3_BC_ensemble.py' and 'LARO.py' for Mujoco and Antmaze tasks respectively.









