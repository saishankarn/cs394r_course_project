# CS394r_course_project
## Context-driven control modulation
#### Sai Shankar Narasimhan and Sharachchandra Bhat

#### [Paper](https://drive.google.com/file/d/1PR1m6-9h_xudI8KWV1Pgo1yV5Co_1kT9/view?usp=sharing), [Video](https://youtu.be/kCg-QeSLrrU)

## Abstract

In this project, we aim to tackle the problem of cruise control in the presence of noisy local sensory-control information and delayed perfect sensory-control information from the cloud. We propose a reinforcement learning based solution that effectively combines the real-time but noisy local information and accurate but delayed cloud information to perform the cruise control task. Additionally, we have provided qualitative and quantitative results validat-
ing the effectiveness of our proposed solution. Link for the video presentation:https://youtu.be/kCg-QeSLrrU

## Installation

We recommend setting up a Python 3.8 Virtual Environment and installing all the dependencies listed in the requirements file. 

```
conda create -n env python=3.8
git clone git@github.com:saishankarn/cs394r_course_project.git

cd cs394r_course_project
pip install -r requirements.txt
```


## Repository Structure

```
cs394r_course_project/
├── gym_cruise_ctrl
|    └── gym_cruise_ctrl
|        └── envs
|            ├── cruise_ctrl_env_2d.py # Cruise control environment for 2D state features, environment name - 'cruise-ctrl-v0', id - 0
|            ├── cruise_ctrl_env_3d.py # Cruise control environment for 3D state features, environment name - 'cruise-ctrl-v1', id - 1
|            └── cruise_ctrl_env_8d.py # Cruise control environment for 8D state features, environment name - 'cruise-ctrl-v2', id - 2
|           
└── scripts
    └── frameworks
        ├── SAC # Soft Actor Critic implementation
        └── AC # Actor Critic and Soft Actor Critic ablation
```



# Experiments for the environment 2D state features
Train RL agent for the noiseless case. The hyper parameters can be provided as arguments. Refer to scripts/frameworks/SAC/main.py 
```
python scripts/frameworks/SAC/main.py --train --log_dir <path to the log directory> 
```
Train RL agent for the noisy case.
```
python scripts/frameworks/SAC/main.py --train --noise_required --log_dir <path to the log directory> 
```
Test RL agent for the noiseless case for 1000 episodes. 
```
python scripts/frameworks/SAC/main.py --test --log_dir <path to the log directory> --num_test_episodes 1000
```
Test RL agent for the noisy case for 1000 episodes. 
The policy parameters are loaded from the "own_sac_best_policy.pt" in the log_dir provided.
```
python scripts/frameworks/SAC/main.py --test --noise_required --log_dir <path to the log directory> --num_test_episodes 1000
```
Visualize the generated ego vehicle's acceleration profile for an episode. 
The policy parameters are loaded from the "own_sac_best_policy.pt" in the log_dir provided.
```
python scripts/frameworks/SAC/main.py --viz --log_dir <path to the log directory>
```
Run a training routine for 10 random seeds.
The training hyper parameters and the log directory can be edited in scripts/frameworks/SAC/run_multiple_seeds.sh 
To get the tensorboard summaries of different runs to generate the plots in Figure 3 run, 
```
sh scripts/frameworks/SAC/run_multiple_seeds.sh
```

# Experiments for the environment 3D state features
The commands to run the training and testing for 3D state features are similar to the commands for 2D state features
As an example, train RL agent for the noiseless case.
```
python scripts/frameworks/SAC/main.py --train --log_dir <path to the log directory> --env_id 1
```

# Experiments for the environment 8D state features
The commands to run the training and testing for 8D state features are similar to the commands for 2D and 3D state features
The 8D state features include noisy local sensory-control inputs, hence we always train the RL agent for the noisy case.
```
python scripts/frameworks/SAC/main.py --train --noise_required --log_dir <path to the log directory> --env_id 2
```

# Ablation studies for Soft Actor Critic framework
To run our Actor Critic implementation and the ablation experiments on Soft Actor Critic, run
```
python scripts/frameworks/AC/main.py
```
This runs all the ablation experiments in series.
