# DR-SAC

This repository contains a Distributionally Robust Soft Actor-Critic (SAC) implementation.

## Setup

Install required dependencies:
```
conda env create -f requirement.yaml -n DRAC
conda activate DRAC
```

## Running the Code

### Basic Training

Train with default configuration (Pendulum environment):

```
python sac.py # SAC
```

### Changing Environments

Train on a different environment:

```
python sac.py env=lunarlander
```

Available environments:

* `pendulum` (default)
* `cartpole`
* `lunarlander`
* `humanoid`
* `halfcheetah`

### Using Robust Policy

Train with robust policy:

```
python sac.py robust=true
```

### Adding Noise

Add noise to the environment:

```
python sac.py noise=true std=0.1
```

### Changing Hyperparameters

Change any hyperparameter directly from command line:

```
python sac.py batch_size=512 a_lr=0.001 net_width=128
```

### Evaluation Mode

Run in evaluation mode:

```
python sac.py eval_model=true
```

## Offline Dataset Generation

### Hugging Face SAC Expert

To refresh Ant-v5 offline data with a stronger policy (e.g. [farama-minari/Ant-v5-SAC-expert](https://huggingface.co/farama-minari/Ant-v5-SAC-expert)), install the optional tools and run the helper in `read_data.py`:

```
pip install stable-baselines3 huggingface_sb3
python read_data.py --source hf \
    --env_id Ant-v5 \
    --hf_repo farama-minari/Ant-v5-SAC-expert \
    --save_path folder/dataset/ANTV5/hf_expert/dataset
```

`read_data.py` now exposes CLI switches so you can choose between Minari/D4RL (`--source d4rl`, default) and Hugging Face SB3 rollouts (`--source hf`). The script writes out the familiar `dataset/{s.npy,a.npy,...}` structure, which you can point to via `data_path=/absolute/path/to/dataset`.

### Behavior Cloning Warmup & Self-Rollouts

Offline SAC can now bootstrap from behavior cloning before TD backups by setting `bc_pretrain_steps>0` (Ant config enables 50k steps by default). You can also let the freshly cloned policy collect additional transitions prior to offline updates:

```
python train_sac.py env/sac=ant \
    bc_pretrain_steps=75000 \
    rollout_after_bc=true rollout_steps=300000 rollout_epsilon=0.05
```

The BC stage trains the actor with pure supervised loss, then (optionally) the agent interacts with the live Gym environment to augment the replay buffer before standard SAC training resumes.
