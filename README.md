# DR-SAC

This repository contains a Distributionally Robust Soft Actor-Critic (SAC) implementation.

## Setup

Install required dependencies:

```
conda env create -f requirements.yaml -n DRAC
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
