# DR-SAC

This repository provides an implementation of Distributionally Robust Soft Actor-Critic (DR-SAC) (ICLR 2026).

Paper links

OpenReview: https://openreview.net/forum?id=a19MA0ksbc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)

arXiv: https://arxiv.org/abs/2506.12622

## Setup

Install required dependencies:

```
conda env create -f requirement.yaml -n DRSAC
conda activate DRSAC
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

### Multirun with Different Configurations

Run multiple configurations in parallel:

```
python sac.py --multirun env=pendulum,lunarlander robust=true,false
```

This will run 4 experiments with all combinations of environments and robust settings.

## Configuration Structure

### Main Configuration

The main configuration file (`config.yaml`) contains default settings for the algorithm.

### Environment Configurations

Environment-specific configurations are stored in the `config/env/` directory. They define:

* Environment name and index
* Recommended training steps

## Dataset and Selected Models

```

wget -O ./models/models.zip "https://uofi.box.com/s/9bfnbhexghgv6xbfmu4rj946ng9sv9oa"
echo "Unzipping models..."
unzip ./models/models.zip -d ./models/selected_models

wget -O ./data/datasets.zip "https://uofi.box.com/s/3qzfdtm5wx2lwckam9es0aptep58tc6d"
echo "Unzipping datasets..."
unzip ./datasets/datasets.zip -d ./datasets
```

## Output Structure

When running with Hydra:

* Logs, configs, and outputs are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`
* For multirun experiments, outputs are saved to `multirun/YYYY-MM-DD/HH-MM-SS/`
* TensorBoard logs are in the `tensorboard/` subdirectory
* Models are saved to `models/SAC_model/{ENV_NAME}/`

## Advanced Usage

### Adding New Environments

To add a new environment:

1. Create a YAML file in `config/env/`
2. Define environment parameters (name, index, training steps)
3. Update the environment lists in `sac.py` if needed

### Creating Custom Configuration Groups

You can create custom configuration groups for different experiments:

1. Create a directory in `config/` (e.g., `config/experiment/`)
2. Add YAML files with different configurations
3. Run with: `python sac_hydra.py +experiment=my_config`

### Citation

```
@inproceedings{
  cui2026drsac,
  title={{DR}-{SAC}: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty},
  author={Mingxuan Cui and Duo Zhou and Yuxuan Han and Grani A. Hanasusanto and Qiong Wang and Huan Zhang and Zhengyuan Zhou},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=a19MA0ksbc}
}
```
