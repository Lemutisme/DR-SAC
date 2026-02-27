import argparse
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

import random
import hydra
import logging
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from RFQI.fqi import PQL_BCQ
from RFQI.rfqi import RFQI
from RFQI.data_container import DATA
from continuous_cartpole import register
    
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, eval_episodes=10, seeds_list=[], debug_print=False, random_action_prob=0):
    episode_reward = 0.0
    rewards = []
    for j in range(eval_episodes):
        if len(seeds_list) > 0:
            state, _ = env.reset(seed=seeds_list[j])
        else:
            state, _ = env.reset()
        done = False
        while not done:
            if random.random() < random_action_prob:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)
        episode_reward = 0.0
    reward_std = np.std(rewards)
    avg_reward = np.mean(rewards)  
    if debug_print:  
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
    return avg_reward, reward_std
        
@hydra.main(version_base=None, config_path="config", config_name="fqi_config")
def main(cfg: DictConfig):
    """
    Main function to train and evaluate an SAC agent on different environments.
    """
    # Set up logger
    log = logging.getLogger(__name__)
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create a summary log file for key information
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.log"

    # Configure file logging manually to ensure it works
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'))
    log.addHandler(file_handler)

    # Create file handler for summary log
    summary_handler = logging.FileHandler(summary_path)
    summary_handler.setLevel(logging.INFO)
    summary_formatter = logging.Formatter('[%(asctime)s] %(message)s')
    summary_handler.setFormatter(summary_formatter)

    # Create a separate logger for summary information
    summary_logger = logging.getLogger("summary")
    summary_logger.setLevel(logging.INFO)
    summary_logger.addHandler(summary_handler)
    summary_logger.propagate = False
    summary_logger.info(f"Starting SAC training with configuration: {cfg.env_name}")

    # Log system information
    import platform
    import torch.cuda
    system_info = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

    log.info(f"System information:")
    for key, value in system_info.items():
        log.info(f"  {key}: {value}")
    summary_logger.info(f"System: {system_info['Platform']}, PyTorch: {system_info['PyTorch']}, GPU: {system_info['GPU']}")

    # 1. Define environment names and abbreviations
    EnvName = [
        'Pendulum-v1',
        "ContinuousCartPole-v0",
        'LunarLanderContinuous-v3',
        'Humanoid-v5',
        'HalfCheetah-v5',
        'Hopper-v5',
        'Reacher-v5'
    ]
    BrifEnvName = [
        'PV1',
        "CPV0",
        'LLdV3',
        'HumanV5',
        'HCV5',
        'HPV5',
        'RV5'
    ]

    # Create a config object from Hydra for compatibility with rest of code
    opt = DictConfig({})
    for key, value in cfg.items():
        if key not in ['hydra']:  # Skip hydra config
            setattr(opt, key, value)

    # 2. Create training and evaluation environments
    # Import environment modifier if environment modifications are enabled
    if hasattr(cfg, 'env_mods') and cfg.env_mods.use_mods:
        # Import the environment_modifiers module
        from environment_modifiers import create_env_with_mods
        log.info("Using environment modifications from config")
        env, eval_env = create_env_with_mods(EnvName[opt.env_index], cfg.env_mods)
        
        # Log the modifications being applied
        # log.info(f"Applied modifications: {OmegaConf.to_yaml(cfg.env_mods)}") # kind of repeated
        summary_logger.info(f"Environment modifications enabled: {cfg.env_mods.use_mods}") 
    else:
        # Use legacy noise settings if env_mods is not used
        if not opt.noise:
            env = gym.make(EnvName[opt.env_index])
            eval_env = gym.make(EnvName[opt.env_index])
        else:
            if opt.env_index == 0:
                env = gym.make("CustomPendulum-v1", spread=opt.spread, type=opt.type, adv=opt.adv) # Add noise when updating angle
                eval_env = gym.make("CustomPendulum-v1", spread=opt.scale*opt.spread, type=opt.type, adv=opt.adv) # Add noise when updating angle

    # 3. Extract environment properties
    opt.state_dim = env.observation_space.shape[0]
    opt.max_state = None
    opt.action_dim = env.action_space.shape[0]  # Continuous action dimensionprint
    opt.max_action = float(env.action_space.high[0])  # Action range [-max_action, max_action]
    opt.min_action = float(env.action_space.low[0])
    opt.max_e_steps = env._max_episode_steps 


    # 4. Print environment info
    log.info(
        f"Env: {EnvName[opt.env_index]}  "
        f"state_dim: {opt.state_dim}  "
        f"action_dim: {opt.action_dim}  "
        f"max_a: {opt.max_action}  "
        f"min_a: {env.action_space.low[0]}  "
        f"max_e_steps: {opt.max_e_steps}"
    )

    # 5. Seed everything for reproducibility
    env_seed = opt.seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env.action_space.seed(opt.seed)
    log.info(f"Random Seed: {opt.seed}")

    # 6. Set up TensorBoard for logging (if requested)
    writer = None
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        writepath = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "tensorboard"
        writepath.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=writepath)
        log.info(f"TensorBoard logs will be saved to {writepath}")

    # 7. Create a directory for saving models
    model_dir = Path(f'models/FQI_model/{BrifEnvName[opt.env_index]}')
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Models will be saved to {model_dir}")

    # 8. Initialize the SAC agent
    if opt.robust:
        agent = RFQI(**OmegaConf.to_container(opt, resolve=True))
    else:     
        agent = PQL_BCQ(**OmegaConf.to_container(opt, resolve=True))

    # 9. Load a saved model if requested
    if opt.load_model:
        log.info("Loading pre-trained model")
        # params = f"{opt.std}_{opt.robust}"
        agent.load(BrifEnvName[opt.env_index], opt.load_path)

    # 10. If rendering mode is on, run an infinite evaluation loop
    if opt.render:
        while True:
            env = gym.make(EnvName[opt.env_index], render_mode='human')
            ep_r = eval_policy(agent, env, eval_episodes=1)
            log.info(f"Env: {EnvName[opt.env_index]}, Episode Reward: {ep_r}")

    # 11. If evaluating only, print result
    elif opt.eval_model:
        eval_num = 50
        log.info(f"Evaluating agent across {eval_num} episodes")
        if opt.random_action_prob != 0:
            log.info(f"Action perturbation: random action probability={opt.random_action_prob}.")
        seeds_list = [random.randint(0, 100000) for _ in range(eval_num)] if not hasattr(opt, 'seeds_list') else opt.seeds_list        

        scores = []
        # Use tqdm for evaluation progress
        for i in tqdm(range(eval_num), desc="Evaluation Progress", ncols=100):
            score, _ = eval_policy(agent, eval_env, eval_episodes=1, seeds_list=[seeds_list[i]], random_action_prob=opt.random_action_prob)
            scores.append(score)
            # Update progress bar with current mean score
            if i > 0 and i % 5 == 4:
                current_mean = np.mean(scores[:i])
                tqdm.write(f"Current mean score after {i+1} episodes: {current_mean:.2f}")
                # Log intermediate results to summary
                summary_logger.info(f"Intermediate evaluation ({i+1}/{eval_num}): Mean score = {current_mean:.2f}")

        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        p90_score = np.quantile(scores, 0.9)
        p10_score = np.quantile(scores, 0.1)

        # Save results to output directory
        results_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "results.txt"
        with open(results_path, 'a') as f:
            f.write(f"{[BrifEnvName[opt.env_index], mean_score, std_score, p90_score, p10_score]}\n")

        log.info(f"Results: {BrifEnvName[opt.env_index]}, Mean: {mean_score:.2f}, Std: {std_score:.2f}")
        log.info(f"90th percentile: {p90_score:.2f}, 10th percentile: {p10_score:.2f}")
        log.info(f"Results saved to {results_path}")

        # Log final results to summary file
        summary_logger.info("-" * 50)
        summary_logger.info("EVALUATION COMPLETED")
        summary_logger.info(f"Environment: {EnvName[opt.env_index]}")
        summary_logger.info(f"Evaluation over {eval_num} episodes:")
        summary_logger.info(f"  Mean reward: {mean_score:.2f} ± {std_score:.2f}")
        summary_logger.info(f"  90th percentile: {p90_score:.2f}")
        summary_logger.info(f"  10th percentile: {p10_score:.2f}")
        summary_logger.info("-" * 50)

    # 12. Otherwise, proceed with training
    else:
        training_iters = 0
        
        # Offline learning doesn't have exploration stage
        if opt.automatic_beta == 'True':
            automatic_beta = True
        else:
            automatic_beta = False# load data
        data = DATA(opt.state_dim, opt.action_dim, opt.max_action, opt.device, opt.data_size)
        data.load(opt.data_path, opt.reward_adapt, opt.reward_normalize, opt.env_index)

        # train VAE
        filter_scores = []
        training_iters = 0
        with tqdm(total=opt.max_trn_steps, desc="Training Progress", ncols=100) as pbar:
            if not opt.robust:
                while training_iters < opt.max_vae_trn_step:
                    vae_loss = agent.train_vae(data, iterations=int(opt.eval_freq), batch_size=opt.batch_size)  
                    training_iters += opt.eval_freq
                    pbar.update(opt.eval_freq)
                    log.info(f"Training iterations: {training_iters}. State VAE loss: {vae_loss:.3f}.")

                if opt.automatic_beta:  # args.automatic_beta:
                    test_loss = agent.test_vae(data, batch_size=100000)
                    beta = np.percentile(test_loss, opt.beta_percentile)
                    agent.beta = beta
                    log.info("Test vae", opt.beta_percentile,"percentile:", beta)
                else:
                    pass

            # train policy for 'eval_freq' steps
            while training_iters < opt.max_trn_steps:
                if opt.robust:
                    agent.train(data, trn_steps=int(opt.eval_freq), batch_size=opt.batch_size, writer=writer, log_base=training_iters)
                else:
                    agent.train(data, iterations=int(opt.eval_freq), batch_size=opt.batch_size)
                    
                training_iters += opt.eval_freq # loop
                pbar.update(opt.eval_freq)

                        
                if training_iters % opt.eval_interval == 0:
                    # Temporarily close progress bars for evaluation
                    ep_r, _ = eval_policy(agent, eval_env, eval_episodes=10)

                    if writer is not None:
                        writer.add_scalar('ep_r', ep_r, global_step=training_iters)

                    log.info(f"EnvName: {BrifEnvName[opt.env_index]}, "
                             f"Steps: {int(training_iters/1000)}k, "
                             f"Episode Reward: {ep_r:.3f}.")
                            
                # (e) Save model at fixed intervals
                if opt.save_model and training_iters % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.env_index])
                        

        # Evaluate the trained agent
        eval_num = 20
        log.info(f"Training completed. Evaluating across {eval_num} episodes")
        scores = []

        # Create a progress bar for evaluation
        for i in tqdm(range(eval_num), desc="Final Evaluation", ncols=100):
            score, _ = eval_policy(agent, eval_env, eval_episodes=1)
            scores.append(score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        p90_score = np.quantile(scores, 0.9)
        p10_score = np.quantile(scores, 0.1)

        log.info(f"Final evaluation - Mean: {mean_score:.2f}, Std: {std_score:.2f}")
        log.info(f"90th percentile: {p90_score:.2f}, 10th percentile: {p10_score:.2f}")

        # Log final results to summary file
        summary_logger.info("-" * 50)
        summary_logger.info("TRAINING COMPLETED")
        summary_logger.info(f"Environment: {EnvName[opt.env_index]}")
        summary_logger.info(f"Total steps: {training_iters}")
        # summary_logger.info(f"Total episodes: {total_episode}")
        summary_logger.info(f"Final evaluation over {eval_num} episodes:")
        summary_logger.info(f"  Mean reward: {mean_score:.2f} ± {std_score:.2f}")
        summary_logger.info(f"  90th percentile: {p90_score:.2f}")
        summary_logger.info(f"  10th percentile: {p10_score:.2f}")
        summary_logger.info("-" * 50)

        # Save final model
        if opt.save_model:
            agent.save(BrifEnvName[opt.env_index])
            log.info(f"Final model saved to models/SAC_model/{BrifEnvName[opt.env_index]}")

    env.close()
    eval_env.close()

    if writer is not None:
        writer.close()

    return agent

if __name__ == "__main__":
    main()