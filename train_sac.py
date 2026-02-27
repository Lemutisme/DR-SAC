from utils import evaluate_policy_SAC as evaluate_policy
from environment_modifiers import register
from continuous_cartpole import register
from sac import SAC_continuous

import hydra
import logging

import random
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="sac_config")
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
        "Hopper-v5",
        "Reacher-v5"
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
    opt.action_dim = env.action_space.shape[0]  # Continuous action dimensionprint
    opt.max_action = float(env.action_space.high[0])  # Action range [-max_action, max_action]
    opt.min_action = float(env.action_space.low[0])  # Action range [-max_action, max_action]
    opt.max_e_steps = env._max_episode_steps    

    # 4. Print environment info
    log.info(
        f"Env: {EnvName[opt.env_index]}  "
        f"state_dim: {opt.state_dim}  "
        f"action_dim: {opt.action_dim}  "
        f"max_a: {opt.max_action}  "
        f"min_a: {opt.min_action}  "
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
    model_dir = Path(f'models/SAC_model/{BrifEnvName[opt.env_index]}')
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Models will be saved to {model_dir}")

    # 8. Initialize the SAC agent
    agent = SAC_continuous(**OmegaConf.to_container(opt, resolve=True))

    # 9. Load a saved model if requested
    if opt.load_model:
        log.info("Loading pre-trained model")
        agent.load(BrifEnvName[opt.env_index], opt.load_path)

    # 10. If rendering mode is on, run an infinite evaluation loop
    if opt.render:
        while True:
            env =  gym.make(EnvName[opt.env_index], render_mode='human')
            ep_r = evaluate_policy(env, agent, turns=1)
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
        # type_lst = ['gaussian','laplace', 't', 'uniform', 'uniform']
        # scale_lst = [2.0, 1.5, 1.0, 0.5, 3.5]
        # type = 'gaussian'
        for i in tqdm(range(eval_num), desc="Evaluation Progress", ncols=100):
            # if i % (eval_num // 5) == 0:
            #     if i > 0:
            #          summary_logger.info(f"Mean score of last env: {np.mean(scores[i-eval_num//5:i]):.2f}")
            #     type = type_lst[i // (eval_num//5)]
            #     scale = scale_lst[i // (eval_num//5)]
            # eval_env = gym.make("CustomPendulum-v1", spread=opt.spread, type=type, adv=opt.adv) # Add noise when updating angle
            score = evaluate_policy(eval_env, agent, turns=1, seeds_list=[seeds_list[i]], random_action_prob=opt.random_action_prob)
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
        total_steps = 0
        total_episode = 0
        
        # Offline learning doesn't have exploration stage
        if opt.mode == 'offline':
            agent.replay_buffer.load(opt.data_path, opt.reward_adapt, opt.reward_normalize, opt.env_index)
            with tqdm(total=opt.max_train_steps, desc="Training Progress", ncols=100) as pbar:
                while total_steps < opt.vae_steps:
                    vae_loss = agent.tran_vae_train(opt.debug_print, writer, total_steps, iterations=opt.eval_interval)
                    total_steps += opt.eval_interval
                    pbar.update(opt.eval_interval)
                    
                    if total_steps % opt.eval_interval == 0:
                        if writer is not None:
                            writer.add_scalar('ep_r', ep_r, global_step=total_steps)

                        log.info(f"EnvName: {BrifEnvName[opt.env_index]}, "
                                 f"Steps: {int(total_steps/1000)}k, "
                                 f"VAE Loss: {vae_loss}")
                                            
                while total_steps < opt.max_train_steps:
                    agent.train(writer, total_steps)
                    total_steps += 1
                    pbar.update(1)
                    
                    # Learning rate decay
                    agent.a_lr *= 0.999
                    agent.c_lr *= 0.999
                    
                    # (d) Evaluate and log periodically
                    if total_steps % opt.eval_interval == 0:
                        # Temporarily close progress bars for evaluation
                        ep_r = evaluate_policy(eval_env, agent, turns=10, seeds_list=[random.randint(0, 100000) for _ in range(10)])

                        if writer is not None:
                            writer.add_scalar('ep_r', ep_r, global_step=total_steps)

                        log.info(f"EnvName: {BrifEnvName[opt.env_index]}, "
                                 f"Steps: {int(total_steps/1000)}k, "
                                 f"Episode Reward: {ep_r}")
                        
                    # (e) Save model at fixed intervals
                    if opt.save_model and total_steps % opt.save_interval == 0:
                        agent.save(BrifEnvName[opt.env_index])
                        
        elif opt.mode == 'generate':
            with tqdm(total=opt.max_train_steps, desc="Training Progress", ncols=100) as pbar:
                while total_steps < opt.max_train_steps:
                    # (a) Reset environment with incremented seed
                    state, info = env.reset(seed=env_seed)
                    env_seed += 1
                    total_episode += 1
                    done = False

                    # (b) Interact with environment until episode finishes
                    while not done:
                        # Random exploration for some episodes (each episode is up to max_e_steps)
                        if np.random.random() < opt.epsilon:
                            # Sample action directly from environment's action space
                            action = env.action_space.sample()  # Range: [-max_action, max_action]
                        else:
                            # Select action from agent
                            action = agent.select_action(state, deterministic=False)

                        # Step the environment
                        next_state, reward, dw, tr, info = env.step(action)

                        # Check for terminal state
                        done = (dw or tr)

                        # Store transition in replay buffer
                        agent.replay_buffer.add(state, action, reward, next_state, done)

                        # Move to next step
                        state = next_state
                        total_steps += 1

                        # Update progress bars
                        pbar.update(1)
                        
                    if total_episode % 1000 == 0:    
                        log.info(f"Data collected: {total_steps} in {total_episode} episodes.")
                        
            agent.replay_buffer.save()
                    
        elif opt.mode == 'continual':
            # Create a progress bar for the total training steps
            with tqdm(total=opt.max_train_steps, desc="Training Progress", ncols=100) as pbar:
                while total_steps < opt.max_train_steps:
                    # (a) Reset environment with incremented seed
                    state, info = env.reset(seed=env_seed)
                    env_seed += 1
                    total_episode += 1
                    done = False
                    ep_reward = 0

                    # Create a progress bar for steps within this episode
                    episode_pbar = tqdm(total=opt.max_e_steps, desc=f"Episode {total_episode}", 
                                        leave=False, ncols=100, position=1)

                    # (b) Interact with environment until episode finishes
                    episode_steps = 0
                    while not done:
                        # Random exploration for some episodes (each episode is up to max_e_steps)
                        if total_steps < (opt.explore_episode * opt.max_e_steps):
                            # Sample action directly from environment's action space
                            action = env.action_space.sample()  # Range: [-max_action, max_action]
                        else:
                            # Select action from agent 
                            action = agent.select_action(state, deterministic=False)

                        # Step the environment
                        next_state, reward, dw, tr, info = env.step(action)
                        ep_reward += reward

                        # Check for terminal state
                        done = (dw or tr)

                        # Store transition in replay buffer
                        agent.replay_buffer.add(state, action, reward, next_state, done)

                        # Move to next step
                        state = next_state
                        total_steps += 1
                        episode_steps += 1

                        # Update progress bars
                        pbar.update(1)
                        episode_pbar.update(1)

                        # Update progress bar description with more info
                        if total_steps % 10 == 0:
                            pbar.set_postfix({
                                'episode': total_episode,
                                'reward': f"{ep_reward:.2f}"
                            })

                        # (c) Train the agent at fixed intervals (batch updates)
                        if (total_steps >= opt.explore_episode * opt.max_e_steps) and (total_steps % opt.update_every == 0):
                            writer_copy = writer
                            train_bar = tqdm(range(opt.update_every), 
                                            desc="Model Update", 
                                            leave=False, ncols=100, position=2)

                            for i in train_bar:
                                agent.train(writer_copy, total_steps)
                                writer_copy = False

                            # Learning rate decay
                            agent.a_lr *= 0.999
                            agent.c_lr *= 0.999

                        # (d) Evaluate and log periodically
                        if total_steps % opt.eval_interval == 0:
                            # Temporarily close progress bars for evaluation
                            episode_pbar.close()
                            pbar.set_description("Evaluating...")
                            ep_r = evaluate_policy(eval_env, agent, turns=10)

                            if writer is not None:
                                writer.add_scalar('ep_r', ep_r, global_step=total_steps)

                            log.info(
                                f"EnvName: {BrifEnvName[opt.env_index]}, "
                                f"Steps: {int(total_steps/1000)}k, "
                                f"Episodes: {total_episode}, "
                                f"Episode Reward: {ep_r}"
                            )

                            # Reset progress bar description
                            pbar.set_description("Training Progress")
                            episode_pbar = tqdm(total=opt.max_e_steps, initial=episode_steps,
                                                desc=f"Episode {total_episode}", 
                                                leave=False, ncols=100, position=1)

                        # (e) Save model at fixed intervals
                        if opt.save_model and total_steps % opt.save_interval == 0:
                            agent.save(BrifEnvName[opt.env_index])

                    # Close episode progress bar when episode ends
                    episode_pbar.close()

                    # Log episode stats
                    log.info(f"Episode {total_episode} completed with reward {ep_reward:.2f} in {episode_steps} steps")

        # Evaluate the trained agent
        eval_num = 20
        log.info(f"Training completed. Evaluating across {eval_num} episodes")
        scores = []

        # Create a progress bar for evaluation
        for i in tqdm(range(eval_num), desc="Final Evaluation", ncols=100):
            score = evaluate_policy(eval_env, agent, turns=1)
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
        summary_logger.info(f"Total steps: {total_steps}")
        summary_logger.info(f"Total episodes: {total_episode}")
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

if __name__ == '__main__':
    main()
