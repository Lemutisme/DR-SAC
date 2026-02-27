import numpy as np
import gymnasium as gym
import random
import hydra
import logging

from d3rlpy import load_learnable
from d3rlpy.algos import TD3, TD3Config, SAC, SACConfig, DDPG, DDPGConfig, CQL, CQLConfig
from d3rlpy.preprocessing import ConstantShiftRewardScaler
from d3rlpy.logging import FileAdapterFactory
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import EnvironmentEvaluator
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

from utils import Reward_adapter
from ReplayBuffer import ReplayBuffer
from continuous_cartpole import register
from environment_modifiers import register

def eval_policy(env, agent, turns = 1, seeds_list = [], random_action_prob=0.0):
    total_scores = 0
    for j in range(turns):
        if len(seeds_list) > 0:
            s, _ = env.reset(seed=seeds_list[j])
        else:
            s, _ = env.reset()
        done = False
        while not done:
            s_batch = s[np.newaxis, :]
            if random.random() < random_action_prob:
                a = env.action_space.sample()
            else:
                # Take deterministic actions at test time
                a = agent.predict(s_batch)[0]
            s_next, r, dw, tr, _ = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)

def load_dataset(path, reward_adapt, EnvIdex):
    path = Path(get_original_cwd()) / path / "dataset"
        
    size = int(np.load(f"{path}/size.npy")[0])
    print(f"{size} data loaded.")
    s = np.load(f"{path}/s.npy")
    a = np.load(f"{path}/a.npy")
    r = np.load(f"{path}/r.npy")
    if reward_adapt:
        print(f"Before adaptation: Max: {r.max():.4f}, Min: {r.min():.4f}, Mean: {r.mean():.4f}.")
        for i in range(size):
            r[i] = Reward_adapter(r[i], EnvIdex)
        print(f"After adaptation: Max: {r.max():.4f}, Min: {r.min():.4f}, Mean: {r.mean():.4f}.")
    s_next = np.load(f"{path}/s_next.npy")
    dw =np.load(f"{path}/dw.npy")
    
    return s, a, r, s_next, dw


@hydra.main(version_base=None, config_path="config", config_name="d3rl_config")
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
    # file_handler = logging.FileHandler(output_dir / "train.log")
    # file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'))
    # log.addHandler(file_handler)

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
    summary_logger.info(f"Starting {cfg.model} training with configuration: {cfg.env_name}")

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
    model_dir = Path(f'models/d3rl_model/{BrifEnvName[opt.env_index]}')
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Models will be saved to {model_dir}")

    # 8. Initialize the SAC agent
    if opt.model == 'TD3':
        # if opt.mode == 'continual' and opt.env_index == 0:
        #     reward_scaler = ConstantShiftRewardScaler(shift=8, multiplier=0.125)
        # else:
        #     reward_scaler = 'default'
        config = TD3Config(batch_size=opt.batch_size,
                           gamma=opt.gamma,
                           tau=opt.tau,
                           actor_learning_rate=opt.a_lr,
                           critic_learning_rate=opt.c_lr)
        agent = TD3(config, opt.device, False)
    elif opt.model == 'SAC':
        config = SACConfig(batch_size=opt.batch_size,
                           gamma=opt.gamma,
                           tau=opt.tau,
                           actor_learning_rate=opt.a_lr,
                           critic_learning_rate=opt.c_lr,
                           n_critics=opt.n_critic)
        agent = SAC(config, opt.device, False)
    elif opt.model == 'DDPG':
        config = DDPGConfig(batch_size=opt.batch_size,
                           gamma=opt.gamma,
                           tau=opt.tau,
                           actor_learning_rate=opt.a_lr,
                           critic_learning_rate=opt.c_lr,
                           n_critics=opt.n_critic)
        agent = DDPG(config, opt.device, False)
    
    elif opt.model == 'CQL':
        config = CQLConfig(batch_size=opt.batch_size,
                           gamma=opt.gamma,
                           tau=opt.tau,
                           actor_learning_rate=opt.a_lr,
                           critic_learning_rate=opt.c_lr,
                           n_critics=opt.n_critic)
        agent = CQL(config, opt.device, False)
    else:
        raise NotImplementedError

    # 9. Load a saved model if requested
    if opt.load_model:
        log.info("Loading pre-trained model")
        agent = load_learnable(opt.load_path)

    # 10. If rendering mode is on, run an infinite evaluation loop
    if opt.render:
        while True:
            ep_r = eval_policy(env, agent, opt.max_action, turns=1)
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
            score = eval_policy(eval_env, agent, turns=1, seeds_list=[seeds_list[i]], random_action_prob=opt.random_action_prob)
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
        evaluator = EnvironmentEvaluator(eval_env, n_trials=20)
        
        # Offline learning doesn't have exploration stage
        if opt.mode == 'offline':
            s, a, r, s_next, dw = load_dataset(opt.data_path, opt.reward_adapt, opt.env_index)
            dataset = MDPDataset(s, a, r, dw)
            agent.build_with_dataset(dataset)
            agent.fit(dataset=dataset,
                      n_steps=opt.max_train_steps, 
                      logger_adapter=FileAdapterFactory(root_dir=output_dir),
                      save_interval=opt.save_interval,
                      evaluators={'reward':evaluator},
                      show_progress=True,
                     )
            agent.save("model.d3")
            
        elif opt.mode == 'generate':
            buffer = ReplayBuffer(opt.state_dim, opt.action_dim, max_size=int(1e6), device=opt.device)
            total_steps = 0
            total_episode = 0
            with tqdm(total=opt.max_train_steps, desc="Training Progress", ncols=100) as pbar:
                while total_steps < opt.max_train_steps:
                    # (a) Reset environment with incremented seed
                    state, _ = env.reset(seed=env_seed)
                    env_seed += 1
                    total_episode += 1
                    done = False

                    # (b) Interact with environment until episode finishes
                    while not done:
                        # Random exploration for some episodes (each episode is up to max_e_steps)
                        if np.random.random() < opt.epsilon:
                            # Sample action directly from environment's action space
                            action = env.action_space.sample()  
                        else:
                            # Select action from agent
                            state_batch = state[np.newaxis, :]
                            action = agent.predict(state_batch)[0]

                        # Step the environment
                        next_state, reward, dw, tr, _ = env.step(action)

                        # Check for terminal state
                        done = (dw or tr)

                        # Store transition in replay buffer
                        buffer.add(state, action, reward, next_state, done)

                        # Move to next step
                        state = next_state
                        total_steps += 1

                        # Update progress bars
                        pbar.update(1)
                        
                    if total_episode % 100 == 0:    
                        log.info(f"Data collected: {total_steps} in {total_episode} episodes.")
                        
            buffer.save()
            
        elif opt.mode == 'continual':
            agent.fit_online(env=env,
                             n_steps=opt.max_train_steps,
                             update_interval=opt.update_every,
                             n_updates=opt.update_every,
                             update_start_step=opt.learning_starts,
                             eval_env=env,
                             save_interval=opt.save_interval,
                             logger_adapter=FileAdapterFactory(root_dir=output_dir),
                             show_progress=True)
            agent.save("model.d3")
            
        
        else:
            raise NotImplementedError       
                              

        # Evaluate the trained agent
        eval_num = 20
        log.info(f"Training completed. Evaluating across {eval_num} episodes")
        scores = []

        # Create a progress bar for evaluation
        for i in tqdm(range(eval_num), desc="Final Evaluation", ncols=100):
            score = eval_policy(eval_env, agent, turns=1)
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
        # summary_logger.info(f"Total steps: {training_iters}")
        # summary_logger.info(f"Total episodes: {total_episode}")
        summary_logger.info(f"Final evaluation over {eval_num} episodes:")
        summary_logger.info(f"  Mean reward: {mean_score:.2f} ± {std_score:.2f}")
        summary_logger.info(f"  90th percentile: {p90_score:.2f}")
        summary_logger.info(f"  10th percentile: {p10_score:.2f}")
        summary_logger.info("-" * 50)


    env.close()
    eval_env.close()

    if writer is not None:
        writer.close()

    return agent

if __name__ == "__main__":
    main()



