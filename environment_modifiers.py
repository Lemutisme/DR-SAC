"""
Environment modifiers for SAC training.
This module provides functions to create environments with various distribution shifts.
"""

import logging
import numpy as np
from numpy.random import normal
import scipy.stats as stats
import math
import gymnasium as gym
import mujoco

from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gymnasium.envs.box2d.lunar_lander import LunarLander
from continuous_cartpole import ContinuousCartPoleEnv
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

#----------------------------- ↓↓↓↓↓ Custom Env (Change Step Function) ↓↓↓↓↓ ------------------------------#
# Custom Pendulum Env (Add Noise in Step Function)
class CustomPendulum(PendulumEnv):
    def __init__(self, render_mode=None, spread=0.0, type="gaussian", adv=False):
        super().__init__(render_mode=render_mode)
        self.noise_spread = spread #
        self.type = type # noise distribution
        self.adv = adv # adverse noise
        
        if type == "gaussian":
            self.dist = stats.norm(loc=0, scale=spread)
        elif type == "cauchy":
            self.dist = stats.cauchy(loc=0, scale=spread)
        elif type == "laplace":
            self.dist = stats.laplace(loc=0, scale=spread)
        elif type == "t":
            self.dist = stats.t(df=2, loc=0, scale=spread)
        elif type == "uniform":
            self.dist = stats.uniform(loc=-0.5*spread, scale=spread)
        logger.info(f"Pendulum Env with {type} theta noise spread={spread}.")
        if adv:
            logger.info(f"Always adverse noise.")
        
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt 
        #############################################################	
        # Universal noise
        if not self.adv:
            newth += self.dist.rvs()
        
        # Always adverse noise 
        else:
            newth += abs(self.dist.rvs()) * np.sign(newth) 
        #############################################################	

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

register(
    id="CustomPendulum-v1",
    entry_point="environment_modifiers:CustomPendulum",
    max_episode_steps=200,
    kwargs={'spread': 0.0, 'type':'gaussian', 'adv':False}
)

#----------------------------- ↓↓↓↓↓ Parameter Shifted Env ↓↓↓↓↓ ------------------------------#
# Customized Pendulum
class ParameterShiftedPendulum(PendulumEnv):
    def __init__(self, render_mode=None, gravity_factor=1.0, mass_factor=1.0, length_factor=1.0):
        """
        Pendulum environment with shifted physical parameters.
        
        Args:
            gravity_factor: Factor to multiply gravity by
            mass_factor: Factor to multiply mass by
            length_factor: Factor to multiply length by
        """
        super().__init__(render_mode=render_mode)
        # Modify the physical parameters
        self.g = self.g * gravity_factor  # Gravity
        self.m = self.m * mass_factor  # Mass
        self.l = self.l * length_factor  # Length
        
        logger.info(f"Parameter Shifted Pendulum: g={self.g:.2f}, m={self.m:.2f}, l={self.l:.2f}")

# Register the environments
register(
    id="ParameterShiftedPendulum-v1",
    entry_point="environment_modifiers:ParameterShiftedPendulum",
    max_episode_steps=200,
    kwargs={'gravity_factor':1.0, 'mass_factor':1.0, 'length_factor':1.0}
)

# Customized LunarLander
class ParameterShiftedLunarLander(LunarLander):
    def __init__(self, render_mode=None, gravity_factor=1.0, wind_power=0.0, turbulence_power=1.5, engine_factor=1.0):
        """
        LunarLander environment with shifted parameters.

        Args:
            gravity_factor: Factor to multiply gravity by
            wind_power: Constant wind force applied
            turbulence_power: Random turbulence amplitude
        """
        super().__init__(render_mode=render_mode, continuous=True,
                         gravity=-10.0*gravity_factor,
                         enable_wind=(wind_power!=0), wind_power=wind_power,
                         turbulence_power=turbulence_power)
        # self.gravity_factor = gravity_factor
        # self.wind_power = wind_power
        # self.turbulence_power = turbulence_power
        self.engine_factor = engine_factor
        logger.info(f"Parameter Shifted LunarLander: gravity_factor={gravity_factor:.2f}, "
                    f"wind_power={wind_power:.2f},"
                    f"turbulence_power={turbulence_power:.2f},"
                    f"engine_factor={engine_factor:.2f}")
    
    def step(self, action):
        # Constant
        FPS = 50
        SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

        MAIN_ENGINE_POWER = 13.0 * self.engine_factor
        SIDE_ENGINE_POWER = 0.6 * self.engine_factor

        INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

        LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
        LEG_AWAY = 20
        LEG_DOWN = 18
        LEG_W, LEG_H = 2, 8
        LEG_SPRING_TORQUE = 40

        SIDE_ENGINE_HEIGHT = 14
        SIDE_ENGINE_AWAY = 12
        MAIN_ENGINE_Y_LOCATION = (
            4  # The Y location of the main engine on the body of the Lander.
        )

        VIEWPORT_W = 600
        VIEWPORT_H = 400
        
        assert self.lander is not None

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(
                torque_mag,
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

register(
    id="ParameterShiftedLunarLander-v3",
    entry_point="environment_modifiers:ParameterShiftedLunarLander",
    max_episode_steps=1000,
    kwargs={'gravity_factor':1.0, 'wind_power':0.0, 'turbulence_power':1.5, "engine_factor": 1.0}
)

# Customized Continuous Cartpole
class ParameterShiftedCartpole(ContinuousCartPoleEnv):
    def __init__(self, gravity_factor=1.0, length_factor=1.0, mass_factor=1.0, force_mag_factor=1.0):
        """
        Cartpole environment with shifted parameters.

        Args:
            gravity_factor: Factor to multiply gravity by
            length_factor: Factor to multiply length by
            force_mag_factor: Factor to multiply force magnitude by
        """
        super().__init__()
        self.gravity *= gravity_factor           # 9.8
        self.length *= length_factor             # 0.5
        self.masscart *= mass_factor
        self.masspole *= mass_factor
        self.total_mass = (self.masspole + self.masscart)
        self.force_mag *= force_mag_factor       # 30
        logger.info(f"Parameter Shifted Cartpole: gravity_factor={gravity_factor:.2f}, length factor={length_factor:.2f}, "
                    f"mass factor={mass_factor:.2f}, force_mag_factor={force_mag_factor:.2f}")
        
register(
    id="ParameterShiftedCartpole-v0",
    entry_point="environment_modifiers:ParameterShiftedCartpole",
    max_episode_steps=500,
    kwargs={'gravity_factor':1.0, 'length_factor':1.0, 'mass_factor':1.0, 'force_mag_factor':1.0}
)

# Customized Halfcheetah
class ParameterShiftedHalfCheetah(HalfCheetahEnv):
    def __init__(self, back_stiff=1.0, front_stiff=1.0,
                       back_damping=1.0, front_damping=1.0):
        super().__init__()
        for name in ['fthigh', 'fshin', 'ffoot']:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.model.jnt_stiffness[joint_id] *= front_stiff
            self.model.dof_damping[joint_id] *= front_damping
        
        for name in ['bthigh', 'bshin', 'bfoot']:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.model.jnt_stiffness[joint_id] *= back_stiff
            self.model.dof_damping[joint_id] *= back_damping
        
        logger.info("Parameter Shifted HalfCheetah: "
                    f"joint stiffness={self.model.jnt_stiffness[:]}, "
                    f"joint damping={self.model.dof_damping[:]}.")
        
register(
    id="ParameterShiftedHalfCheetah-v5",
    entry_point="environment_modifiers:ParameterShiftedHalfCheetah",
    max_episode_steps=1000,
    kwargs={'back_stiff':1.0, 'front_stiff':1.0, 'back_damping':1.0, 'front_damping':1.0}
)

# Customized Reacher
class ParameterShiftedReacher(ReacherEnv):
    def __init__(self, joint0_stiff=0.0, joint1_stiff=0.0,
                       joint0_damping=1.0, joint1_damping=1.0,
                       act_ctrlrange=1.0):
        """
        Cartpole environment with shifted parameters.

        Args:
           
        """
        # Now modify attributes
        super().__init__()
        
        joint0_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'joint0')
        joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'joint1')
        
        self.model.jnt_stiffness[joint0_id] = joint0_stiff
        self.model.jnt_stiffness[joint1_id] = joint1_stiff
        self.model.dof_damping[joint0_id] *= joint0_damping
        self.model.dof_damping[joint1_id] *= joint1_damping
        self.model.actuator_ctrlrange *= act_ctrlrange
        
        logger.info(f"Parameter Shifted Reacher: "
                    f"joint stiffness={self.model.jnt_stiffness}, "
                    f"joint damping={self.model.dof_damping}, "
                    f"acuator control range={self.model.actuator_ctrlrange}")
        
register(
    id="ParameterShiftedReacher-v5",
    entry_point="environment_modifiers:ParameterShiftedReacher",
    max_episode_steps=50,
    kwargs={'joint0_stiff':0.0, 'joint1_stiff':0.0, 'joint0_damping':1.0, 'joint1_damping':1.0, 'act_ctrlrange':1.0}
)

#----------------------------- ↓↓↓↓↓ Env Modification Wrapper ↓↓↓↓↓ ------------------------------#
# Observation Noise Wrapper
class ObservationNoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_type='gaussian', noise_level=0.1, noise_freq=1.0, noise_dim=[]):
        """
        Add noise to observations (all dimensions).

        Args:
            env: The environment to wrap
            noise_type: Type of noise ('gaussian', 'uniform', 'saltpepper', 'adversarial')
            noise_level: Standard deviation/magnitude of the noise
            noise_freq: Frequency of applying noise (1.0 = every step)
        """
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.noise_freq = noise_freq
        if not noise_dim:
            self.noise_dim = range(env.observation_space.shape[0])
        else:
            self.noise_dim = noise_dim
        self.observation_space = env.observation_space
        logger.info(f"Observation {noise_type} noise with level {noise_level} in dim {self.noise_dim}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_noise(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Apply noise based on frequency
        if np.random.random() < self.noise_freq:
            obs = self._add_noise(obs)
        return obs, reward, terminated, truncated, info

    def _add_noise(self, obs):
        if self.noise_type == 'gaussian':
            noise = np.zeros_like(obs)
            for i in self.noise_dim:
                noise[i] = np.random.normal(0, self.noise_level)
            return obs + noise
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_level, self.noise_level, size=obs.shape)
            return obs + noise
        elif self.noise_type == 'saltpepper':
            # Salt and pepper noise
            mask = np.random.random(size=obs.shape) < self.noise_level / 2
            obs_copy = obs.copy()
            obs_copy[mask] = np.max(obs)
            mask = np.random.random(size=obs.shape) < self.noise_level / 2
            obs_copy[mask] = np.min(obs)
            return obs_copy
        elif self.noise_type == 'adversarial':
            # Simple adversarial noise - pushes values away from center
            direction = np.sign(obs)
            noise = np.abs(np.random.normal(0, self.noise_level, size=obs.shape)) * direction
            return obs + noise
        else:
            return obs

# Action Perturbation Wrapper
class ActionPerturbationWrapper(gym.Wrapper):
    def __init__(self, env, perturb_type='drop', perturb_prob=0.1, perturb_level=0.1):
        """
        Perturb actions before they are executed.

        Args:
            env: The environment to wrap
            perturb_type: Type of perturbation ('drop', 'noise', 'delay', 'stuck')
            perturb_prob: Probability of applying the perturbation
            perturb_level: Magnitude of the perturbation
        """
        super().__init__(env)
        self.perturb_type = perturb_type
        self.perturb_prob = perturb_prob
        self.perturb_level = perturb_level
        self.last_action = None
        self.stuck_action = None
        self.stuck_count = 0
        self.delay_buffer = []
        self._max_episode_steps = env._max_episode_steps
        logger.info(f"Action perturbation: {perturb_type} with prob {perturb_prob}, level {perturb_level}")

    def reset(self, **kwargs):
        self.last_action = None
        self.stuck_action = None
        self.stuck_count = 0
        self.delay_buffer = []
        return self.env.reset(**kwargs)

    def step(self, action):
        # Process action if perturbation occurs
        if np.random.random() < self.perturb_prob:
            if self.perturb_type == 'drop':
                # Drop action (use last action or zero)
                action = self.last_action if self.last_action is not None else np.zeros_like(action)

            elif self.perturb_type == 'noise':
                # Add random noise to action
                action = action + np.random.normal(loc=0, scale=self.perturb_level, size=action.shape).astype(np.float32)
                action = np.clip(action, self.action_space.low, self.action_space.high)

            elif self.perturb_type == 'delay':
                # Delay action - apply an old action
                self.delay_buffer.append(action.copy())
                if len(self.delay_buffer) > int(1/self.perturb_level):
                    action = self.delay_buffer.pop(0)
                elif self.last_action is not None:
                    action = self.last_action
                else:
                    action = np.zeros_like(action)

            elif self.perturb_type == 'stuck':
                # Action gets stuck for several timesteps
                if self.stuck_count > 0:
                    action = self.stuck_action
                    self.stuck_count -= 1
                else:
                    self.stuck_action = action.copy()
                    self.stuck_count = int(np.random.exponential(scale=1/self.perturb_level))

        self.last_action = action.copy()
        return self.env.step(action)

# Reward Shift Wrapper
class RewardShiftWrapper(gym.Wrapper):
    def __init__(self, env, shift_type='scale', shift_param=0.9, noise_level=0.1):
        """
        Modify rewards to create distribution shifts.

        Args:
            env: The environment to wrap
            shift_type: Type of shift ('scale', 'delay', 'noise', 'sparse', 'sign_flip')
            shift_param: Parameter for the specific shift type
            noise_level: Standard deviation of noise (for 'noise' type)
        """
        super().__init__(env)
        self.shift_type = shift_type
        self.shift_param = shift_param
        self.noise_level = noise_level
        self.reward_buffer = []
        self._max_episode_steps = env._max_episode_steps
        logger.info(f"Reward shift: {shift_type} with param {shift_param}")

    def reset(self, **kwargs):
        self.reward_buffer = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply reward shift
        if self.shift_type == 'scale':
            # Scale rewards by a factor
            modified_reward = reward * self.shift_param

        elif self.shift_type == 'delay':
            # Delayed reward signal
            self.reward_buffer.append(reward)
            if len(self.reward_buffer) > int(self.shift_param):
                modified_reward = self.reward_buffer.pop(0)
            else:
                modified_reward = 0

        elif self.shift_type == 'noise':
            # Add noise to rewards
            modified_reward = reward + np.random.normal(0, self.noise_level)

        elif self.shift_type == 'sparse':
            # Make rewards more sparse by zeroing out small rewards
            if abs(reward) < self.shift_param:
                modified_reward = 0
            else:
                modified_reward = reward

        elif self.shift_type == 'sign_flip':
            # Randomly flip the sign of the reward with some probability
            if np.random.random() < self.shift_param:
                modified_reward = -reward
            else:
                modified_reward = reward

        else:
            modified_reward = reward

        return obs, modified_reward, terminated, truncated, info

# Transition Perturbation Wrapper
class TransitionPerturbationWrapper(gym.Wrapper):
    """
    This wrapper perturbs the transition dynamics by applying forces,
    teleporting the agent, or other dynamics modifications.
    """
    def __init__(self, env, perturb_type='teleport', perturb_prob=0.05, perturb_level=0.1):
        """
        Args:
            env: The environment to wrap
            perturb_type: Type of perturbation ('teleport', 'random_force', 'friction')
            perturb_prob: Probability of applying the perturbation
            perturb_level: Magnitude of the perturbation
        """
        super().__init__(env)
        self.perturb_type = perturb_type
        self.perturb_prob = perturb_prob
        self.perturb_level = perturb_level
        self.state_dim = env.observation_space.shape[0]
        logger.info(f"Transition perturbation: {perturb_type} with prob {perturb_prob}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Only apply perturbation with some probability
        if np.random.random() < self.perturb_prob:
            if self.perturb_type == 'teleport':
                # Teleport the agent by modifying the observation
                # Note: This doesn't actually change internal state, just what agent observes
                perturbation = np.random.normal(0, self.perturb_level, size=obs.shape)
                obs = obs + perturbation
                obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

            elif self.perturb_type == 'random_force' and hasattr(self.env.unwrapped, 'apply_force'):
                # Apply a random force if the environment supports it
                force = np.random.normal(0, self.perturb_level, size=2)
                self.env.unwrapped.apply_force(force)

            elif self.perturb_type == 'friction' and hasattr(self.env.unwrapped, 'world'):
                # Temporarily change friction (for Box2D environments)
                if hasattr(self.env.unwrapped, 'lander'):
                    for fixture in self.env.unwrapped.lander.fixtures:
                        fixture.friction = fixture.friction * (1 + np.random.normal(0, self.perturb_level))

        return obs, reward, terminated, truncated, info


def create_env_with_mods(env_name, env_config):
    """
    Create environment with modifications based on configuration.
    
    Args:
        env_name (str): Name of the environment to create
        env_config (DictConfig): Configuration for environment modifications

    Returns:
        tuple: (train_env, eval_env) - Training and evaluation environments
    """
    logger.info(f"Creating environment: {env_name}")
    
    # Create base environments
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)
                    
    # If no modifications, return base environments
    if not env_config.use_mods:
        logger.info("No environment modifications applied")
        return train_env, eval_env

    #----------------------------- ↓↓↓↓↓ Self-defined Env ↓↓↓↓↓ ------------------------------#
    # Custom environments
    if env_config.use_mods and env_config.custom.enabled:
         if env_name == "Pendulum-v1": 
             logger.info("Customizing training environment")
             train_env = gym.make("CustomPendulum-v1",
                                  spread=env_config.custom.spread, 
                                  type=env_config.custom.type, 
                                  adv=env_config.custom.adv)
             
             if env_config.eval.use_modified:
                 logger.info("Customizing evaluation environment")
                 if env_config.eval.scale_noise:
                     env_config.custom.spread *= env_config.eval.noise_scale_factor
                 eval_env = gym.make("CustomPendulum-v1",
                                    spread=env_config.custom.spread, 
                                    type=env_config.custom.type, 
                                    adv=env_config.custom.adv)
             else:
                 eval_env = gym.make(env_name)
                      

    # Param_shift environments
    if env_config.use_mods and env_config.param_shift.enabled:
        assert env_config.param_shift.train_apply or env_config.param_shift.eval_apply
        if env_name == "Pendulum-v1": 
            if env_config.param_shift.train_apply:
                logger.info("Applying parameter shift to training environment")
                train_env = gym.make("ParameterShiftedPendulum-v1",
                                    gravity_factor=env_config.param_shift.gravity_factor,
                                    mass_factor=env_config.param_shift.mass_factor,
                                    length_factor=env_config.param_shift.length_factor)
            else:
                train_env = gym.make(env_name)
                
            if env_config.param_shift.eval_apply:
                logger.info("Applying parameter shift to evaluation environment")
                eval_env = gym.make("ParameterShiftedPendulum-v1",
                                    gravity_factor=env_config.param_shift.gravity_factor,
                                    mass_factor=env_config.param_shift.mass_factor,
                                    length_factor=env_config.param_shift.length_factor)
            else:
                eval_env = gym.make(env_name)
                
        elif env_name == "LunarLanderContinuous-v3":
            train_env = gym.make("ParameterShiftedLunarLander-v3",
                                 gravity_factor=env_config.param_shift.gravity_factor, 
                                 wind_power=env_config.param_shift.wind_power, 
                                 turbulence_power=env_config.param_shift.turbulence_power,
                                 engine_factor=env_config.param_shift.engine_factor)
            if env_config.eval.use_modified:
                logger.info("Using modified environment for evaluation")
                eval_env = gym.make("ParameterShiftedLunarLander-v3",
                                    gravity_factor=env_config.param_shift.gravity_factor, 
                                    wind_power=env_config.param_shift.wind_power, 
                                    turbulence_power=env_config.param_shift.turbulence_power, 
                                    engine_factor=env_config.param_shift.engine_factor)
            else:
                eval_env = gym.make(env_name)
                
        elif env_name == "ContinuousCartPole-v0":
            train_env = gym.make("ParameterShiftedCartpole-v0",
                                 gravity_factor=env_config.param_shift.gravity_factor, 
                                 length_factor=env_config.param_shift.lenngth_factor,
                                 mass_factor=env_config.param_shift.mass_factor,
                                 force_mag_factor=env_config.param_shift.force_mag_factor)
            if env_config.eval.use_modified:
                logger.info("Using modified environment for evaluation")
                eval_env = gym.make("ParameterShiftedCartpole-v0",
                                 gravity_factor=env_config.param_shift.gravity_factor, 
                                 length_factor=env_config.param_shift.length_factor,
                                 mass_factor=env_config.param_shift.mass_factor, 
                                 force_mag_factor=env_config.param_shift.force_mag_factor)
            else:
                eval_env = gym.make(env_name)
                
        elif env_name == "HalfCheetah-v5":
            train_env = gym.make("ParameterShiftedHalfCheetah-v5",
                                 back_stiff=env_config.param_shift.back_stiff,
                                 front_stiff=env_config.param_shift.front_stiff,
                                 back_damping=env_config.param_shift.back_damping,
                                 front_damping=env_config.param_shift.front_damping)
            if env_config.eval.use_modified:
                logger.info("Using modified environment for evaluation")
                eval_env = gym.make("ParameterShiftedHalfCheetah-v5",
                                     back_stiff=env_config.param_shift.back_stiff,
                                     front_stiff=env_config.param_shift.front_stiff,
                                     back_damping=env_config.param_shift.back_damping,
                                     front_damping=env_config.param_shift.front_damping)
            else:
                eval_env = gym.make(env_name)
                
        elif env_name == "Reacher-v5":
            train_env = gym.make("ParameterShiftedReacher-v5",
                                 joint0_stiff=env_config.param_shift.joint0_stiff,
                                 joint1_stiff=env_config.param_shift.joint1_stiff,
                                 joint0_damping=env_config.param_shift.joint0_damping,
                                 joint1_damping=env_config.param_shift.joint1_damping,
                                 act_ctrlrange=env_config.param_shift.act_ctrlrange)
            if env_config.eval.use_modified:
                logger.info("Using modified environment for evaluation")
                eval_env = gym.make("ParameterShiftedReacher-v5",
                                     joint0_stiff=env_config.param_shift.joint0_stiff,
                                     joint1_stiff=env_config.param_shift.joint1_stiff,
                                     joint0_damping=env_config.param_shift.joint0_damping,
                                     joint1_damping=env_config.param_shift.joint1_damping,
                                     act_ctrlrange=env_config.param_shift.act_ctrlrange)
            else:
                eval_env = gym.make(env_name)

    #----------------------------- ↓↓↓↓↓ Add General Wrapper ↓↓↓↓↓ ------------------------------#
    # Apply modifications to training environment
    logger.info("Applying modifications to training environment")

    # Apply observation noise if configured
    if env_config.observation_noise.enabled:
        noise_dim = []
        if env_name == 'LunarLanderContinuous-v3':
            noise_dim = range(6)
        logger.info(f"Adding observation noise: {env_config.observation_noise.type} with level {env_config.observation_noise.level}")
        train_env = ObservationNoiseWrapper(
            train_env,
            noise_type=env_config.observation_noise.type,
            noise_level=env_config.observation_noise.level,
            noise_freq=env_config.observation_noise.frequency,
            noise_dim=noise_dim
        )

    # Apply action perturbation if configured
    if env_config.action_perturb.enabled:
        logger.info(f"Adding action perturbation: {env_config.action_perturb.type}")
        train_env = ActionPerturbationWrapper(
            train_env,
            perturb_type=env_config.action_perturb.type,
            perturb_prob=env_config.action_perturb.probability,
            perturb_level=env_config.action_perturb.level
        )

    # Apply reward shift if configured
    if env_config.reward_shift.enabled:
        logger.info(f"Adding reward shift: {env_config.reward_shift.type}")
        train_env = RewardShiftWrapper(
            train_env,
            shift_type=env_config.reward_shift.type,
            shift_param=env_config.reward_shift.param,
            noise_level=env_config.reward_shift.noise_level
        )

    # Apply transition perturbation if configured
    if env_config.transition_perturb.enabled:
        logger.info(f"Adding transition perturbation: {env_config.transition_perturb.type}")
        train_env = TransitionPerturbationWrapper(
            train_env,
            perturb_type=env_config.transition_perturb.type,
            perturb_prob=env_config.transition_perturb.probability,
            perturb_level=env_config.transition_perturb.level
        )

    # Create evaluation environment with potentially different modifications
    logger.info("Applying modifications to evaluation environment")

    # Decide evaluation environment configuration based on settings
    if env_config.eval.use_modified:
        logger.info("Using modified environment for evaluation")

        # Create evaluation environment with same modifications but potentially different parameters
        if env_config.observation_noise.enabled:
            eval_noise_level = env_config.observation_noise.level
            if env_config.eval.scale_noise:
                eval_noise_level *= env_config.eval.noise_scale_factor

            eval_env = ObservationNoiseWrapper(
                eval_env,
                noise_type=env_config.observation_noise.type,
                noise_level=eval_noise_level,
                noise_freq=env_config.observation_noise.frequency,
                noise_dim=noise_dim
            )
            logger.info(f"Evaluation environment using observation noise level: {eval_noise_level}")

        # Apply other modifications to eval env if needed...
        if env_config.action_perturb.enabled and env_config.eval.include_action_perturb:
            eval_env = ActionPerturbationWrapper(
                eval_env,
                perturb_type=env_config.action_perturb.type,
                perturb_prob=env_config.action_perturb.probability,
                perturb_level=env_config.action_perturb.level
            )
            logger.info(f"Evaluation environment using action noise level: {env_config.action_perturb.level}")

        if env_config.reward_shift.enabled and env_config.eval.include_reward_shift:
            eval_env = RewardShiftWrapper(
                eval_env,
                shift_type=env_config.reward_shift.type,
                shift_param=env_config.reward_shift.param,
                noise_level=env_config.reward_shift.noise_level
            )

        if env_config.transition_perturb.enabled and env_config.eval.include_transition_perturb:
            eval_env = TransitionPerturbationWrapper(
                eval_env,
                perturb_type=env_config.transition_perturb.type,
                perturb_prob=env_config.transition_perturb.probability,
                perturb_level=env_config.transition_perturb.level
            )
    else:
        logger.info("Using standard environment for evaluation")

    return train_env, eval_env
