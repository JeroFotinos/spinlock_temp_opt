import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class TemperatureControlEnv(gym.Env):
    """
    Gymnasium environment for controlling the temperature of a device.

    Parameters
    ----------
    n : int, optional
        Number of resistances (actions). Default is 3.
    m : int, optional
        Number of sensors (observations). Default is 3.
    target_temp : float, optional
        Target temperature in degrees Celsius. Default is 27.0.
    temp_lower_bound : float, optional
        The minimum temperature (should be the room temperature). Default is 25.0.
    max_current : float, optional
        Maximum current that can be applied to each resistance. Default is 5.0.
    """

    def __init__(
        self,
        n: int = 3,
        m: int = 3,
        target_temp: float = 27.0,
        temp_lower_bound: float = 25.0,
        max_current: float = 5.0,
    ):
        super(TemperatureControlEnv, self).__init__()

        self.n: int = n  # Number of resistances (actions)
        self.m: int = m  # Number of sensors (observations)
        self.target_temp: float = target_temp
        self.temp_lower_bound: float = temp_lower_bound
        self.max_current: float = max_current

        # Action space: Continuous currents for each resistance
        self.action_space: spaces.Box = spaces.Box(
            low=0.0, high=self.max_current, shape=(self.n,), dtype=np.float32
        )

        # Observation space: Temperatures from m sensors
        self.observation_space: spaces.Box = spaces.Box(
            low=self.temp_lower_bound, high=50.0, shape=(self.m,), dtype=np.float32
        )

        # Internal state (initial temperatures)
        self.state: np.ndarray = np.full(
            (self.m,), self.temp_lower_bound, dtype=np.float32
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one time step in the environment.

        Parameters
        ----------
        action : np.ndarray
            The applied currents to each resistance.

        Returns
        -------
        tuple
            (new state, reward, done, truncated, info)
        """
        heat_effect: float = np.sum(action) * 0.1  # Simplified effect of resistances
        self.state = np.clip(
            self.state + heat_effect - 0.05, self.temp_lower_bound, 50.0
        )  # Cooling effect

        cost: float = np.sum((self.state - self.target_temp) ** 2)
        reward: float = -cost
        done: bool = False

        # When Dani sends me the real physical simulation, I will replace the previous lines by something like
        # self.state = get_evolved_observable_state(self.state, action)
        # reward = - np.sum((get_evolved_complete_state(self.complete_state, action, delta_t) - self.target_temp) ** 2)
        # where get_evolved_observable_state just gives me the subset of temperature values in get_evolved_complete_state
        # where we actually have sensors

        # Optionally, we can normalize the observed state (absolute values are
        # meaningful, thats why we don't use stable_baselines3.common.vec_env.VecNormalize)
        normalized_state = (self.state - self.target_temp) / (50.0 - self.temp_lower_bound)
        return normalized_state, reward, done, False, {}

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        Returns
        -------
        tuple
            (initial state, info dictionary)
        """
        super().reset(seed=seed)
        self.state = np.full((self.m,), self.temp_lower_bound, dtype=np.float32)
        return self.state, {}

    def render(self, mode: str = "human") -> None:
        """
        Renders the environment state.
        """
        print(f"Temperatures: {self.state}")

    def close(self) -> None:
        """
        Closes the environment.
        """
        pass


# Create the environment
env = make_vec_env(lambda: TemperatureControlEnv(), n_envs=1)


# ------------- Train PPO agent on CPU -------------

# Define the network architecture
policy_kwargs = {
    "net_arch": [128, 128]
}
# Shared architecture: net_arch=[128, 128]
# Different architectures for actor and critic:
# net_arch=dict(pi=[32, 32], vf=[64, 64])

# Limit the Kullback–Leibler (KL) divergence between updates,
# because the clipping is not enough to prevent large updates.
kld = 0.01

# Device: "cpu" or "cuda". PPO is meant to be run primarily on the CPU,
# especially when you are not using a CNN. See Stable Baselines 3 PPO docs:
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
device = "cpu"  # Use "cuda" for GPU training

# Model creation
model = PPO("MlpPolicy", env, verbose=1, device=device, policy_kwargs=policy_kwargs, target_kl=kld)

# Model training
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_temperature_control")

# Load the model and test it
model = PPO.load("ppo_temperature_control")
obs = env.reset()
rewards_list = []
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, dones, infos = env.step(action)
    rewards_list.append(reward)
    env.render()

# Plot rewards over time
import matplotlib.pyplot as plt
plt.plot(rewards_list)
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.title("Episode Rewards Over Time")
plt.show()