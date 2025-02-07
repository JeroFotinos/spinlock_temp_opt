import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Updated Heat Equation Solver with Vectorized Update ---
class HeatEquationSolver:
    def __init__(self, nodes_x, nodes_y, dx, dy, dt, alpha, Lx, Ly):
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.dx = dx
        self.dy = dy
        self.dt = dt  # time step for the solver
        self.alpha = alpha
        # Initialize grid state at room temperature (25°C)
        self.state = np.full((nodes_y, nodes_x), 25.0)
        self.apply_boundary_conditions()
        # Gaussian width (in meters) for the heat source
        self.sigma = 0.01  
        # Create meshgrid for physical coordinates
        self.x_vals = np.linspace(0, Lx, nodes_x)
        self.y_vals = np.linspace(0, Ly, nodes_y)
        self.X, self.Y = np.meshgrid(self.x_vals, self.y_vals)

    def apply_boundary_conditions(self):
        # Enforce room temperature (25°C) at the boundaries
        self.state[0, :] = 25.0
        self.state[-1, :] = 25.0
        self.state[:, 0] = 25.0
        self.state[:, -1] = 25.0

    def get_evolved_complete_state_multi(self, t_i, t_final, current_state, a_vector, centers):
        """
        Vectorized evolution of the temperature grid from t_i to t_final.
        a_vector: array of amplitudes (one per resistor).
        centers: list of (x, y) positions for each heat source.
        """
        new_state = current_state.copy()
        # Compute the total heat source: sum of Gaussian sources
        total_heat_source = np.zeros_like(new_state)
        for a, center in zip(a_vector, centers):
            x_center, y_center = center
            heat_source = a * np.exp(-(((self.X - x_center)**2 + (self.Y - y_center)**2) /
                                        (2 * self.sigma**2)))
            total_heat_source += heat_source

        t = t_i
        while t < t_final:
            w = new_state.copy()
            # Compute the Laplacian using slicing (vectorized)
            laplacian = ((w[:-2, 1:-1] - 2*w[1:-1, 1:-1] + w[2:, 1:-1]) / (self.dx**2) +
                         (w[1:-1, :-2] - 2*w[1:-1, 1:-1] + w[1:-1, 2:]) / (self.dy**2))
            # Update interior points in one vectorized operation.
            new_state[1:-1, 1:-1] = w[1:-1, 1:-1] + self.dt * (self.alpha * laplacian + total_heat_source[1:-1, 1:-1])
            t += self.dt
        return new_state

# --- Updated Environment Methods with Episode Termination ---
class TemperatureControlEnv(gym.Env):
    """
    Gym environment for temperature control with discrete (on/off) actions.
    Each episode lasts for total_episode_time (default 600 sec) with t_action (default 0.5 sec) per step.
    """
    def __init__(
        self,
        n: int = 3,         # Number of resistors (and actions)
        m: int = 3,         # Number of sensors (observations)
        target_temp: float = 27.0,
        temp_lower_bound: float = 25.0,
        t_action: float = 0.5,         # Duration (in seconds) for each action
        total_episode_time: float = 600.0,  # Total episode time in seconds (10 minutes)
        # Grid parameters:
        Lx: float = 0.7,
        Ly: float = 0.5,
        nodes_x: int = 200,
        nodes_y: int = 150,
    ):
        super(TemperatureControlEnv, self).__init__()
        self.n = n
        self.m = m
        self.target_temp = target_temp
        self.temp_lower_bound = temp_lower_bound
        self.t_action = t_action
        self.total_episode_time = total_episode_time

        # Calculate maximum steps per episode (e.g., 600 / 0.5 = 1200 steps)
        self.max_steps = int(total_episode_time / t_action)
        self.current_step = 0

        # Define the action and observation spaces
        self.action_space = spaces.MultiDiscrete([2] * self.n)
        self.observation_space = spaces.Box(
            low=self.temp_lower_bound, high=50.0, shape=(self.m,), dtype=np.float32
        )

        # Set up the simulation grid parameters.
        self.Lx = Lx
        self.Ly = Ly
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.dx = Lx / (nodes_x - 1)
        self.dy = Ly / (nodes_y - 1)

        # Material properties (Aluminum)
        thermal_conductivity = 237          # W/m·K
        specific_heat_capacity = 0.90 * 1000  # J/kg·K
        density = 2710                      # kg/m³
        self.alpha = thermal_conductivity / (density * specific_heat_capacity)

        # Choose a time step dt for the solver (stability condition)
        self.dt_solver = min(self.dx**2, self.dy**2) / (4 * self.alpha)

        # Create an instance of the heat equation solver.
        self.solver = HeatEquationSolver(nodes_x, nodes_y, self.dx, self.dy, self.dt_solver, self.alpha, Lx, Ly)

        # Define the positions for the resistors (and colocated sensors).
        self.centers = [(Lx*(i+1)/(n+1), Ly/2) for i in range(n)]
        # For demonstration, place sensors at a fraction of the resistor positions.
        # self.sensor_positions = [(x[0]/2, x[1]/2) for x in self.centers]
        # self.sensor_positions = self.centers
        self.sensor_positions = [(x[0]+0.1, x[1]+0.1) for x in self.centers]

        # For later analysis, store the full grid states at each action step.
        self.state_history = []

    def _get_sensor_observations(self):
        """
        Extracts sensor readings from the grid by selecting the cell nearest to each sensor's position.
        """
        obs = []
        for (x, y) in self.sensor_positions:
            j = int(round((x / self.Lx) * (self.nodes_x - 1)))
            i = int(round((y / self.Ly) * (self.nodes_y - 1)))
            obs.append(self.solver.state[i, j])
        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray):
        # Convert discrete actions into heat source amplitudes.
        power_on = 50
        a_vector = np.array(action, dtype=np.float32) * power_on

        # Evolve the grid state over t_action seconds.
        new_state = self.solver.get_evolved_complete_state_multi(
            t_i=0.0,
            t_final=self.t_action,
            current_state=self.solver.state,
            a_vector=a_vector,
            centers=self.centers
        )
        self.solver.state = new_state
        self.solver.apply_boundary_conditions()
        self.state_history.append(self.solver.state.copy())

        # Compute the cost as the integrated squared error over the grid.
        cost = np.sum((self.solver.state - self.target_temp) ** 2)  # * self.dx * self.dy
        reward = -cost

        obs = self._get_sensor_observations()

        # Increment the step counter.
        self.current_step += 1

        # Use two separate flags for termination.
        terminated = False  # Set to True if a terminal condition is met
        truncated = self.current_step >= self.max_steps

        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None):
        # Reset the step counter.
        self.current_step = 0
        # Reset the grid state to room temperature.
        self.solver.state = np.full((self.nodes_y, self.nodes_x), self.temp_lower_bound) + np.random.normal(0.0, 2.0, (self.nodes_y, self.nodes_x))
        self.solver.apply_boundary_conditions()
        self.state_history = [self.solver.state.copy()]
        # Return observation and info (empty dict)
        return self._get_sensor_observations(), {}

    def render(self, mode="human"):
        # For instance, print sensor readings.
        sensor_obs = self._get_sensor_observations()
        print(f"Sensor readings: {sensor_obs}")

    def close(self):
        pass

# --- Training using PPO ---
# Create a vectorized environment using the custom discrete-action environment.
env = make_vec_env(lambda: TemperatureControlEnv(), n_envs=1)
policy_kwargs = {"net_arch": [8, 8]}

# PPO supports discrete action spaces.
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device="cpu")  # , target_kl=0.01
model.learn(total_timesteps=12000)
model.save("ppo_temperature_control_discrete")

# --- Testing the trained agent with rewards displayed ---
model = PPO.load("ppo_temperature_control_discrete")
obs = env.reset()  # reset returns a tuple (observation, info)

rewards_list = []
num_steps = 500

for step in range(num_steps):
    action, _states = model.predict(obs)
    # Unpack the 4 values: observation, reward, done, and info
    obs, reward, done, info = env.step(action)
    
    # If using a vectorized environment, reward may be an array (with one element).
    step_reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
    rewards_list.append(step_reward)
    
    # Print the reward for this step
    print(f"Step {step}: Reward: {step_reward}")
    
    env.render()

print("Rewards during the episode:", rewards_list)

# --- Plotting Temperature Evolution ---
# Retrieve the stored grid states from the underlying TemperatureControlEnv instance.
grid_states = env.envs[0].unwrapped.state_history

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a figure and axis for the heatmap.
fig, ax = plt.subplots()
# Display the first grid state.
cax = ax.imshow(grid_states[0], cmap='hot', interpolation='nearest', vmin=25, vmax=100)
fig.colorbar(cax)
ax.set_title("Temperature Evolution during Testing Episode")

# Define the animation update function.
def animate(i):
    cax.set_data(grid_states[i])
    ax.set_title(f"Temperature Evolution at Step {i}")
    return [cax]

# Create the animation. Adjust the interval (milliseconds) as desired.
ani = animation.FuncAnimation(fig, animate, frames=len(grid_states), interval=200, blit=False)

plt.show()
