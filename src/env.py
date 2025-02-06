from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np


class AbstractThermalControlEnv(ABC):
    """
    Abstract interface for a thermal control environment.

    This environment simulates a device whose temperature field is computed by solving PDEs
    on a 3D mesh using finite difference methods. The complete state (the full temperature field)
    is hidden from the agent; the agent observes only sensor-based averages. However, the cost is
    computed using the complete state. This interface defines the expected methods for any
    concrete implementation of the thermal control environment.

    Methods:
        reset() -> np.ndarray:
            Reset the simulation and return the initial observation.
        step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
            Apply an action, evolve the simulation by Δt, and return the new observation, reward,
            termination flag, and additional information.
        compute_cost(full_state: np.ndarray) -> float:
            Compute the cost (e.g., the integral of the squared deviation from the target temperature)
            using the complete temperature field.
        get_full_state() -> np.ndarray:
            Return the complete (hidden) state.
        render(mode: str = 'human'):
            Optionally render the current state.

    Properties:
        action_space -> Any:
            The description of the action space (e.g., a gym.spaces.Box instance).
        observation_space -> Any:
            The description of the observation space (e.g., a gym.spaces.Box instance).
    """

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """
        Define and return the action space of the environment.
        For example, if actions are continuous currents for n resistors, one might use:
            gym.spaces.Box(low=0.0, high=I_max, shape=(n,), dtype=np.float32)
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """
        Define and return the observation space of the environment.
        This should describe the sensor-based observations, for instance:
            gym.spaces.Box(low=T_min, high=T_max, shape=(m,), dtype=np.float32)
        where m is the number of sensors.
        """
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the simulation to an initial state.

        Returns:
            observation (np.ndarray): The initial sensor-based observation.
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply an action to the environment and evolve the simulation by a fixed time interval Δt.

        Args:
            action (np.ndarray): The control input (e.g., currents for each resistor).

        Returns:
            observation (np.ndarray): The sensor-based observation after applying the action.
            reward (float): The reward computed from the complete state. Typically, reward = -cost.
            done (bool): Flag indicating whether the episode has ended.
            info (dict): A dictionary with additional diagnostic information.
        """
        pass

    @abstractmethod
    def compute_cost(self, full_state: np.ndarray) -> float:
        """
        Compute the cost functional based on the complete state (temperature field).

        For example, if the target temperature is T_target and the cost may be defined as:
            C[T(x)] = ∫_B [T(x) - T_target]^2 dx,
        then this method should implement the numerical approximation of that integral.

        Args:
            full_state (np.ndarray): The complete temperature field (e.g., as a 3D array).

        Returns:
            cost (float): The computed cost.
        """
        pass

    @abstractmethod
    def get_full_state(self) -> np.ndarray:
        """
        Return the complete (hidden) state of the simulation.

        Returns:
            full_state (np.ndarray): The full temperature field over the simulation domain.
        """
        pass

    def render(self, mode: str = "human"):
        """
        Optionally render the current state of the environment.

        Args:
            mode (str): The mode of rendering (default is 'human').
        """
        # By default, do nothing. Concrete implementations can override this method.
        pass

    def close(self):
        """
        Optionally perform any necessary cleanup.
        """
        pass


# class NaiveEnv(AbstractThermalControlEnv):
