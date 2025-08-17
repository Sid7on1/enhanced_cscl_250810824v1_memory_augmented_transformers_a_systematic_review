import logging
import math
from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from threading import Lock

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
VELOCITY_THRESHOLD = 0.5
FLOW_THEORY_THRESHOLD = 0.8

# Define exception classes
class RewardSystemError(Exception):
    """Base class for reward system exceptions."""
    pass

class InvalidRewardValueError(RewardSystemError):
    """Raised when an invalid reward value is provided."""
    pass

class RewardSystemConfigurationError(RewardSystemError):
    """Raised when there is an error in the reward system configuration."""
    pass

# Define data structures/models
@dataclass
class Reward:
    """Represents a reward with its value and metadata."""
    value: float
    metadata: Dict[str, str]

# Define the reward system configuration
class RewardSystemConfiguration:
    """Represents the configuration for the reward system."""
    def __init__(self, velocity_threshold: float = VELOCITY_THRESHOLD, flow_theory_threshold: float = FLOW_THEORY_THRESHOLD):
        """
        Initializes the reward system configuration.

        Args:
        - velocity_threshold (float): The velocity threshold for the reward system.
        - flow_theory_threshold (float): The flow theory threshold for the reward system.
        """
        self.velocity_threshold = velocity_threshold
        self.flow_theory_threshold = flow_theory_threshold

# Define the reward system class
class RewardSystem(ABC):
    """Abstract base class for reward systems."""
    def __init__(self, configuration: RewardSystemConfiguration):
        """
        Initializes the reward system.

        Args:
        - configuration (RewardSystemConfiguration): The configuration for the reward system.
        """
        self.configuration = configuration
        self.lock = Lock()

    @abstractmethod
    def calculate_reward(self, state: Dict[str, float]) -> Reward:
        """
        Calculates the reward for the given state.

        Args:
        - state (Dict[str, float]): The state for which to calculate the reward.

        Returns:
        - Reward: The calculated reward.
        """
        pass

    def validate_reward_value(self, value: float) -> None:
        """
        Validates the reward value.

        Args:
        - value (float): The reward value to validate.

        Raises:
        - InvalidRewardValueError: If the reward value is invalid.
        """
        if value < 0 or value > 1:
            raise InvalidRewardValueError("Invalid reward value")

class VelocityThresholdRewardSystem(RewardSystem):
    """Reward system based on the velocity threshold."""
    def calculate_reward(self, state: Dict[str, float]) -> Reward:
        """
        Calculates the reward for the given state based on the velocity threshold.

        Args:
        - state (Dict[str, float]): The state for which to calculate the reward.

        Returns:
        - Reward: The calculated reward.
        """
        with self.lock:
            velocity = state.get("velocity", 0)
            if velocity > self.configuration.velocity_threshold:
                reward_value = 1
            else:
                reward_value = 0
            self.validate_reward_value(reward_value)
            return Reward(reward_value, {"velocity": str(velocity)})

class FlowTheoryRewardSystem(RewardSystem):
    """Reward system based on the flow theory."""
    def calculate_reward(self, state: Dict[str, float]) -> Reward:
        """
        Calculates the reward for the given state based on the flow theory.

        Args:
        - state (Dict[str, float]): The state for which to calculate the reward.

        Returns:
        - Reward: The calculated reward.
        """
        with self.lock:
            flow = state.get("flow", 0)
            if flow > self.configuration.flow_theory_threshold:
                reward_value = 1
            else:
                reward_value = 0
            self.validate_reward_value(reward_value)
            return Reward(reward_value, {"flow": str(flow)})

# Define utility methods
def create_reward_system(configuration: RewardSystemConfiguration) -> RewardSystem:
    """
    Creates a reward system based on the given configuration.

    Args:
    - configuration (RewardSystemConfiguration): The configuration for the reward system.

    Returns:
    - RewardSystem: The created reward system.
    """
    return VelocityThresholdRewardSystem(configuration)

def calculate_reward(reward_system: RewardSystem, state: Dict[str, float]) -> Reward:
    """
    Calculates the reward for the given state using the given reward system.

    Args:
    - reward_system (RewardSystem): The reward system to use.
    - state (Dict[str, float]): The state for which to calculate the reward.

    Returns:
    - Reward: The calculated reward.
    """
    return reward_system.calculate_reward(state)

# Define unit test compatibility
import unittest

class TestRewardSystem(unittest.TestCase):
    def test_velocity_threshold_reward_system(self):
        configuration = RewardSystemConfiguration()
        reward_system = VelocityThresholdRewardSystem(configuration)
        state = {"velocity": 0.6}
        reward = reward_system.calculate_reward(state)
        self.assertEqual(reward.value, 1)

    def test_flow_theory_reward_system(self):
        configuration = RewardSystemConfiguration()
        reward_system = FlowTheoryRewardSystem(configuration)
        state = {"flow": 0.9}
        reward = reward_system.calculate_reward(state)
        self.assertEqual(reward.value, 1)

if __name__ == "__main__":
    unittest.main()