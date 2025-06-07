"""
This file contains a simple PPO-based policy.
"""
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation

from control_policy import ChlorinationControlPolicy


class MyPolicy(ChlorinationControlPolicy):
    def __init__(self, env):
        self._policy = PPO("MlpPolicy", NormalizeObservation(env), device="cpu")
        super().__init__(env=env)

    def load_from_file(self, f_in: str) -> None:
        self._policy.load(f_in, device="cpu")

    def compute_action(self, observations: np.ndarray) -> np.ndarray:
        return self._policy.predict(observations)[0]
