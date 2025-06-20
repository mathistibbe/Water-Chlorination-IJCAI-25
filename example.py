"""
Example of how to use the starter code.
"""
from stable_baselines3.common.vec_env import DummyVecEnv

from env import WaterChlorinationEnv
from gymnasium.wrappers import NormalizeObservation, RescaleAction
from scenarios import load_scenario
from control_policy import ChlorinationControlPolicyRandom, ChlorinationControlPolicyConstant
from load_my_policy import load_policy
from utils import compare_models, plot_reward_histogram
import numpy as np

if __name__ == "__main__":
    plot_reward_histogram("reward_log.txt")
    # Compare multiple policies
    with WaterChlorinationEnv(**load_scenario(scenario_id=6)) as env:
        # Collect data and train afterwards (or try to create a chlorine concentration predictor)
        models = [
            #ChlorinationControlPolicyRandom(env),
            ChlorinationControlPolicyConstant(env, constant_action=np.array(5 * [24])),

            #load_policy(env, zip_file_name="my_ppo_model.zip"),
            #load_policy(env, zip_file_name="my_ppo_model_10k-steps_1_scenario.zip"),
            load_policy(env, zip_file_name="models/ppo_10k_steps.zip"),

        ]
        compare_models(models, env, save_results_to="results.png", extend_eval=True)


