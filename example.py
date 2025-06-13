"""
Example of how to use the starter code.
"""
import numpy as np

from env import WaterChlorinationEnv
from scenarios import load_scenario
from control_policy import ChlorinationControlPolicyRandom, ChlorinationControlPolicyConstant
from load_my_policy import load_policy
from utils import compare_models

if __name__ == "__main__":
    # Compare multiple policies
    with WaterChlorinationEnv(**load_scenario(scenario_id=6)) as env:
        # Collect data and train afterwards (or try to create a chlorine concentration predictor)
        models = [
            ChlorinationControlPolicyRandom(env),
            #ChlorinationControlPolicyConstant(env, constant_action=np.array([500, 500, 500, 500, 500])),
            #load_policy(env, zip_file_name="my_ppo_model.zip"),
            #load_policy(env, zip_file_name="my_ppo_model_10k-steps_1_scenario.zip"),
            #load_policy(env, zip_file_name="my_ppo_model_10k-steps_3_scenario.zip"),
            #load_policy(env, zip_file_name="my_ppo_model_10k-steps_1_scenario_custom_reward_1.0.zip"),
        ]
        compare_models(models, env, save_results_to="results.png")


