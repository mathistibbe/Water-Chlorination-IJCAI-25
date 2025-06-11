"""
Example of how to use the starter code.
"""
from env import WaterChlorinationEnv
from scenarios import load_scenario
from control_policy import ChlorinationControlPolicyRandom
from evaluation import evaluate
from load_my_policy import load_policy
from utils import compare_models

if __name__ == "__main__":
    # Compare multiple policies
    with WaterChlorinationEnv(**load_scenario(scenario_id=6)) as env:
        models = [
            load_policy(env, zip_file_name="my_ppo_model.zip"),
            load_policy(env, zip_file_name="my_ppo_model_10k-steps_1_scenario.zip"),
            load_policy(env, zip_file_name="my_ppo_model_10k-steps_3_scenario.zip"),
        ]
        compare_models(models, env, save_results_to="results.png")


