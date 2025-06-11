"""
Note that every submission must contains a file "load_my_policy.py" which contains a function
load_policy(env: WaterChlorinationEnv) -> ChlorinationControlPolicy for loading and returning the final policy/controller.
"""
from env import WaterChlorinationEnv
from control_policy import ChlorinationControlPolicy

from my_policy import MyPolicy


def load_policy(env: WaterChlorinationEnv, zip_file_name: str = "my_ppo_model.zip") -> ChlorinationControlPolicy:
    # Create and load our final policy/controller
    my_policy = MyPolicy(env)
    my_policy.load_from_file(zip_file_name)

    return my_policy
