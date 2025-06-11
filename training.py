from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from env import WaterChlorinationEnv
from scenarios import load_scenario


def make_env(scenario_id):
    def _init():
        return NormalizeObservation(WaterChlorinationEnv(**load_scenario(scenario_id=scenario_id)))

    return _init


if __name__ == "__main__":
    # Use PPO to train a MLP for mapping observations to actions
    # TODO: You might want to use more than one scenario when training the policy
    # envs = [make_env(i) for i in range(3)]
    # vec_env = DummyVecEnv(envs)
    #
    # max n_steps for the 6 day scenario:
    # n_steps = (24 (days) * 3 (days of simulation) * 60 (minutes per day)) / 5 (minutes per step) = 864
    # n_steps = 864
    # n_envs = ...
    with WaterChlorinationEnv(**load_scenario(scenario_id=6)) as env:
        model = PPO("MlpPolicy", env=NormalizeObservation(env), n_steps=128, batch_size=64, device='cpu', verbose=0)
        model.learn(total_timesteps=10_000, progress_bar=True)
        model.save("my_ppo_model_10k-steps_1_scenario_custom_reward_1.0.zip")
