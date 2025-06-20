from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from gymnasium.wrappers import NormalizeObservation, RescaleAction
from env import WaterChlorinationEnv
from scenarios import load_scenario
from utils import LoggingCallback

def make_env(scenario_id):
    def _init():
        env = WaterChlorinationEnv(**load_scenario(scenario_id))
        env = NormalizeObservation(env)  # Normalize observations
        return env
    return _init

if __name__ == "__main__":
    envs_to_use = [10]
    envs = [make_env(i) for i in envs_to_use]
    vec_env = DummyVecEnv(envs)  # or DummyVecEnv(envs) for easier debugging
    continue_at = 0

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=32,
        batch_size=64,
        policy_kwargs={"log_std_init": 3.0, "ortho_init": False},
        learning_rate=3e-4,
        ent_coef=0.00,
        gamma=0.99,
        device='cpu',
        verbose=1,
        tensorboard_log="./tensorboard_logs",
    )

    if continue_at > 0:
        model.load(f"models/ppo_{continue_at}k_steps.zip")
        pass
    else:
        import torch
        with torch.no_grad():
            model.policy.action_net.bias.fill_(24.0)

    iterations = 1000
    total_steps = continue_at * 1_000
    for i in range(continue_at, iterations + 1):
        print(f"Iteration: {i} (Steps: {total_steps})")

        model.learn(total_timesteps=100_000, progress_bar=True,
                    tb_log_name="WaterChlorination", callback=LoggingCallback())

        total_steps += 100_000
        model.save(f"models/ppo_{total_steps // 1000}k_steps.zip")

