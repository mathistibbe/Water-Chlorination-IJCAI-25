from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from gymnasium.wrappers import NormalizeObservation
from env import WaterChlorinationEnv
from scenarios import load_scenario

def make_env(scenario_id):
    def _init():
        env = WaterChlorinationEnv(**load_scenario(scenario_id))
        return env
    return _init

if __name__ == "__main__":
    envs_to_use = [0]
    envs = [make_env(i) for i in envs_to_use]
    vec_env = DummyVecEnv(envs)  # or DummyVecEnv(envs) for easier debugging
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    continue_at = 21
    vec_env = VecNormalize.load(f"models/vecnorm_{continue_at}k_steps.pkl", vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.3,
        device='cpu',
        verbose=1,
        tensorboard_log="./tensorboard_logs",
    )

    if continue_at > 0:
        model.load(f"models/ppo_{continue_at}k_steps.zip")
        pass

    iterations = 1000
    total_steps = continue_at * 1_000
    for i in range(continue_at, iterations + 1):
        print(f"Iteration: {i} (Steps: {total_steps})")

        model.learn(total_timesteps=10_000, progress_bar=True, reset_num_timesteps=False,
                    tb_log_name="WaterChlorination", )

        total_steps += 10_000
        model.save(f"models/ppo_{total_steps // 1000}k_steps.zip")
        vec_env.save(f"models/vecnorm_{total_steps // 1000}k_steps.pkl")