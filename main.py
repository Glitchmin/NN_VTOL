import os
import gymnasium as gym
import PyFlyt.gym_envs  # noqa side-effect import
# import QuadX_train_angle_error_v0  # noqa side-effect import
import VTOL_train_angle_error_v0  # noqa side-effect import

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)


class RewardPrintCallback(BaseCallback):
    """Prints mean episode reward every `check_freq` environment steps."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.ep_returns, self.return_sum = [], 0.0
        self.ep_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("rewards"):
            self.return_sum += self.locals["rewards"][0]

        if self.locals.get("dones") and self.locals["dones"][0]:
            self.ep_returns.append(self.return_sum)
            self.return_sum = 0.0
            self.ep_count += 1

        if self.n_calls % self.check_freq == 0:
            if self.ep_returns:
                mean_r = sum(self.ep_returns) / len(self.ep_returns)
                print(f"[{self.n_calls:>8}] episodes={self.ep_count:<5} "
                      f"mean reward ={mean_r:8.2f}")
        return True


# ------------------------------------------------------------------ #
#                             helpers                                #
# ------------------------------------------------------------------ #
def make_env():
    """Factory for the training/eval environments."""
    return gym.make("VTOL-AngleErr-v0", render_mode=None)


def make_eval_env():
    """Factory for the training/eval environments."""
    return gym.make("VTOL-AngleErr-v0", render_mode=None)


def lr_schedule(progress):
    return 3e-4 * progress


# ------------------------------------------------------------------ #
#                             main                                   #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # -------- training VecEnv --------------------------------------
    env = DummyVecEnv([make_env])

    eval_env = DummyVecEnv([make_eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints",  # → checkpoints/best_model.zip
        log_path="checkpoints/eval_logs",
        eval_freq=100_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[32, 32])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0
    )

    total_timesteps = 10_000_000

    callback = CallbackList([
        RewardPrintCallback(check_freq=1_000, verbose=1),
        eval_callback,
    ])

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("ppo_model_best.zip")
