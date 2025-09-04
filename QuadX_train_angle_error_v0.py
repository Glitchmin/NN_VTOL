from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv


class QuadXAngleErrorEnv(QuadXBaseEnv):

    def __init__(
            self,
            *,
            sparse_reward: bool = False,
            max_duration_seconds: float = 2,
            agent_hz: int = 120,
            render_mode: None | str = None,
            render_resolution: tuple[int, int] = (480, 480),
    ):
        super().__init__(
            flight_mode=-1,
            flight_dome_size=np.inf,
            max_duration_seconds=max_duration_seconds,
            angle_representation="euler",
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
            start_pos=np.array([[0, 0, 200]]),
        )

        # ------------------------- ACTION ---------------------------
        self.action_space: gym.Space = Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        # ---------------------- OBSERVATION -------------------------
        self.state_length = 12
        self.observation_space: gym.Space = Box(
            low=-np.inf, high=np.inf, shape=(self.state_length,), dtype=np.float32
        )
        self.global_step_count = 0

        # runtime
        self.sparse_reward = sparse_reward
        self.radio_cmd = np.zeros(4, dtype=np.float32)
        self.prev_err = np.zeros(4, dtype=np.float32)
        self.state = np.zeros(self.state_length, dtype=np.float32)
        self.action = np.zeros(4, dtype=np.float32)
        self.prev_action = None
        self.first_state = True
        self.prev_alt = None
        self.episode_idx = 0

        # derived
        self.dt = np.float32(1.0 / 240.)  # <<< float32!

        # ----------------- RANDOM SET-POINT -----------------------------

    def _sample_radio_setpoint(self) -> np.ndarray:
        return np.array([
            np.random.uniform(0.3, 0.8),
            np.random.uniform(-np.deg2rad(20), np.deg2rad(20)),
            np.random.uniform(-np.deg2rad(20), np.deg2rad(20)),
            np.random.uniform(-np.deg2rad(60), np.deg2rad(60)),
        ], dtype=np.float32)

    # ----------------------------- RESET ----------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.episode_idx += 1
        super().begin_reset(seed, options or {})
        self.env.reset()

        self.radio_cmd = self._sample_radio_setpoint()
        self.action.fill(0.0)
        self.prev_action = None
        self.prev_err.fill(0.0)
        self.first_state = True
        self.compute_state()
        super().end_reset(seed, options or {})
        return self.state.copy(), self.info

    # ---------------------- STATE CONSTRUCTION ----------------------
    def compute_state(self) -> None:
        ang_vel, ang_pos, *_ = super().compute_attitude()
        roll, pitch, _ = ang_pos.astype(np.float32)
        yaw_rate = np.float32(ang_vel[2])

        sp_t, sp_roll, sp_pitch, sp_yaw_rate = self.radio_cmd

        err = np.array([
            np.sum(np.abs(np.clip((self.action+1)/2, 0, 1) - sp_t)),
            sp_roll - roll,
            sp_pitch - pitch,
            sp_yaw_rate - yaw_rate
        ], dtype=np.float32)

        if self.first_state:
            d_err = np.zeros_like(err)
            self.first_state = False
        else:
            d_err = (err - self.prev_err) / self.dt

        self.prev_err = err

        obs = np.concatenate(
            [err, d_err, [sp_t], [roll, pitch, yaw_rate]]).astype(
            np.float32)
        self.state = obs

    # ------------------------------ STEP ----------------------------
    def step(self, action: np.ndarray):
        self.global_step_count += 1

        self.prev_action = self.action.copy()
        self.action = action
        # self.env.set_setpoint(0, np.clip(self.radio_cmd[0] + self.action, 0, 1))
        self.env.set_setpoint(0, np.clip((self.action+1)/2, 0, 1))

        self.reward = 0.1
        for _ in range(self.env_step_ratio):
            if self.termination or self.truncation:
                break
            self.env.step()
            self.compute_state()
            self._compute_term_trunc_reward()

        self.step_count += 1

        return (
            self.state.copy(),
            float(self.reward),
            bool(self.termination),
            bool(self.truncation),
            self.info,
        )

    # ------------------- TERMINATION / REWARD -----------------------
    def _compute_term_trunc_reward(self) -> None:
        """Safety checks + dense attitude/velocity reward.


        Reward  = − ∑ wᵢ · eᵢ²
        e = [throttle-err, roll-err, pitch-err, yaw-rate-err]
        """
        super().compute_base_term_trunc_reward()

        # ---------- safety termination (same as before) -------------
        roll, pitch = self.env.state(0)[1][:2].astype(np.float32)
        yaw_rate = np.float32(self.env.state(0)[0][2])

        sp_t, sp_roll, sp_pitch, sp_yaw_rate = self.radio_cmd

        # e_t = np.sum(self.action)
        e_t = np.sum(np.abs(np.clip((self.action+1)/2, 0, 1) - sp_t))
        e_roll = roll - sp_roll
        e_pitch = pitch - sp_pitch
        e_yaw = yaw_rate - sp_yaw_rate
        e_over = np.sum(self.radio_cmd[0] + self.action < 0) + np.sum(self.radio_cmd[0] + self.action > 1)
        e_over = 0

        if self.render_mode == "human" or self.episode_idx % 200 == 0:
            print(
                f"ERR t {e_t:4.2f} "
                f"pitch {np.rad2deg(e_pitch):5.1f}°  "
                f"yaẇ {np.rad2deg(e_yaw):6.1f}°/s  "
                f" roll {np.rad2deg(e_roll):5.1f}° | "
                f"CMD t {sp_t:4.2f} "
                f"pitch {np.rad2deg(sp_pitch):5.1f}°  "
                f"yaw {np.rad2deg(sp_yaw_rate):6.1f}°/s  "
                f"roll {np.rad2deg(sp_roll):5.1f}° | "
                f"ATT  roll {np.rad2deg(roll):5.1f}°  "
                f"pitch {np.rad2deg(pitch):5.1f}°  "
                f"yaw rate{np.rad2deg(yaw_rate):6.1f}°/s  "
                f"alt {self.env.state(0)[-1][2]:5.1f} m"
                f" | ACTION: {np.clip((self.action+1)/2, 0, 1)}"
            )

        err_vec = np.abs(np.array([e_t, e_roll, e_pitch, e_yaw, e_over], dtype=np.float32))

        w = np.array([.2, .2, .2, .07, 1], dtype=np.float32)
        err_vec *= w
        self.reward -= np.sum(np.abs(err_vec)) - .5

        if self.render_mode == "human" or self.episode_idx % 200 == 0:
            print(f"reward t {-err_vec[0]:5.1f}  "
                  f"roll {-err_vec[1]:5.1f}  "
                  f"reward pitch {-err_vec[2]:5.1f}  "
                  f"reward yaw {-err_vec[3]:5.1f}  "
                  f"reward overflow {-err_vec[4]:5.1f}")


# ------------------------- GYM REGISTRATION -------------------------
from gymnasium.envs.registration import register

register(
    id="QuadX-AngleErr-v0",
    entry_point="QuadX_train_angle_error_v0:QuadXAngleErrorEnv",
    max_episode_steps=int(120 * 2),
)
