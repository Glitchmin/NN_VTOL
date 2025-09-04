from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from vtol_base_env import VTOLBaseEnv
wind_is_active = False
wind_vector = np.zeros(3, dtype=np.float32)

def simple_wind_func(time: float, position: np.ndarray) -> np.ndarray:
    global wind_is_active
    global wind_vector
    if wind_is_active:
        return np.tile(wind_vector, (position.shape[0], 1))
    else:
        return np.zeros_like(position)


class VTOLAngleErrorEnv(VTOLBaseEnv):

    def __init__(
            self,
            *,
            sparse_reward: bool = False,
            max_duration_seconds: float = 2,
            agent_hz: int = 240,
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
            low=-1, high=1, shape=(6,), dtype=np.float32
        )

        # ---------------------- OBSERVATION -------------------------
        self.state_length = 23
        self.observation_space: gym.Space = Box(
            low=-np.inf, high=np.inf, shape=(self.state_length,), dtype=np.float32
        )
        self.global_step_count = 0

        # runtime
        self.sparse_reward = sparse_reward
        self.radio_cmd = np.zeros(5, dtype=np.float32)
        self.prev_err = np.zeros(3, dtype=np.float32)
        self.integral_err = np.zeros(3, dtype=np.float32)
        self.state = np.zeros(self.state_length, dtype=np.float32)
        self.action = np.zeros(7, dtype=np.float32)
        self.first_state = True
        self.prev_alt = None
        self.prev_pwm = None
        self.episode_idx = 0
        self.gimbal_bias_range_abs = 0.05
        self.gimbal_bias = [0,0]

        # derived
        self.dt = np.float32(1.0 / 240)  # <<< float32!

        #sensor noise
        self.sensor_noise = True
        gyro_noise_density = 0.02 * np.pi / 180
        bandwidth = agent_hz / 2
        self.gyro_noise_std = gyro_noise_density * np.sqrt(bandwidth)
        self.angle_noise_std = 0.02

        self.wind_start_thresh = 1_000_000  # Global steps to start adding wind
        self.max_wind_speed = 10  # Max wind speed in m/s
        self.min_wind_start_step = 100
        self.wind_start_step = -1  # Step to start wind in current episode (-1 means no wind)

        # ----------------- RANDOM SET-POINT -----------------------------


    def _sample_radio_setpoint(self) -> np.ndarray:
        cmd_type = self.episode_idx % 4
        if cmd_type == 0:
            return np.array([
                np.random.uniform(0.3, 0.7),
                np.random.uniform(-np.deg2rad(15), np.deg2rad(15)),
                np.random.uniform(-np.deg2rad(15), np.deg2rad(15)),
                np.random.uniform(-np.deg2rad(30), np.deg2rad(30)),
                np.random.uniform(0, .4),
            ], dtype=np.float32)
        if cmd_type == 1:
            return np.array([
                np.random.uniform(0.3, 0.7),
                np.random.uniform(-np.deg2rad(30), np.deg2rad(30)),
                0,
                0,
                np.random.uniform(0, .4),
            ], dtype=np.float32)
        if cmd_type == 2:
            return np.array([
                np.random.uniform(0.3, 0.7),
                0,
                np.random.uniform(-np.deg2rad(30), np.deg2rad(30)),
                0,
                np.random.uniform(0, .4),
            ], dtype=np.float32)
        if cmd_type == 3:
            return np.array([
                np.random.uniform(0.3, 0.7),
                0,
                0,
                np.random.uniform(-np.deg2rad(120), np.deg2rad(120)),
                np.random.uniform(0, .4),
            ], dtype=np.float32)

    # ----------------------------- RESET ----------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.episode_idx += 1
        super().begin_reset(seed, options or {})
        self.env.reset()

        self.wind_start_step = -1  # Default to no wind
        self.env.register_wind_field_function(simple_wind_func)

        if self.global_step_count > self.wind_start_thresh:
            # Schedule a start time for the wind
            global wind_vector
            self.wind_start_step = self.np_random.integers(
                self.min_wind_start_step, 200
            )

            wind_speed = self.np_random.uniform(self.max_wind_speed/4, self.max_wind_speed)
            wind_angle = self.np_random.uniform(0, 2 * np.pi)
            wind_vector = np.array([
                wind_speed * np.cos(wind_angle),
                wind_speed * np.sin(wind_angle),
                self.np_random.uniform(-5,5)
            ], dtype=np.float32)

        self.radio_cmd = self._sample_radio_setpoint()
        self.action.fill(0.0)
        self.prev_pwm = None
        self.prev_err.fill(0.0)
        self.integral_err.fill(0.0)
        self.first_state = True
        self.compute_state()
        self.gimbal_bias = [
            np.random.uniform(-self.gimbal_bias_range_abs, self.gimbal_bias_range_abs),
            np.random.uniform(-self.gimbal_bias_range_abs, self.gimbal_bias_range_abs)]
        super().end_reset(seed, options or {})

        return self.state.copy(), self.info

    # ---------------------- STATE CONSTRUCTION ----------------------
    def compute_state(self) -> None:
        """Modified to include sensor noise."""
        ang_vel, ang_pos, *_ = super().compute_attitude()
        roll, pitch, _ = ang_pos.astype(np.float32)
        roll_rate, pitch_rate, yaw_rate = np.float32(ang_vel)

        if self.sensor_noise:
            roll += np.random.normal(0, self.angle_noise_std)
            pitch += np.random.normal(0, self.angle_noise_std)
            roll_rate += np.random.normal(0, self.gyro_noise_std)
            pitch_rate += np.random.normal(0, self.gyro_noise_std)
            yaw_rate += np.random.normal(0, self.gyro_noise_std)

        sp_t, sp_roll, sp_pitch, sp_yaw_rate, sp_arm_tilt = self.radio_cmd

        err = np.array([
            sp_roll - roll,
            sp_pitch - pitch,
            sp_yaw_rate - yaw_rate
        ], dtype=np.float32)

        self.integral_err += self.dt * err
        abs_int_limits = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        self.integral_err = np.clip(self.integral_err, -abs_int_limits, abs_int_limits)

        if self.prev_pwm is None:
            self.prev_pwm = np.zeros(7)

        obs = np.concatenate([
            err,
            self.prev_err,
            self.integral_err,
            [sp_t, sp_arm_tilt],
            [roll, roll_rate, pitch, pitch_rate, yaw_rate],
            self.prev_pwm
        ]).astype(np.float32)

        self.state = obs

        if self.render_mode == "human" or self.episode_idx % 200 == 0:
            print(f"OBS err {err} | "
                  f"prev_err {self.prev_err}  |"
                  f"int_err {self.integral_err}  |"
                  f"sp_t {sp_t}  "
                  f"sp arm_tilt {sp_arm_tilt} "
                  f"additional {[roll, roll_rate, pitch, pitch_rate, yaw_rate]}  "
                  f"prev_pwm {self.prev_pwm}")

    def scale_throttle(self):
        target_sum = np.clip(self.radio_cmd[0], 0.05, 1.0) * 3

        for _ in range(4):
            delta = target_sum - np.sum(self.action[:3])
            if np.isclose(delta, 0):
                break

            adjustable = (self.action[:3] < 1.0) if delta > 0 else (self.action[:3] > 0.05)
            num_adjustable = np.sum(adjustable)

            if num_adjustable == 0:
                break

            self.action[:3][adjustable] += delta / num_adjustable
            self.action[:3] = np.clip(self.action[:3], 0.05, 1.0)

    # ------------------------------ STEP ----------------------------
    def step(self, action: np.ndarray):
        self.global_step_count += 1

        self.action = action
        self.action[:3] = (self.action[:3] + 1) / 2
        self.action[:3] = np.clip(self.action[:3], .05, 1)
        self.scale_throttle()
        self.action[3] /= 4
        gimbals = [self.radio_cmd[4] + self.action[3], self.radio_cmd[4] - self.action[3]]
        gimbals = np.clip(gimbals, 0, 1)
        gimbals += self.gimbal_bias
        gimbals = np.clip(gimbals, -.11, 1)
        self.pwm = np.concatenate([self.action[:3], gimbals, self.action[4:]])
        # self.pwm = np.ceil(self.pwm * 1000) / 1000
        if self.prev_pwm is None:
            self.prev_pwm = self.pwm.copy()
        self.env.set_setpoint(0,  self.pwm)

        roll, pitch = self.env.state(0)[1][:2].astype(np.float32)
        roll_rate, pitch_rate, yaw_rate = np.float32(self.env.state(0)[0])
        sp_t, sp_roll, sp_pitch, sp_yaw_rate, sp_arm_tilt = self.radio_cmd
        self.prev_err = np.array([sp_roll - roll, sp_pitch - pitch, sp_yaw_rate - yaw_rate])

        global wind_is_active
        if self.wind_start_step != -1 and self.step_count >= self.wind_start_step:
            wind_is_active = True

        for _ in range(self.env_step_ratio):
            if self.termination or self.truncation:
                break
            self.env.step()

        self.reward = 0.1
        self.compute_state()
        self._compute_term_trunc_reward()
        self.step_count += 1
        self.prev_pwm = self.pwm.copy()

        return (
            self.state.copy(),
            float(self.reward),
            bool(self.termination),
            bool(self.truncation),
            self.info,
        )

    # ------------------- TERMINATION / REWARD -----------------------
    def _compute_term_trunc_reward(self) -> None:
        super().compute_base_term_trunc_reward()

        # ---------- safety termination (same as before) -------------
        roll, pitch = self.env.state(0)[1][:2].astype(np.float32)
        roll_rate, pitch_rate, yaw_rate = np.float32(self.env.state(0)[0])

        sp_t, sp_roll, sp_pitch, sp_yaw_rate, sp_arm_tilt = self.radio_cmd

        e_roll = roll - sp_roll
        e_pitch = pitch - sp_pitch
        e_yaw = yaw_rate - sp_yaw_rate
        e_pitch_r = pitch_rate
        e_roll_r = roll_rate
        e_thr = np.sum(abs(self.action[:3] - sp_t))
        delta_m_max = 0.3
        delta_m = np.abs(self.pwm[:3] - self.prev_pwm[:3])
        e_osc_m = np.sum(np.maximum(0, delta_m_max - delta_m))
        delta_s_max = 0.3
        delta_s = np.abs(self.pwm[3:5] - self.prev_pwm[3:5])
        e_osc_s = np.sum(np.maximum(0, delta_s_max - delta_s))

        if self.render_mode == "human" or self.episode_idx % 200 == 0:
            print(
                f"ERR pitch {np.rad2deg(e_pitch):5.1f}°  "
                f"yaẇ {np.rad2deg(e_yaw):6.1f}°/s  "
                f" roll {np.rad2deg(e_roll):5.1f}° | "
                f"CMD t {sp_t:4.2f} "
                f"pitch {np.rad2deg(sp_pitch):5.1f}°  "
                f"yaw {np.rad2deg(sp_yaw_rate):6.1f}°/s  "
                f"roll {np.rad2deg(sp_roll):5.1f}° "
                f"arm {sp_arm_tilt:5.1f} | "
                f"ATT  roll {np.rad2deg(roll):5.1f}°  "
                f"pitch {np.rad2deg(pitch):5.1f}°  "
                f"yaw rate {np.rad2deg(yaw_rate):6.1f}°/s  "
                f"alt {self.env.state(0)[-1][2]:5.1f} m \n "
                f"wind active: {wind_is_active} wind vec: {wind_vector} "
                f" | ACTION: {self.action} \n"
                f" | PWM: {self.pwm}"
            )

        err_vec = np.abs(np.array([e_roll, e_pitch, e_yaw, e_osc_m, e_osc_s], dtype=np.float32))

        w = np.array([.3, .3, .25, -.25, -.15], dtype=np.float32)

        err_vec *= w

        self.reward -= np.sum(err_vec)
        if self.render_mode == "human" or self.episode_idx % 200 == 0:
            print(f"roll {-err_vec[0]:5.1f}  "
                  f"reward pitch {-err_vec[1]:5.1f}  "
                  f"reward yaw {-err_vec[2]:5.1f}  "
                  f"reward osc m {-err_vec[3]:5.1f}  "
                  f"reward osc s {-err_vec[4]:5.1f}  "
                  f"reward sum {-np.sum(err_vec)}")

# ------------------------- GYM REGISTRATION -------------------------
from gymnasium.envs.registration import register

register(
    id="VTOL-AngleErr-v0",
    entry_point="VTOL_train_angle_error_v0:VTOLAngleErrorEnv",
    max_episode_steps=int(2 * 480),
)
