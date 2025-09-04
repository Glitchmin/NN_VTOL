#!/usr/bin/env python3
"""
Evaluate a PPO agent on **QuadX-AngleErr-v0** and plot the four error
channels it is trained on:

    e_throttle   [unitless]
    e_roll       [rad]
    e_pitch      [rad]
    e_yaw_rate   [rad s⁻¹]

The environment’s observation is the 12-vector
    x = [err, Δerr, sp_t, roll, pitch, yaw_rate],
so the first four elements are exactly the errors we want to inspect.

Usage:

    python eval_angle_error.py \
        --model ppo_model_best.zip \
        --steps 1500 \
        --radio 0.1 0.0 1.57 0.3      # throttle roll pitch yawRate
"""
from __future__ import annotations
import argparse
import sys
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import onnx

import QuadX_train_angle_error_v0  # noqa: F401 (registers QuadX-AngleErr-v0)
import VTOL_train_angle_error_v0  # noqa: F401 (registers QuadX-AngleErr-v0)
from stable_baselines3 import PPO
import onnxruntime as ort

RADIO_COMMANDS = np.array(
    [
[0.38, 0.0, 0.0, 0.0, 0.05],
        [0.38, 0.0, 0.0, 0.0, 0.05],
        [0.38, 0.0, 0.0, 0.0, 0.05],
        [0.38, 0.0, 0.0, 0.0, 0.05],
    ],
    dtype=np.float32,
)


def ask_user_radio() -> np.ndarray:
    """Interactively request a 4-tuple radio command from the user."""
    prompt = (
        "Enter desired RADIO set-point as 4 floats:\n"
        "throttle [0–1]  roll [rad]  pitch [rad]  yaw-rate [rad/s]\n> "
    )
    while True:
        try:
            vals = [float(x) for x in input(prompt).strip().split()]
            if len(vals) != 5:
                raise ValueError
            return np.array(vals, dtype=np.float32)
        except ValueError:
            print("Please type **exactly four** numeric values.", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="QuadX-AngleErr evaluator")
    p.add_argument(
        "--model",
        default="ppo_model_best.zip",
        help="Path to Stable-Baselines3 PPO model (.zip)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum agent steps to run",
    )
    p.add_argument(
        "--radio",
        nargs=5,
        type=float,
        metavar=("THROTTLE", "ROLL", "PITCH", "YAW_RATE", "ARM_TILT"),
        help="Fixed radio set-point (throttle [0–1], roll [rad], pitch [rad], yaw-rate [rad/s], arm tilt 1 = 90deg)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    # Load the trained model
    model = PPO.load(args.model)
    onnx_path = "model_stable_240Hz.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)

    # Spawn evaluation environment
    env = gym.make(
        "VTOL-AngleErr-v0",
        render_mode="human",
        max_episode_steps=args.steps
    )
    obs, info = env.reset()

    # Fix (or ask for) radio command
    if args.radio is not None:
        radio_cmd = np.asarray(args.radio, dtype=np.float32)
    else:
        radio_cmd = ask_user_radio()

    # Overwrite the sampled radio command chosen by reset()
    env.unwrapped.radio_cmd = radio_cmd
    env.unwrapped.compute_state()
    obs = env.unwrapped.state.copy()

    # Run episode and record errors
    e_roll, e_pitch, e_yaw_rate, t_axis = [], [], [], []
    dt = 1.0 / 240  # should be 1/240 by default

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)

        if obs.ndim == 1:
            obs_processed = obs.reshape(1, -1).astype(np.float32)
        else:
            obs_processed = obs.astype(np.float32)

        print(obs_processed)

        # Run inference using the ONNX model
        # action = ort_session.run(None, {"obs": obs_processed})[0][0]
        print(action)
        action = np.clip(action, -1, 1)
        obs, reward, done, truncated, info = env.step(action)

        # The first four entries of `obs` are the errors
        e_roll.append(obs[0])
        e_pitch.append(obs[1])
        e_yaw_rate.append(obs[2])
        t_axis.append(step * dt)
        env.unwrapped.radio_cmd = RADIO_COMMANDS[((4 * step) // args.steps)]

        if done or truncated:
            break

    env.close()

    # Plot tracking errors
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, e_roll, label="roll error [rad]")
    plt.plot(t_axis, e_pitch, label="pitch error [rad]")
    plt.plot(t_axis, e_yaw_rate, label="yaw-rate error [rad/s]")
    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.title("VTOL-AngleErr-v0 Tracking Errors")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
