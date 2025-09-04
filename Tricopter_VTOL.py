"""Implementation of a Tricopter VTOL UAV (with simple DEBUG prints)."""

from __future__ import annotations

import numpy as np
import yaml
import copy
from pybullet_utils import bullet_client

from PyFlyt.core.abstractions.base_controller import ControlClass
from PyFlyt.core.abstractions.base_drone import DroneClass
from PyFlyt.core.abstractions.lifting_surfaces import LiftingSurface, LiftingSurfaces
from PyFlyt.core.abstractions.gimbals import Gimbals
from PyFlyt.core.abstractions.motors import Motors
from PyFlyt.core.abstractions.pid import PID


class Tricopter_VTOL(DroneClass):
    def __init__(
            self,
            p: bullet_client.BulletClient,
            start_pos: np.ndarray,
            start_orn: np.ndarray,
            control_hz: int = 240,
            physics_hz: int = 480,
            drone_model: str = "Pyza",
            model_dir: None | str = None,
            np_random: None | np.random.Generator = None,
            use_camera: bool = False,
            use_gimbal: bool = False,
            camera_angle_degrees: int = 30,
            camera_FOV_degrees: int = 90,
            camera_resolution: tuple[int, int] = (128, 128),
            camera_position_offset: np.ndarray = np.array([-1.0, 0.0, 3.0]),
            camera_fps: None | int = None,
    ):
        """Creates a Tricopter VTOL UAV with co-dependent ailerons, no rudder,
        and two tilting front motors.

        Args:
            p (bullet_client.BulletClient): PyBullet client
            start_pos (np.ndarray): initial XYZ position
            start_orn (np.ndarray): initial quaternion orientation
            control_hz (int): control-loop rate (Hz)
            physics_hz (int): physics update rate (Hz)
            drone_model (str): YAML parameter file stem under `model_dir`
            model_dir (None | str): override directory for YAML/URDF assets
            np_random (None | np.random.Generator): RNG instance
        """
        super().__init__(
            p=p,
            start_pos=start_pos,
            start_orn=start_orn,
            control_hz=control_hz,
            physics_hz=physics_hz,
            drone_model=drone_model,
            model_dir=model_dir,
            np_random=np_random,
        )

        # ------------------------------------------------------------------#
        #  Load all aerodynamic and actuator parameters from YAML            #
        # ------------------------------------------------------------------#
        with open(self.param_path, "rb") as f:
            all_params = yaml.safe_load(f)

        def randomize_params(params):
            """Add domain randomization to motor and gimbal parameters."""
            p = copy.deepcopy(params)

            # Randomize motor params
            p['motor_params']['tau'] *= np.random.uniform(0.5, 1.5)
            p['motor_params']['thrust_coef'] *= np.random.uniform(0.8, 1.2)
            p['motor_params']['torque_coef'] *= np.random.uniform(0.8, 1.2)
            p['motor_params']['total_thrust'] *= np.random.uniform(0.8, 1.2)

            p['gimbal_params']['tau'] *= np.random.uniform(0.8, 1.2)

            return p

        all_params = randomize_params(all_params)

        # --------------------- lifting surfaces ---------------------------#
        surfaces: list[LiftingSurface] = [
            LiftingSurface(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                surface_id=0,
                lifting_unit=np.array([0.0, 0.0, 1.0]),
                forward_unit=np.array([1.0, 0.0, 0.0]),
                **all_params["horizontal_tail_params"],
            ),
            LiftingSurface(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                surface_id=1,
                lifting_unit=np.array([0.0, 1.0, 0.0]),
                forward_unit=np.array([1.0, 0.0, 0.0]),
                **all_params["vertical_tail_params"],
            ),
            LiftingSurface(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                surface_id=6,
                lifting_unit=np.array([0.0, 0.0, 1.0]),
                forward_unit=np.array([1.0, 0.0, 0.0]),
                **all_params["left_wing_flapped_params"],
            ),
            LiftingSurface(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                surface_id=7,
                lifting_unit=np.array([0.0, 0.0, 1.0]),
                forward_unit=np.array([1.0, 0.0, 0.0]),
                **all_params["right_wing_flapped_params"],
            ),
            LiftingSurface(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                surface_id=2,
                lifting_unit=np.array([0.0, 0.0, 1.0]),
                forward_unit=np.array([1.0, 0.0, 0.0]),
                **all_params["main_wing_params"],
            ),
        ]

        self.lifting_surfaces = LiftingSurfaces(lifting_surfaces=surfaces)

        # --------------------- motors -------------------------------------#
        motor_params = all_params["motor_params"]
        motor_ids = [9, 11, 5]

        thrust_coef = np.array([motor_params["thrust_coef"]] * 3)
        torque_coef = np.array(
            [
                -motor_params["torque_coef"],
                +motor_params["torque_coef"],
                +motor_params["torque_coef"],
            ]
        )
        thrust_unit = np.array([[0.0, 0.0, 1.0]] * 3)
        noise_ratio = np.array([1.0] * 3) * motor_params["noise_ratio"]
        max_rpm = np.array([1.0] * 3) * np.sqrt(
            motor_params["total_thrust"] / (3 * motor_params["thrust_coef"])
        ) * [np.random.uniform(0.95, 1.05), np.random.uniform(0.95, 1.05), np.random.uniform(.8,2.)]
        tau = np.array([1.0] * 3) * motor_params["tau"]

        self.motors = Motors(
            p=self.p,
            physics_period=self.physics_period,
            np_random=self.np_random,
            uav_id=self.Id,
            motor_ids=motor_ids,
            tau=tau,
            max_rpm=max_rpm,
            thrust_coef=thrust_coef,
            torque_coef=torque_coef,
            thrust_unit=thrust_unit,
            noise_ratio=noise_ratio,
        )

        # --------------------- tilting-motor gimbals ----------------------#
        gimbal_params = all_params["gimbal_params"]
        gimbal_unit_1 = np.array(
            [
                [0.0, 1.0, 0.0],  # body-Y ⇒ pitch
                [0.0, 1.0, 0.0],
            ]
        )
        gimbal_unit_2 = np.array(
            [
                [1.0, 0.0, 0.0],  # disable 2nd axis (roll) for both
                [1.0, 0.0, 0.0],
            ]
        )
        gimbal_tau = np.array([gimbal_params["tau"]] * 2)
        gimbal_range = np.array(
            [
                np.array([gimbal_params["range_deg"]] * 2),
                np.array([gimbal_params["range_deg"]] * 2),
            ]
        )

        self.gimbals = Gimbals(
            p=self.p,
            physics_period=self.physics_period,
            np_random=self.np_random,
            gimbal_unit_1=gimbal_unit_1,
            gimbal_unit_2=gimbal_unit_2,
            gimbal_tau=gimbal_tau,
            gimbal_range_degrees=gimbal_range,
        )
        ctrl_params = all_params["control_params"]

        self.Kp_ang_vel = np.array(ctrl_params["ang_vel"]["kp"])
        self.Ki_ang_vel = np.array(ctrl_params["ang_vel"]["ki"])
        self.Kd_ang_vel = np.array(ctrl_params["ang_vel"]["kd"])
        self.lim_ang_vel = np.array(ctrl_params["ang_vel"]["lim"])

        self.Kp_ang_pos = np.array(ctrl_params["ang_pos"]["kp"])
        self.Ki_ang_pos = np.array(ctrl_params["ang_pos"]["ki"])
        self.Kd_ang_pos = np.array(ctrl_params["ang_pos"]["kd"])
        self.lim_ang_pos = np.array(ctrl_params["ang_pos"]["lim"])

        self.output_map = np.array(
            [
                [+1.0, -0.5, -0.0, +1.0],
                [-1.0, -0.5, -0.0, +1.0],
                [+0.0, +1.0, +0.0, +1.0],
                [+0.0, +0.0, -1.0, +0.0],
            ]
        )

        self.mode = -1

        # ------------------------------------------------------------------#
        #  Camera support placeholder                                        #
        # ------------------------------------------------------------------#
        self.use_camera = False

    # ======================================================================#
    #  Core drone-class overrides                                           #
    # ======================================================================#
    def reset(self) -> None:
        """Resets the vehicle to the initial state."""
        self.setpoint = np.zeros(7) if self.mode == -1 else np.zeros(5)
        self.pwm = np.zeros(7)

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.lifting_surfaces.reset()
        self.motors.reset()
        self.gimbals.reset()

    def set_mode(self, mode: int) -> None:
        """Sets the current flight mode of the vehicle."""
        # print(f"[DEBUG] set_mode() called with mode={mode}")  # DEBUG

        if (mode != -1 and mode != 0) and mode not in self.registered_controllers.keys():
            raise ValueError(
                f"`mode` must be -1 or 0 or be registered in {self.registered_controllers.keys()=}, got {mode}."
            )

        self.mode = mode

        if mode == -1:
            return
        if mode == 0:
            self.setpoint = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            # print(f"[DEBUG] Active flight-mode set to {self.mode}")  # DEBUG
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.control_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos,
                self.Ki_ang_pos,
                self.Kd_ang_pos,
                self.lim_ang_pos,
                self.control_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID]
            for controller in self.PIDs:
                controller.reset()

    def register_controller(
            self,
            controller_id: int,
            controller_constructor: type[ControlClass],
            base_mode: int,
    ) -> None:
        """Registers a new controller for the UAV."""
        self.registered_controllers[controller_id] = controller_constructor
        self.registered_base_modes[controller_id] = base_mode

    def update_control(self, physics_step: int) -> None:
        """Runs through controllers (none for mode −1)."""
        if physics_step % self.physics_control_ratio != 0:
            return
        a_output = self.setpoint[1:4].copy()
        if self.mode == 0:
            a_output = self.PIDs[1].step(self.state[1], a_output)
            a_output[2] = self.setpoint[3]
            a_output = self.PIDs[0].step(self.state[0], a_output)
            # print(self.state[0], "after PID", a_output)
            cmd = np.array([*a_output, self.setpoint[0]])
            # print("cmd", cmd)
            self.pwm = np.matmul(self.output_map, cmd)
            # print(self.pwm)
            self.pwm = np.concatenate([self.pwm, [-self.pwm[-1], a_output[0], a_output[1]]])
            self.pwm[3:5] += self.setpoint[4]
            self.pwm = np.clip(self.pwm, -1, 1)
        if self.mode == -1:
            self.pwm = self.setpoint
        # print("setpoint:", self.setpoint, "a_output", a_output, "pwm:", self.pwm,"\n")

    def update_physics(self) -> None:
        """Propagates actuator commands into PyBullet."""
        # print(f"[DEBUG] update_physics() — setpoint: {self.setpoint}")  # DEBUG

        # Gimbal pitch commands (deg)
        gimbal_cmd = np.zeros((2, 2))
        gimbal_cmd[:, 0] = self.pwm[3:5]
        # print(f"[DEBUG] gimbal_cmd (deg): {gimbal_cmd}")  # DEBUG

        # Re-orient each front-motor thrust vector
        rot_mats = self.gimbals.compute_rotation(gimbal_cmd)
        base_vec = np.array([0.0, 0.0, 1.0])

        for i in range(2):
            self.motors.thrust_unit[i] = (rot_mats[i] @ base_vec).reshape((3, 1))
            # print(
            #     f"[DEBUG] Motor {i} thrust-unit: {self.motors.thrust_unit[i].ravel()}"
            # )  # DEBUG

        # Motors
        motor_rpms_cmd = self.pwm[:3]
        # print(f"[DEBUG] Motor RPM commands: {motor_rpms_cmd}")  # DEBUG
        self.motors.physics_update(motor_rpms_cmd)

        # Control surfaces
        surface_cmd = np.array(
            [
                self.pwm[6],  # elevator
                0.0,  # rudder fixed
                -self.pwm[5],  # left aileron
                self.pwm[5],  # right aileron
                0.0,  # main wing fixed
            ]
        )
        # print(f"[DEBUG] Lifting-surface commands: {surface_cmd}")  # DEBUG
        self.lifting_surfaces.physics_update(surface_cmd)

    def update_state(self) -> None:

        """Updates current state vectors and auxiliary actuator states."""
        # Raw Bullet data
        lin_pos, ang_pos_quat = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel_world, ang_vel_world = self.p.getBaseVelocity(self.Id)

        # Transform velocities into body frame
        rot_mat = np.array(self.p.getMatrixFromQuaternion(ang_pos_quat)).reshape(3, 3).T
        lin_vel_body = rot_mat @ lin_vel_world
        ang_vel_body = rot_mat @ ang_vel_world

        # Euler attitude for readability
        ang_pos_euler = self.p.getEulerFromQuaternion(ang_pos_quat)

        # Build state array shape (4, 3)
        self.state = np.stack(
            [ang_vel_body, ang_pos_euler, lin_vel_body, lin_pos], axis=0
        )

        # Update aero lookup tables, etc.
        self.lifting_surfaces.state_update(rot_mat)

        # Gather actuator internal states
        self.aux_state = np.concatenate(
            [
                self.motors.get_states(),
                self.gimbals.get_states(),
            ]
        )

        # -------------------- DEBUG prints -----------------------------#
        # print(
        #     "[DEBUG] update_state() — "
        #     f"ω_body={self.state[0]}, "
        #     f"euler={self.state[1]}, "
        #     f"v_body={self.state[2]}, "
        #     f"pos_world={self.state[3]}"
        # )  # DEBUG
        # print(f"[DEBUG] Motor internal state:  {self.motors.get_states()}")  # DEBUG
        # print(f"[DEBUG] Gimbal internal state: {self.gimbals.get_states()}")  # DEBUG

    def update_last(self, physics_step: int) -> None:
        """Called at the very end of Aviary.step(); nothing extra needed."""
        pass
