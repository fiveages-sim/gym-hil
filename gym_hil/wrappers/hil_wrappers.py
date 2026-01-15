#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import time

import gymnasium as gym
import numpy as np

from gym_hil.mujoco_gym_env import MAX_GRIPPER_COMMAND

DEFAULT_EE_STEP_SIZE = {"x": 0.025, "y": 0.025, "z": 0.025,"r_x": 0.025, "r_y": 0.025, "r_z": 0.025,}


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        info["discrete_penalty"] = 0.0
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.1
        ):
            info["discrete_penalty"] = self.penalty

        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return observation, reward, terminated, truncated, info


class EEActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_step_size, use_gripper=False):
        super().__init__(env)
        self.ee_action_step_size = ee_action_step_size
        self.use_gripper = use_gripper
        self._is_dual = (
            hasattr(env.action_space, "shape")
            and env.action_space.shape
            and env.action_space.shape[0] > 7
            and env.action_space.shape[0] % 2 == 0
        )

        self._ee_step_size = np.array(
            [
                ee_action_step_size["x"],
                ee_action_step_size["y"],
                ee_action_step_size["z"],
                # ee_action_step_size["r_x"],
                # ee_action_step_size["r_y"],
                # ee_action_step_size["r_z"],
            ]
        )
        num_actions = 6
        action_space_bounds_min = -np.ones(num_actions)
        action_space_bounds_max = np.ones(num_actions)

        if self.use_gripper:
            action_space_bounds_min = np.concatenate([action_space_bounds_min, [0.0]])
            action_space_bounds_max = np.concatenate([action_space_bounds_max, [2.0]])
            num_actions += 1

        if self._is_dual:
            action_space_bounds_min = np.tile(action_space_bounds_min, 2)
            action_space_bounds_max = np.tile(action_space_bounds_max, 2)
            num_actions *= 2

        ee_action_space = gym.spaces.Box(
            low=action_space_bounds_min,
            high=action_space_bounds_max,
            shape=(num_actions,),
            dtype=np.float32,
        )
        self.action_space = ee_action_space

    def action(self, action):
        """
        Mujoco env is expecting a 7D action space
        [x, y, z, rx, ry, rz, gripper_open]
        For the moment we only control the x, y, z, gripper
        Now supports rotation control if action has 6 dimensions
        """
        def _map_single_arm(single_action):
            action_xyz = single_action[:3] * self._ee_step_size
            if len(single_action) >= 6:
                rotation_step_size = 0.1
                actions_orn = single_action[3:6] * rotation_step_size
            else:
                actions_orn = np.zeros(3)

            gripper_open_command = [0.0]
            if self.use_gripper:
                gripper_open_command = [single_action[-1] - 1.0]

            return np.concatenate([action_xyz, actions_orn, gripper_open_command])

        if self._is_dual and len(action) % 2 == 0:
            half = len(action) // 2
            left_action = _map_single_arm(action[:half])
            right_action = _map_single_arm(action[half:])
            return np.concatenate([left_action, right_action])

        return _map_single_arm(action)


class InputsControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=False,
        auto_reset=False,
        input_threshold=0.001,
        use_gamepad=True,
        controller_config_path=None,
        lock_z_on_xy=True,
    ):
        """
        Initialize the inputs controller wrapper.

        Args:
            env: The environment to wrap
            x_step_size: Base movement step size for X axis in meters
            y_step_size: Base movement step size for Y axis in meters
            z_step_size: Base movement step size for Z axis in meters
            use_gripper: Whether to use gripper control
            auto_reset: Whether to auto reset the environment when episode ends
            input_threshold: Minimum movement delta to consider as active input
            use_gamepad: Whether to use gamepad or keyboard control
            controller_config_path: Path to the controller configuration JSON file
            lock_z_on_xy: Whether to lock Z movement when X/Y translation is active
        """
        super().__init__(env)
        from gym_hil.wrappers.intervention_utils import (
            GamepadController,
            GamepadControllerHID,
            KeyboardController,
        )

        # use HidApi for macos
        if use_gamepad:
            if sys.platform == "darwin":
                self.controller = GamepadControllerHID(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                )
            else:
                self.controller = GamepadController(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                    config_path=controller_config_path,
                )
        else:
            self.controller = KeyboardController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )

        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.lock_z_on_xy = lock_z_on_xy
        self._last_gripper_action_left = 1.0
        self._last_gripper_action_right = 1.0
        self._last_dual_action = None
        self.controller.start()

    def get_gamepad_action(self):
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller (may include rotation)
        deltas = self.controller.get_deltas()
        
        # Handle both 3D and 6D deltas
        if len(deltas) == 6:
            delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz = deltas
            if self.lock_z_on_xy and (delta_x != 0 or delta_y != 0):
                delta_z = 0.0
            # Create 6D action including rotation
            gamepad_action = np.array([delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz], dtype=np.float32)
        else:
            delta_x, delta_y, delta_z = deltas[:3]
            if self.lock_z_on_xy and (delta_x != 0 or delta_y != 0):
                delta_z = 0.0
            # Create 3D action for translation only
            gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        intervention_is_active = self.controller.should_intervene()

        if self.use_gripper:
            active_arm = self.controller.get_active_arm()
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gripper_value = 2.0
                if active_arm == "right":
                    self._last_gripper_action_right = gripper_value
                else:
                    self._last_gripper_action_left = gripper_value
            elif gripper_command == "close":
                gripper_value = 0.0
                if active_arm == "right":
                    self._last_gripper_action_right = gripper_value
                else:
                    self._last_gripper_action_left = gripper_value
            else:
                gripper_value = (
                    self._last_gripper_action_right
                    if active_arm == "right"
                    else self._last_gripper_action_left
                )
            gamepad_action = np.concatenate([gamepad_action, [gripper_value]])

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        cfg.
            action: Original action from agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        if is_intervention:
            gamepad_action = np.asarray(gamepad_action, dtype=np.float32)
            if isinstance(action, np.ndarray):
                action_size = action.size
            else:
                action_size = len(action)

            if action_size == gamepad_action.size:
                action = gamepad_action
            elif action_size == gamepad_action.size * 2:
                base_action = self._last_dual_action
                if base_action is None or base_action.size != action_size:
                    base_action = np.asarray(action, dtype=gamepad_action.dtype)
                    if base_action.size != action_size:
                        base_action = np.zeros(action_size, dtype=gamepad_action.dtype)
                        if self.use_gripper and gamepad_action.size > 0:
                            base_action[gamepad_action.size - 1] = 1.0
                            base_action[-1] = 1.0
                action = base_action.copy()
                if self.controller.get_active_arm() == "right":
                    action[gamepad_action.size:] = gamepad_action
                else:
                    action[:gamepad_action.size] = gamepad_action
            else:
                action = gamepad_action
        else:
            if isinstance(action, np.ndarray):
                action_size = action.size
            else:
                action_size = len(action)

            if self.use_gripper and action_size > 0:
                if isinstance(action, np.ndarray):
                    action_arr = action
                else:
                    action_arr = np.asarray(action, dtype=np.float32)
                if action_size % 2 == 0:
                    half = action_size // 2
                    self._last_gripper_action_left = float(action_arr[half - 1])
                    self._last_gripper_action_right = float(action_arr[-1])
                else:
                    self._last_gripper_action_left = float(action_arr[-1])

        if isinstance(action, np.ndarray):
            action_arr = action.astype(np.float32, copy=False)
        else:
            action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.size % 2 == 0 and action_arr.size > 0:
            self._last_dual_action = action_arr.copy()

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode


        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        info["is_intervention"] = is_intervention
        info["active_arm"] = self.controller.get_active_arm()
        action_intervention = action

        info["action_intervention"] = action_intervention
        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def _maybe_reset_simulation(self) -> None:
        base_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        reset_fn = getattr(base_env, "reset_simulation", None)
        if callable(reset_fn):
            reset_fn()

    def reset(self, **kwargs):
        """Reset the environment."""
        self.controller.reset()
        return self.env.reset(**kwargs)

    def close(self):
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()


class ResetDelayWrapper(gym.Wrapper):
    """
    Wrapper that adds a time delay when resetting the environment.

    This can be useful for adding a pause between episodes to allow for human observation.
    """

    def __init__(self, env, delay_seconds=1.0):
        """
        Initialize the time delay reset wrapper.

        Args:
            env: The environment to wrap
            delay_seconds: The number of seconds to delay during reset
        """
        super().__init__(env)
        self.delay_seconds = delay_seconds

    def reset(self, **kwargs):
        """Reset the environment with a time delay."""
        # Add the time delay
        logging.info(f"Reset delay of {self.delay_seconds} seconds")
        time.sleep(self.delay_seconds)

        # Call the parent reset method
        return self.env.reset(**kwargs)
