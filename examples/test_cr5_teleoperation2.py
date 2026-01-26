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
# python examples/test_cr5_teleoperation2.py --two-cameras
# python examples/test_cr5_teleoperation2.py --step-size 0.1 越大越快，
''' python examples/test_cr5_teleoperation2.py \
  --step-size 0.01 \
  --infer-script /home/fa/lerobot_ros2/examples/Dr5_ACT/demo_control.py \
  --infer-dataset /path/to/dataset \
  --infer-train-config /path/to/train_config.json \
  --infer-checkpoint /path/to/checkpoint \
  --infer-device cuda \
  --hz 30
'''

import argparse
import signal
import subprocess
import sys
import time

import numpy as np

from gym_hil.envs.cr5_task_env2 import CR5TaskGymEnv
from gym_hil.envs.real_cr5_env2 import RealCR5PickCubeGymEnv
from gym_hil.wrappers.factory import wrap_env
from gym_hil.wrappers.hil_wrappers import InputsControlWrapper


def main():
    parser = argparse.ArgumentParser(description="Control CR5 robot interactively (ROS2)")
    parser.add_argument("--step-size", type=float, default=0.01, help="Step size for movement in meters")
    parser.add_argument(
        "--render-mode", type=str, default="human", choices=["human", "rgb_array"], help="Rendering mode"
    )
    parser.add_argument(
        "--reset-delay",
        type=float,
        default=2.0,
        help="Delay in seconds when resetting the environment (0.0 means no delay)",
    )
    parser.add_argument(
        "--controller-config", type=str, default=None, help="Path to controller configuration JSON file"
    )
    parser.add_argument(
        "--task-env",
        action="store_true",
        help="Use CR5TaskGymEnv instead of RealCR5PickCubeGymEnv",
    )
    parser.add_argument(
        "--two-cameras",
        action="store_true",
        help="Use only head + left wrist cameras (disable right wrist camera)",
    )
    parser.add_argument(
        "--infer-script",
        type=str,
        default="/home/fa/lerobot_ros2/examples/Dr5_ACT/demo_control.py",
        help="Path to inference script (runs as a separate process).",
    )
    parser.add_argument("--infer-dataset", type=str, default=None, help="Pass --dataset to inference script.")
    parser.add_argument("--infer-train-config", type=str, default=None, help="Pass --train-config to inference script.")
    parser.add_argument("--infer-checkpoint", type=str, default=None, help="Pass --checkpoint to inference script.")
    parser.add_argument("--infer-device", type=str, default=None, help="Pass --device to inference script.")
    parser.add_argument("--hz", type=float, default=20.0, help="Control loop frequency.")
    args = parser.parse_args()

    image_obs = False
    if args.two_cameras and not image_obs:
        print("[info] image_obs=False, --two-cameras has no effect.")

    if args.task_env:
        base_env = CR5TaskGymEnv(
            render_mode=args.render_mode,
            image_obs=image_obs,
            use_right_wrist_camera=not args.two_cameras if image_obs else False,
        )
    else:
        base_env = RealCR5PickCubeGymEnv(
            render_mode=args.render_mode,
            image_obs=image_obs,
            use_right_wrist_camera=not args.two_cameras if image_obs else False,
        )

    # Print observation space for debugging
    print("Observation space:", base_env.observation_space)

    # Reset and check observation structure
    obs, _ = base_env.reset()
    print("Observation keys:", list(obs.keys()))
    if "pixels" in obs:
        print("Pixels keys:", list(obs["pixels"].keys()))

    # Wrap with HIL/gamepad control
    print("\nWrapping environment with gamepad control...")
    ee_step_size = {
        "x": args.step_size,
        "y": args.step_size,
        "z": args.step_size,
        "r_x": args.step_size,
        "r_y": args.step_size,
        "r_z": args.step_size,
    }
    env = wrap_env(
        base_env,
        ee_step_size=ee_step_size,
        use_gamepad=True,
        reset_delay_seconds=args.reset_delay,
        controller_config_path=args.controller_config,
    )

    # Reset environment
    obs, _ = env.reset()

    # Find the InputsControlWrapper to query intervention state without stepping
    control_wrapper = None
    _cur = env
    while hasattr(_cur, "env"):
        if isinstance(_cur, InputsControlWrapper):
            control_wrapper = _cur
            break
        _cur = _cur.env
    if control_wrapper is None and isinstance(_cur, InputsControlWrapper):
        control_wrapper = _cur
    if control_wrapper is None:
        raise RuntimeError("InputsControlWrapper not found. Cannot detect intervention state.")

    action_dim = env.action_space.shape[0]
    dummy_action = np.zeros(action_dim, dtype=np.float32)
    if action_dim >= 14:
        dummy_action[6] = 1.0
        dummy_action[13] = 1.0
    elif action_dim >= 7:
        dummy_action[-1] = 1.0  # gripper keep position for EEActionWrapper

    infer_cmd = [sys.executable, args.infer_script]
    if args.infer_dataset:
        infer_cmd += ["--dataset", args.infer_dataset]
    if args.infer_train_config:
        infer_cmd += ["--train-config", args.infer_train_config]
    if args.infer_checkpoint:
        infer_cmd += ["--checkpoint", args.infer_checkpoint]
    if args.infer_device:
        infer_cmd += ["--device", args.infer_device]

    infer_proc = subprocess.Popen(infer_cmd)
    infer_paused = False

    try:
        while True:
            if infer_proc.poll() is not None:
                raise RuntimeError("Inference process exited unexpectedly.")

            control_wrapper.controller.update()
            intervention_active = control_wrapper.controller.should_intervene()

            if intervention_active:
                if not infer_paused:
                    infer_proc.send_signal(signal.SIGSTOP)
                    infer_paused = True

                # Step the environment only when intervening
                obs, reward, terminated, truncated, info = env.step(dummy_action)

                # Print some feedback
                if info.get("succeed", False):
                    print("\nSuccess! Block has been picked up.")

                if info.get("is_intervention", False):
                    action_intervention = info.get("action_intervention")
                    if isinstance(action_intervention, np.ndarray):
                        if action_dim >= 14:
                            dummy_action[6] = action_intervention[6]
                            dummy_action[13] = action_intervention[13]
                        elif action_dim >= 7:
                            dummy_action[-1] = action_intervention[-1]

                # If auto-reset is disabled, manually reset when episode ends
                if terminated or truncated:
                    print("Episode ended, resetting environment")
                    obs, _ = env.reset()
            else:
                if infer_paused:
                    infer_proc.send_signal(signal.SIGCONT)
                    infer_paused = False

            # Add a small delay to control update rate
            time.sleep(1.0 / max(args.hz, 1e-6))

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if infer_proc.poll() is None:
            infer_proc.terminate()
            infer_proc.wait(timeout=5)
        env.close()
        print("Session ended")


if __name__ == "__main__":
    main()
