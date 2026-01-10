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

from gym_hil.envs.panda_arrange_boxes_gym_env import PandaArrangeBoxesGymEnv
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv

# Try to import ROS2 CR5 environments (env2), but make it optional
try:
    from gym_hil.envs.real_cr5_env2 import RealCR5PickCubeGymEnv
    from gym_hil.envs.cr5_task_env2 import CR5TaskGymEnv
    __all__ = ["PandaPickCubeGymEnv", "PandaArrangeBoxesGymEnv", "RealCR5PickCubeGymEnv", "CR5TaskGymEnv"]
except ImportError:
    # If dependencies are not available, CR5 environments won't be available
    __all__ = ["PandaPickCubeGymEnv", "PandaArrangeBoxesGymEnv"]
