from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 尝试导入 ROS2 机器人相关模块
try:
    from lerobot_camera_ros2 import ROS2CameraConfig
    from lerobot_robot_ros2 import (
        ControlType,
        ROS2Robot,
        ROS2RobotConfig,
        ROS2RobotInterfaceConfig,
    )
    ROS2_AVAILABLE = True
except ImportError as exc:
    ROS2_AVAILABLE = False
    logging.warning("ROS2 模块不可用: %s. RealCR5PickCubeGymEnv 将无法工作。", exc)



# 常量定义
_CR5_ROBOT_ID = "Tianji_stand1_robot"
_ROS2_JOINT_STATES_TOPIC = "/joint_states"
_ROS2_END_EFFECTOR_POSE_TOPIC = "/left_current_pose"
_ROS2_END_EFFECTOR_TARGET_TOPIC = "/left_target"
_ROS2_RIGHT_END_EFFECTOR_POSE_TOPIC = "/right_current_pose"
_ROS2_RIGHT_END_EFFECTOR_TARGET_TOPIC = "/right_target"
_CR5_LEFT_JOINT_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7"
]
_CR5_RIGHT_JOINT_NAMES = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7"
]

_CR5_GRIPPER_ENABLED = True
_CR5_GRIPPER_JOINT_NAME = "left_gripper_joint"
_CR5_GRIPPER_COMMAND_TOPIC = "left_gripper_joint/position_command"
_CR5_RIGHT_GRIPPER_JOINT_NAME = "right_gripper_joint"
_CR5_RIGHT_GRIPPER_COMMAND_TOPIC = "right_gripper_joint/position_command"
_CR5_GRIPPER_MIN_POSITION = 0.0
_CR5_GRIPPER_MAX_POSITION = 1.0
_CR5_MAX_LINEAR_VELOCITY = 0.1
_CR5_MAX_ANGULAR_VELOCITY = 0.5
_CR5_JOINT_STATE_TIMEOUT = 0.0
_CR5_END_EFFECTOR_POSE_TIMEOUT = 0.0
_CAMERA_FPS = 30
_FRONT_CAMERA_NAME = "head_camera"
_FRONT_CAMERA_TOPIC = "/head_camera/rgb"
_FRONT_CAMERA_NODE_NAME = "lerobot_head_camera"
_WRIST_LEFT_CAMERA_NAME = "left_wrist_camera"
_WRIST_LEFT_CAMERA_TOPIC = "/left_wrist_camera/rgb"
_WRIST_LEFT_CAMERA_NODE_NAME = "lerobot_wrist_left_camera"
_WRIST_RIGHT_CAMERA_NAME = "right_wrist_camera"
_WRIST_RIGHT_CAMERA_TOPIC = "/right_wrist_camera/rgb"
_WRIST_RIGHT_CAMERA_NODE_NAME = "lerobot_wrist_right_camera"
_CAMERA_ENCODING = "bgr8"


class RealRobotGymEnv(gym.Env, ABC):
    """真实机器人环境的基类，提供与 MuJoCo 环境相同的接口但不依赖 MuJoCo。"""

    def __init__(
            self,
            seed: int = 0,
            control_dt: float = 0.02,
            render_mode: Literal["rgb_array", "human"] = "rgb_array",
            image_obs: bool = False,
            reward_type: str = "sparse",
            random_block_position: bool = False,
            image_height: int = 720,
            image_width: int = 1280,
    ):
        super().__init__()

        self.reward_type = reward_type
        self.control_dt = control_dt
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.random_block_position = random_block_position
        self.image_height = image_height
        self.image_width = image_width

        # 初始化随机数生成器
        self.np_random = np.random.RandomState(seed)

        # 任务相关设置
        self._block_z = 0.025  # 方块高度的一半
        self._z_init = None
        self._z_success = None

        # 机器人状态缓存
        self._current_robot_state = None
        self._current_block_position = None

        # 设置观察空间
        self._setup_observation_space()

        # 设置动作空间
        self._setup_action_space()

        # 元数据
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

    def _setup_observation_space(self):
        """设置观察空间"""
        # 机器人状态维度：关节位置(7) + 关节速度(7) + 夹爪位置(1) + TCP位置(3) = 18
        agent_dim = 18
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        if self.image_obs:
            camera_spaces = {
                key: spaces.Box(
                    0,
                    255,
                    (self.image_height, self.image_width, 3),
                    dtype=np.uint8,
                )
                for key in self._camera_keys()
            }
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(camera_spaces),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def _setup_action_space(self):
        """设置动作空间 - 增量控制模式"""
        # 动作空间：位置增量(x, y, z) + 姿态增量(rx, ry, rz) + 夹爪命令(grasp_command)
        # 所有值都是归一化的 [-1, 1]，表示相对当前位置/姿态的增量
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _camera_keys(self) -> Tuple[str, ...]:
        """Return the camera keys expected in image observations."""
        return ("front", "wrist_left", "wrist_right")

    @abstractmethod
    def _get_camera_images(self) -> Tuple[np.ndarray, ...]:
        """
        获取相机图像的虚函数，需要子类实现

        Returns:
            Tuple[np.ndarray, ...]: (front_view, wrist_left[, wrist_right])
        """
        pass

    @abstractmethod
    def _send_control_command(self, action: np.ndarray) -> None:
        """
        发送控制指令的虚函数，需要子类实现

        Args:
            action: 控制动作 [x, y, z, rx, ry, rz, grasp_command]
        """
        pass

    @abstractmethod
    def _get_robot_state(self) -> np.ndarray:
        """
        获取机器人状态的虚函数，需要子类实现

        Returns:
            np.ndarray: 机器人状态向量
        """
        pass

    @abstractmethod
    def _get_block_position(self) -> np.ndarray:
        """
        获取方块位置的虚函数，需要子类实现

        Returns:
            np.ndarray: 方块位置 [x, y, z]
        """
        pass

    @abstractmethod
    def _reset_robot_to_home(self) -> None:
        """
        将机器人重置到初始位置的虚函数，需要子类实现
        """
        pass

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # 重置机器人到初始位置
        self._reset_robot_to_home()

        # 获取初始观察
        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """执行一步"""
        # 发送控制指令到真实机器人
        self._send_control_command(action)

        # 等待控制周期
        time.sleep(self.control_dt)

        # 计算观察、奖励和终止条件
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = rew == 1.0

        terminated = bool(success)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> Dict[str, np.ndarray]:
        """计算当前观察"""
        observation = {}

        # 获取机器人状态
        robot_state = self._get_robot_state().astype(np.float32)

        if self.image_obs:
            # 图像观察
            camera_views = self._get_camera_images()
            camera_keys = self._camera_keys()
            if len(camera_views) != len(camera_keys):
                raise ValueError(
                    f"Camera views length ({len(camera_views)}) does not match keys ({len(camera_keys)})"
                )
            observation = {
                "pixels": dict(zip(camera_keys, camera_views)),
                "agent_pos": robot_state,
            }
        else:
            # 仅状态观察
            block_pos = self._get_block_position().astype(np.float32)
            observation = {
                "agent_pos": robot_state,
                "environment_state": block_pos,
            }

        return observation

    def _compute_reward(self) -> float:
        """计算奖励"""
        block_pos = self._get_block_position()

        return 0.0  # 临时返回值

    def _is_success(self) -> bool:
        """检查任务是否成功完成"""

        return False  # 临时简化版本

    def render(self):
        """渲染环境"""
        if self.render_mode == "rgb_array":
            if self.image_obs:
                camera_views = self._get_camera_images()
                camera_keys = self._camera_keys()
                if len(camera_views) != len(camera_keys):
                    raise ValueError(
                        f"Camera views length ({len(camera_views)}) does not match keys ({len(camera_keys)})"
                    )
                return dict(zip(camera_keys, camera_views))
            else:
                # 如果没有图像观察，返回空字典或状态信息
                return {}
        elif self.render_mode == "human":
            # 在人类模式下，可以显示图像或状态信息
            pass

    def close(self) -> None:
        """关闭环境，释放资源"""
        # 子类可以重写此方法来清理资源
        pass

class RealCR5PickCubeGymEnv(RealRobotGymEnv):
    """基于 ROS2 机器人配置的真实 CR5 机器人抓取环境"""

    def __init__(
            self,
            seed: int = 0,
            control_dt: float = 0.1,
            render_mode: Literal["rgb_array", "human"] = "rgb_array",
            image_obs: bool = False,
            reward_type: str = "sparse",
            random_block_position: bool = False,
            image_height: int = 720,
            image_width: int = 1280,
            use_right_wrist_camera: Optional[bool] = None,
            ros2_config: Optional[ROS2RobotConfig] = None,
    ):
        if use_right_wrist_camera is None:
            use_right_wrist_camera = bool(_WRIST_RIGHT_CAMERA_TOPIC)
        self._use_right_wrist_camera = use_right_wrist_camera
        self._left_joint_names = list(_CR5_LEFT_JOINT_NAMES)
        self._right_joint_names = list(_CR5_RIGHT_JOINT_NAMES)
        self._dual_arm_enabled = self._is_dual_arm_config()

        # 调用父类初始化
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            render_mode=render_mode,
            image_obs=image_obs,
            reward_type=reward_type,
            random_block_position=random_block_position,
            image_height=image_height,
            image_width=image_width,
        )

        # 检查 ROS2 模块是否可用
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 模块不可用，请安装 lerobot_robot_ros2。")

        # 设置默认 ROS2 配置
        if ros2_config is None:
            ros2_config = self._create_default_cr5_config()

        self.robot_config = ros2_config
        self._refresh_dual_arm_config()
        self.robot = None  # 将在 super().__init__ 后初始化
        
        # 缓存上一次的控制指令
        self._last_position_command = None
        self._last_orientation_command = None
        self._last_position_command_right = None
        self._last_orientation_command_right = None

        # 初始化机器人连接
        self._initialize_robot()

    def _is_dual_arm_config(self, ros2_interface: Optional[ROS2RobotInterfaceConfig] = None) -> bool:
        interface = ros2_interface
        if interface is None and hasattr(self, "robot_config"):
            interface = self.robot_config.ros2_interface
        if interface is None:
            return bool(self._right_joint_names)
        return bool(
            self._right_joint_names
            or interface.right_end_effector_pose_topic
            or interface.right_end_effector_target_topic
            or interface.right_gripper_command_topic
        )

    def _refresh_dual_arm_config(self) -> None:
        dual_arm = self._is_dual_arm_config(self.robot_config.ros2_interface)
        if dual_arm != self._dual_arm_enabled:
            self._dual_arm_enabled = dual_arm
            self._setup_action_space()
            self._setup_observation_space()

    def _setup_action_space(self):
        """设置动作空间 - 适配单臂/双臂"""
        base_dim = 7
        if self._dual_arm_enabled:
            base_dim *= 2
        self.action_space = spaces.Box(
            low=np.full((base_dim,), -1.0, dtype=np.float32),
            high=np.full((base_dim,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def _camera_keys(self) -> Tuple[str, ...]:
        keys = ("front", "wrist_left")
        if self._use_right_wrist_camera:
            return keys + ("wrist_right",)
        return keys

    def _create_default_cr5_config(self) -> ROS2RobotConfig:
        """创建默认的 CR5 机器人配置"""
        camera_cfg = {}
        if self.image_obs:
            camera_cfg = {
                _FRONT_CAMERA_NAME: ROS2CameraConfig(
                    topic_name=_FRONT_CAMERA_TOPIC,
                    node_name=_FRONT_CAMERA_NODE_NAME,
                    width=self.image_width,
                    height=self.image_height,
                    fps=_CAMERA_FPS,
                    encoding=_CAMERA_ENCODING,
                ),
            }
            if _WRIST_LEFT_CAMERA_TOPIC:
                camera_cfg[_WRIST_LEFT_CAMERA_NAME] = ROS2CameraConfig(
                    topic_name=_WRIST_LEFT_CAMERA_TOPIC,
                    node_name=_WRIST_LEFT_CAMERA_NODE_NAME,
                    width=self.image_width,
                    height=self.image_height,
                    fps=_CAMERA_FPS,
                    encoding=_CAMERA_ENCODING,
                )
            if self._use_right_wrist_camera and _WRIST_RIGHT_CAMERA_TOPIC:
                camera_cfg[_WRIST_RIGHT_CAMERA_NAME] = ROS2CameraConfig(
                    topic_name=_WRIST_RIGHT_CAMERA_TOPIC,
                    node_name=_WRIST_RIGHT_CAMERA_NODE_NAME,
                    width=self.image_width,
                    height=self.image_height,
                    fps=_CAMERA_FPS,
                    encoding=_CAMERA_ENCODING,
                )
        return ROS2RobotConfig(
            id=_CR5_ROBOT_ID,
            ros2_interface=ROS2RobotInterfaceConfig(
                joint_states_topic=_ROS2_JOINT_STATES_TOPIC,
                end_effector_pose_topic=_ROS2_END_EFFECTOR_POSE_TOPIC,
                end_effector_target_topic=_ROS2_END_EFFECTOR_TARGET_TOPIC,
                right_end_effector_pose_topic=_ROS2_RIGHT_END_EFFECTOR_POSE_TOPIC or None,
                right_end_effector_target_topic=_ROS2_RIGHT_END_EFFECTOR_TARGET_TOPIC or None,
                right_gripper_command_topic=_CR5_RIGHT_GRIPPER_COMMAND_TOPIC or None,
                control_type=ControlType.CARTESIAN_POSE,
                joint_names=_CR5_LEFT_JOINT_NAMES + _CR5_RIGHT_JOINT_NAMES,
                gripper_enabled=_CR5_GRIPPER_ENABLED,
                gripper_joint_name=_CR5_GRIPPER_JOINT_NAME,
                gripper_command_topic=_CR5_GRIPPER_COMMAND_TOPIC,
                gripper_min_position=_CR5_GRIPPER_MIN_POSITION,
                gripper_max_position=_CR5_GRIPPER_MAX_POSITION,
                max_linear_velocity=_CR5_MAX_LINEAR_VELOCITY,
                max_angular_velocity=_CR5_MAX_ANGULAR_VELOCITY,
                joint_state_timeout=_CR5_JOINT_STATE_TIMEOUT,
                end_effector_pose_timeout=_CR5_END_EFFECTOR_POSE_TIMEOUT,
            ),
            cameras=camera_cfg,
        )

    def _initialize_robot(self):
        """初始化机器人连接"""
        try:
            self.robot = ROS2Robot(self.robot_config)
            self.robot.connect()
            logging.info("CR5 robot connected successfully")
        except Exception as e:
            logging.error(f"Failed to connect to CR5 robot: {e}")
            raise

    def _setup_observation_space(self):
        """设置观察空间 - 适配 CR5 机器人"""
        left_joint_count = len(self._left_joint_names)
        right_joint_count = len(self._right_joint_names) if self._dual_arm_enabled else 0
        right_pose_topic = None
        if hasattr(self, "robot_config"):
            right_pose_topic = self.robot_config.ros2_interface.right_end_effector_pose_topic
        right_pose_enabled = self._dual_arm_enabled and bool(right_pose_topic or _ROS2_RIGHT_END_EFFECTOR_POSE_TOPIC)
        gripper_count = 1 + (1 if self._dual_arm_enabled and _CR5_RIGHT_GRIPPER_JOINT_NAME else 0)
        tcp_count = 3 + (3 if right_pose_enabled else 0)
        agent_dim = (left_joint_count + right_joint_count) * 2 + tcp_count + gripper_count
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        if self.image_obs:
            # 图像观察（暂时使用占位符）
            camera_spaces = {
                key: spaces.Box(
                    0,
                    255,
                    (self.image_height, self.image_width, 3),
                    dtype=np.uint8,
                )
                for key in self._camera_keys()
            }
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(camera_spaces),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def get_gripper_pose(self):
        obs = self.robot.get_observation()

        gripper_pos = obs[f"{self.robot.config.ros2_interface.gripper_joint_name}.pos"]

        return gripper_pos

    def _get_robot_state(self) -> np.ndarray:
        """从 ROS2 机器人获取机器人状态"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")

        try:
            obs = self.robot.get_observation()

            def _get_joint_values(joint_names, suffix):
                values = []
                for joint_name in joint_names:
                    values.append(obs.get(f"{joint_name}.{suffix}", 0.0))
                return values

            # 提取关节状态
            left_joint_positions = _get_joint_values(self._left_joint_names, "pos")
            left_joint_velocities = _get_joint_values(self._left_joint_names, "vel")
            right_joint_positions = _get_joint_values(self._right_joint_names, "pos") if self._dual_arm_enabled else []
            right_joint_velocities = _get_joint_values(self._right_joint_names, "vel") if self._dual_arm_enabled else []

            # 提取夹爪状态
            gripper_pos = obs.get(f"{self.robot.config.ros2_interface.gripper_joint_name}.pos", 0.0)
            right_gripper_pos = 0.0
            if self._dual_arm_enabled and _CR5_RIGHT_GRIPPER_JOINT_NAME:
                right_gripper_pos = obs.get(f"{_CR5_RIGHT_GRIPPER_JOINT_NAME}.pos", 0.0)

            # 提取 TCP 位置
            tcp_pos = np.array(
                [
                    obs.get("end_effector.position.x", 0.0),
                    obs.get("end_effector.position.y", 0.0),
                    obs.get("end_effector.position.z", 0.0),
                ]
            )
            right_tcp_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if self._dual_arm_enabled and self.robot.config.ros2_interface.right_end_effector_pose_topic:
                right_pose = self.robot.ros2_interface.get_right_end_effector_pose()
                if right_pose is not None:
                    right_tcp_pos = np.array(
                        [
                            right_pose.position.x,
                            right_pose.position.y,
                            right_pose.position.z,
                        ]
                    )

            state_parts = [
                left_joint_positions,
                right_joint_positions,
                left_joint_velocities,
                right_joint_velocities,
                tcp_pos.tolist(),
            ]
            if self._dual_arm_enabled and self.robot.config.ros2_interface.right_end_effector_pose_topic:
                state_parts.append(right_tcp_pos.tolist())
            state_parts.append([gripper_pos])
            if self._dual_arm_enabled and _CR5_RIGHT_GRIPPER_JOINT_NAME:
                state_parts.append([right_gripper_pos])

            return np.concatenate([np.asarray(part, dtype=np.float32) for part in state_parts])

        except Exception as e:
            logging.error(f"Failed to get robot state: {e}")
            # 返回零状态作为后备
            agent_dim = self.observation_space["agent_pos"].shape[0]
            return np.zeros(agent_dim, dtype=np.float32)

    def _send_control_command(self, action: np.ndarray) -> None:
        """通过 ROS2 机器人发送控制指令 - 增量控制模式"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")

        action = np.asarray(action, dtype=np.float32)
        left_action = action[:7]
        right_action = action[7:14] if self._dual_arm_enabled and action.size >= 14 else None

        def quat_mult_ros(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            return np.array(
                [
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ]
            )

        def compute_target(action_vec, current_pos, current_ori, last_pos, last_ori):
            position_scale = 0.1
            orientation_scale = 0.5

            position_action = action_vec[:3]
            if np.allclose(position_action, 0, atol=1e-6):
                if last_pos is not None:
                    target_pos = last_pos.copy()
                else:
                    target_pos = current_pos.copy()
                    last_pos = target_pos.copy()
            else:
                if last_pos is not None:
                    target_pos = last_pos + position_action * position_scale
                else:
                    target_pos = current_pos + position_action * position_scale
                last_pos = target_pos.copy()

            orientation_action = action_vec[3:6]
            if np.allclose(orientation_action, 0, atol=1e-6):
                if last_ori is not None:
                    target_ori = last_ori.copy()
                else:
                    target_ori = current_ori.copy()
                    last_ori = target_ori.copy()
            else:
                rx, ry, rz = orientation_action * orientation_scale
                half_angles = np.array([rx, ry, rz]) / 2.0
                cx, cy, cz = np.cos(half_angles)
                sx, sy, sz = np.sin(half_angles)

                q_inc_x = np.array([sx, 0, 0, cx])
                q_inc_y = np.array([0, sy, 0, cy])
                q_inc_z = np.array([0, 0, sz, cz])

                q_inc = quat_mult_ros(quat_mult_ros(q_inc_z, q_inc_y), q_inc_x)
                base_ori = last_ori if last_ori is not None else current_ori
                target_ori = quat_mult_ros(q_inc, base_ori)
                target_ori = target_ori / np.linalg.norm(target_ori)
                last_ori = target_ori.copy()

            gripper_command = action_vec[6]
            gripper_position = (gripper_command + 1.0) / 2.0
            return target_pos, target_ori, last_pos, last_ori, gripper_position

        current_obs = self.robot.get_observation()
        current_pos = np.array(
            [
                current_obs["end_effector.position.x"],
                current_obs["end_effector.position.y"],
                current_obs["end_effector.position.z"],
            ]
        )
        current_ori = np.array(
            [
                current_obs["end_effector.orientation.x"],
                current_obs["end_effector.orientation.y"],
                current_obs["end_effector.orientation.z"],
                current_obs["end_effector.orientation.w"],
            ]
        )
        (
            target_pos,
            target_ori,
            self._last_position_command,
            self._last_orientation_command,
            gripper_position,
        ) = compute_target(
            left_action,
            current_pos,
            current_ori,
            self._last_position_command,
            self._last_orientation_command,
        )

        target_action = {
            "end_effector.position.x": float(target_pos[0]),
            "end_effector.position.y": float(target_pos[1]),
            "end_effector.position.z": float(target_pos[2]),
            "end_effector.orientation.x": float(target_ori[0]),
            "end_effector.orientation.y": float(target_ori[1]),
            "end_effector.orientation.z": float(target_ori[2]),
            "end_effector.orientation.w": float(target_ori[3]),
            "gripper.position": float(gripper_position),
        }
        self.robot.send_action(target_action)

        if right_action is not None and self.robot.config.ros2_interface.right_end_effector_target_topic:
            right_pose = self.robot.ros2_interface.get_right_end_effector_pose()
            if right_pose is None:
                logging.warning("Right end-effector pose unavailable; skipping right arm command.")
                return
            right_pos = np.array([right_pose.position.x, right_pose.position.y, right_pose.position.z])
            right_ori = np.array(
                [
                    right_pose.orientation.x,
                    right_pose.orientation.y,
                    right_pose.orientation.z,
                    right_pose.orientation.w,
                ]
            )
            (
                target_right_pos,
                target_right_ori,
                self._last_position_command_right,
                self._last_orientation_command_right,
                right_gripper_position,
            ) = compute_target(
                right_action,
                right_pos,
                right_ori,
                self._last_position_command_right,
                self._last_orientation_command_right,
            )

            from geometry_msgs.msg import Pose

            right_target_pose = Pose()
            right_target_pose.position.x = float(target_right_pos[0])
            right_target_pose.position.y = float(target_right_pos[1])
            right_target_pose.position.z = float(target_right_pos[2])
            right_target_pose.orientation.x = float(target_right_ori[0])
            right_target_pose.orientation.y = float(target_right_ori[1])
            right_target_pose.orientation.z = float(target_right_ori[2])
            right_target_pose.orientation.w = float(target_right_ori[3])
            self.robot.ros2_interface.send_right_end_effector_target(right_target_pose)

            if self.robot.config.ros2_interface.right_gripper_command_topic:
                self.robot.ros2_interface.send_right_gripper_command(right_gripper_position)

    def _get_camera_images(self) -> Tuple[np.ndarray, ...]:
        """获取相机图像并调整到指定尺寸"""
        import cv2

        try:
            if self.robot is None:
                raise RuntimeError("Robot not initialized")

            obs = self.robot.get_observation()
            front_view = obs.get(f"{_FRONT_CAMERA_NAME}.rgb")
            if front_view is None:
                front_view = obs.get(_FRONT_CAMERA_NAME)
            wrist_left_view = obs.get(f"{_WRIST_LEFT_CAMERA_NAME}.rgb")
            if wrist_left_view is None:
                wrist_left_view = obs.get(_WRIST_LEFT_CAMERA_NAME)
            wrist_right_view = None
            if self._use_right_wrist_camera:
                wrist_right_view = obs.get(f"{_WRIST_RIGHT_CAMERA_NAME}.rgb")
                if wrist_right_view is None:
                    wrist_right_view = obs.get(_WRIST_RIGHT_CAMERA_NAME)

            # 将图像调整到128x128尺寸
            if front_view is not None and front_view.shape[:2] != (self.image_height, self.image_width):
                front_view = cv2.resize(front_view, (self.image_width, self.image_height))

            if wrist_left_view is not None and wrist_left_view.shape[:2] != (self.image_height, self.image_width):
                wrist_left_view = cv2.resize(wrist_left_view, (self.image_width, self.image_height))
            if self._use_right_wrist_camera and wrist_right_view is not None:
                if wrist_right_view.shape[:2] != (self.image_height, self.image_width):
                    wrist_right_view = cv2.resize(wrist_right_view, (self.image_width, self.image_height))

            # 如果相机返回None，创建占位符图像
            if front_view is None:
                front_view = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            if wrist_left_view is None:
                wrist_left_view = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            if self._use_right_wrist_camera:
                if wrist_right_view is None:
                    wrist_right_view = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                return front_view, wrist_left_view, wrist_right_view
            return front_view, wrist_left_view
        except Exception as e:
            logging.error(f"Failed to get camera images: {e}")
            placeholder = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            if self._use_right_wrist_camera:
                return placeholder, placeholder.copy(), placeholder.copy()
            return placeholder, placeholder.copy()

    def _get_block_position(self) -> np.ndarray:
        """获取方块位置 - 暂时返回固定位置"""
        # TODO: 实现计算机视觉检测方块位置
        # 暂时返回固定位置
        return np.array([0.5, 0.0, self._block_z])

    def _reset_robot_to_home(self) -> None:
        """重置机器人到当前状态，避免强制回到固定 home 位姿"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")

        try:
            obs = self.robot.get_observation()
            self._last_position_command = np.array(
                [
                    obs.get("end_effector.position.x", 0.0),
                    obs.get("end_effector.position.y", 0.0),
                    obs.get("end_effector.position.z", 0.0),
                ],
                dtype=np.float32,
            )
            self._last_orientation_command = np.array(
                [
                    obs.get("end_effector.orientation.x", 0.0),
                    obs.get("end_effector.orientation.y", 0.0),
                    obs.get("end_effector.orientation.z", 0.0),
                    obs.get("end_effector.orientation.w", 1.0),
                ],
                dtype=np.float32,
            )

            if self._dual_arm_enabled and self.robot.config.ros2_interface.right_end_effector_pose_topic:
                right_pose = self.robot.ros2_interface.get_right_end_effector_pose()
                if right_pose is not None:
                    self._last_position_command_right = np.array(
                        [
                            right_pose.position.x,
                            right_pose.position.y,
                            right_pose.position.z,
                        ],
                        dtype=np.float32,
                    )
                    self._last_orientation_command_right = np.array(
                        [
                            right_pose.orientation.x,
                            right_pose.orientation.y,
                            right_pose.orientation.z,
                            right_pose.orientation.w,
                        ],
                        dtype=np.float32,
                    )

        except Exception as e:
            logging.error(f"Failed to reset robot to home: {e}")
            raise

    def _get_tcp_position(self) -> np.ndarray:
        """获取TCP位置 - 用于计算奖励和成功条件"""
        if self.robot is None:
            raise RuntimeError("Robot not initialized")

        try:
            obs = self.robot.get_observation()
            return np.array([
                obs["end_effector.position.x"],
                obs["end_effector.position.y"],
                obs["end_effector.position.z"]
            ])
        except Exception as e:
            logging.error(f"Failed to get TCP position: {e}")
            return np.zeros(3)

    def close(self) -> None:
        """关闭环境，释放资源"""
        if self.robot is not None:
            try:
                self.robot.disconnect()
                logging.info("CR5 robot disconnected")
            except Exception as e:
                logging.error(f"Error disconnecting robot: {e}")
            finally:
                self.robot = None
        super().close()
