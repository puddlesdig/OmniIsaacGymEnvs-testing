# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# (license text unchanged)

import numpy as np
import torch

from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *  # quat_axis, etc.

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.crazyflie import Crazyflie
from omniisaacgymenvs.robots.articulations.views.crazyflie_view import CrazyflieView

EPS = 1e-6  # small constant to avoid divisions by zero

def torch_rand_float(low, high, shape, device):
    """Uniform random tensor in [low, high] with the given shape on device."""
    return (high - low) * torch.rand(shape, device=device, dtype=torch.float32) + low

class CrazyflieTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        # Obs: [pos_err(3), body axes (9), lin vel (3), ang vel(3)] = 18
        self._num_observations = 18
        # Actions: 4 motor thrust fractions
        self._num_actions = 4

        # Spawn close to the floor so the agent must generate thrust to lift.
        self._crazyflie_position = torch.tensor([0.0, 0.0, 0.20])
        # Visual target at 1.0 m above the takeoff point.
        self._ball_position = torch.tensor([0.0, 0.0, 1.0])

        super().__init__(name=name, env=env)

    # --------------------------------------------------------------------- #
    # Config
    # --------------------------------------------------------------------- #
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]  # matches YAML (e.g., 1000)
        self.dt = self._task_cfg["sim"]["dt"]  # 0.01 → 100 Hz with the stock YAML

        # Crazyflie physical-ish parameters (simplified)
        self.arm_length = 0.05
        self.mass = 0.028  # kg
        self.thrust_to_weight = 3.0  # scale for max thrust relative to weight

        # Motor first-order lag (s)
        self.motor_damp_time_up = 0.05
        self.motor_damp_time_down = 0.05

        # Convert to per-step blend factors (0..1), approx 4 time constants to settle
        self.motor_tau_up = 4.0 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4.0 * self.dt / (self.motor_damp_time_down + EPS)

        # Nominal motor asymmetry (normalized to sum to 4)
        self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.motor_assymetry = self.motor_assymetry * 4.0 / np.sum(self.motor_assymetry)

        # Gravity magnitude from YAML (gravity is [0,0,-9.81])
        self.grav_z = -1.0 * float(self._task_cfg["sim"]["gravity"][2])  # → 9.81

    # --------------------------------------------------------------------- #
    # Scene setup
    # --------------------------------------------------------------------- #
    def set_up_scene(self, scene) -> None:
        self.get_crazyflie()
        self.get_target()
        super().set_up_scene(scene)

        self._copters = CrazyflieView(prim_paths_expr="/World/envs/.*/Crazyflie", name="crazyflie_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view")
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(4):
            scene.add(self._copters.physics_rotors[i])

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("crazyflie_view"):
            scene.remove_object("crazyflie_view", registry_only=True)
        if scene.object_exists("ball_view"):
            scene.remove_object("ball_view", registry_only=True)
        for i in range(1, 5):
            scene.remove_object(f"m{i}_prop_view", registry_only=True)

        self._copters = CrazyflieView(prim_paths_expr="/World/envs/.*/Crazyflie", name="crazyflie_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="ball_view")
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(4):
            scene.add(self._copters.physics_rotors[i])

    def get_crazyflie(self):
        copter = Crazyflie(
            prim_path=self.default_zero_env_path + "/Crazyflie",
            name="crazyflie",
            translation=self._crazyflie_position,
        )
        self._sim_config.apply_articulation_settings(
            "crazyflie",
            get_prim_at_path(copter.prim_path),
            self._sim_config.parse_actor_config("crazyflie"),
        )

    def get_target(self):
        radius = 0.2
        color = torch.tensor([1.0, 0.0, 0.0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            translation=self._ball_position,
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings(
            "ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball")
        )
        ball.set_collision_enabled(False)

    # --------------------------------------------------------------------- #
    # Observations
    # --------------------------------------------------------------------- #
    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        self.root_velocities = self._copters.get_velocities(clone=False)

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot

        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)

        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]

        # 0:3 position error to hover target
        self.obs_buf[..., 0:3] = self.target_positions - root_positions
        # 3:12 body orientation basis
        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z
        # 12:18 velocities
        self.obs_buf[..., 12:15] = root_linvels
        self.obs_buf[..., 15:18] = root_angvels

        return {self._copters.name: {"obs_buf": self.obs_buf}}

    # --------------------------------------------------------------------- #
    # Action application
    # --------------------------------------------------------------------- #
    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Optionally update targets every N steps (here target stays fixed)
        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)
        self.actions = actions

        # Clamp to [-1, 1] then map to [0, 1] as thrust fraction
        thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
        thrust_cmds = (thrust_cmds + 1.0) / 2.0

        # First-order motor lag (per-step blending)
        motor_tau = self.motor_tau_up * torch.ones((self._num_envs, 4), dtype=torch.float32, device=self._device)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau = torch.clamp(motor_tau, 0.0, 1.0)

        # Apply lag in sqrt space (roughly linearizes thrust → rotor speed)
        thrust_rot = torch.sqrt(thrust_cmds)
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp ** 2

        # Per-env thrust noise (shape: [num_envs, 4])
        thrust_noise = 0.01 * torch.randn(self._num_envs, 4, dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.clamp(self.thrust_cmds_damp + thrust_cmds * thrust_noise, 0.0, 1.0)

        # Scale to Newtons per motor
        thrusts = self.thrust_max * self.thrust_cmds_damp  # [E, 4]

        # Rotate thrust vectors into world frame and write to rotor force buffers
        root_quats = self.root_rot
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        rot_matrix = torch.cat((rot_x, rot_y, rot_z), dim=1).reshape(-1, 3, 3)

        # Build [Fx, Fy, Fz] for each rotor; here thrust is along body +Z
        force_xy = torch.zeros(self._num_envs, 4, 2, dtype=torch.float32, device=self._device)
        thrusts_3d = torch.cat((force_xy, thrusts.view(-1, 4, 1)), dim=2)  # [E,4,3]

        # Rotate each rotor's thrust into world frame
        for i in range(4):
            mod_thrust_i = torch.matmul(rot_matrix, thrusts_3d[:, i].unsqueeze(-1)).squeeze(-1)  # [E,3]
            self.thrusts[:, i] = mod_thrust_i

        # Clear forces for freshly reset envs
        if len(reset_env_ids) > 0:
            self.thrusts[reset_env_ids] = 0.0

        # Spin rotor visuals (sign flips for counter-rotating props)
        prop_rot = self.thrust_cmds_damp * self.prop_max_rot
        self.dof_vel[:, 0] = prop_rot[:, 0]
        self.dof_vel[:, 1] = -prop_rot[:, 1]
        self.dof_vel[:, 2] = prop_rot[:, 2]
        self.dof_vel[:, 3] = -prop_rot[:, 3]
        self._copters.set_joint_velocities(self.dof_vel)

        # Apply forces
        for i in range(4):
            self._copters.physics_rotors[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)

    # --------------------------------------------------------------------- #
    # Reset & init
    # --------------------------------------------------------------------- #
    def post_reset(self):
        # Per-motor max thrust (N), normalized by asymmetry and 1/4 split
        thrust_max = self.grav_z * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0

        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_max = torch.tensor(thrust_max, device=self._device, dtype=torch.float32)

        self.motor_linearity = 1.0
        self.prop_max_rot = 433.3  # visual rotor RPM-ish scalar

        self.target_positions = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
        self.target_positions[:, 2] = 1.0  # hover height
        self.actions = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        # Episode statistics buffers
        def torch_zeros():
            return torch.zeros(self._num_envs, dtype=torch.float32, device=self._device, requires_grad=False)

        self.extras = {}
        self.episode_sums = {
            "rew_pos": torch_zeros(),
            "rew_orient": torch_zeros(),
            "rew_effort": torch_zeros(),
            "rew_spin": torch_zeros(),
            "raw_dist": torch_zeros(),
            "raw_orient": torch_zeros(),
            "raw_effort": torch_zeros(),
            "raw_spin": torch_zeros(),
        }

        # Cache state views
        self.root_pos, self.root_rot = self._copters.get_world_poses()
        self.root_velocities = self._copters.get_velocities()
        self.dof_pos = self._copters.get_joint_positions()
        self.dof_vel = self._copters.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # Control buffers (already allocated above; keep explicit for clarity)
        self.thrusts[:] = 0.0
        self.thrust_cmds_damp[:] = 0.0
        self.thrust_rot_damp[:] = 0.0

        # Initialize hover targets
        self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()

        # Target directly above takeoff point at z=1.0 m
        self.target_positions[envs_long, 0:2] = torch.zeros((num_sets, 2), device=self._device, dtype=torch.float32)
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device, dtype=torch.float32) * 1.0

        # Move the visual marker (ball) to the target
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        # Slight visual offset if desired; currently zero
        ball_pos[:, 2] += 0.0
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, :] = torch_rand_float(
            -0.0, 0.0, (num_resets, self._copters.num_dof), device=self._device
        )
        self.dof_vel[env_ids, :] = 0.0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)

        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0.0

        # Apply resets
        self._copters.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._copters.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._copters.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._copters.set_velocities(root_velocities[env_ids], indices=env_ids)

        # Bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.thrust_cmds_damp[env_ids] = 0.0
        self.thrust_rot_damp[env_ids] = 0.0

        # Episode stats (per-env means)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids]) / float(self._max_episode_length)
            self.episode_sums[key][env_ids] = 0.0

    # --------------------------------------------------------------------- #
    # Rewards & terminations
    # --------------------------------------------------------------------- #
    def calculate_metrics(self) -> None:
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_vels = self.root_velocities
        root_angvels = root_vels[:, 3:]

        # Distance to full 3D target (used for general closeness)
        target_dist = torch.sqrt(torch.square(self.target_positions - root_positions).sum(-1))
        pos_reward = 1.0 / (1.0 + target_dist)

        # Uprightness (world Z ⋅ body Z)
        ups = quat_axis(root_quats, 2)
        up_reward = torch.clamp(ups[..., 2], 0.0, 1.0)

        # Effort (action magnitude)
        effort = torch.square(self.actions).sum(-1)

        # Angular rate (smaller is better)
        spin = torch.square(root_angvels).sum(-1)

        pos_err_xy = self.target_positions[:, :2] - root_positions[:, :2]    # [E,2]
        dist_xy = torch.norm(pos_err_xy, dim=-1)                              # [E]
        vel_xy = torch.norm(root_vels[:, :2], dim=-1)                         # [E]

        pos_xy_reward = torch.exp(-2.0 * dist_xy)  # strong pull to XY target
        vel_xy_pen = vel_xy                        # discourage lateral sliding

        # Save values for terminations/logging
        self.target_dist = target_dist
        self.root_positions = root_positions
        self.orient_z = ups[..., 2]

        # Clear, additive shaping (tweak weights to taste)
        self.rew_buf[:] = (
            1.5 * pos_reward            # overall closeness in 3D
        + 1.0 * pos_xy_reward         # explicit XY station-keeping
        + 0.6 * up_reward             # stay upright
        + 0.2 * torch.exp(-1.0 * spin)# low angular rates
        - 0.02 * vel_xy_pen           # damp sideways drift
        - 0.01 * effort               # keep thrust modest
        )

        # Episode logs
        self.episode_sums["rew_pos"] += pos_reward
        self.episode_sums["rew_orient"] += up_reward
        self.episode_sums["rew_effort"] += torch.exp(-0.5 * effort)
        self.episode_sums["rew_spin"] += torch.exp(-1.0 * spin)

        self.episode_sums["raw_dist"] += target_dist
        self.episode_sums["raw_orient"] += self.orient_z
        self.episode_sums["raw_effort"] += effort
        self.episode_sums["raw_spin"] += spin

    def is_done(self) -> None:
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # If far from target (e.g., numerical blowup), end
        die = torch.where(self.target_dist > 5.0, ones, die)

        # Gentler floor early on so it has time to lift
        grace = self.progress_buf < int(1.5 / self.dt)  # ~1.5 s at dt=0.01
        low_floor_now = self.root_positions[..., 2] < 0.0
        low_floor_later = self.root_positions[..., 2] < 0.5

        die = torch.where(grace & low_floor_now, ones, die)
        die = torch.where(~grace & low_floor_later, ones, die)

        # Too high or upside down
        die = torch.where(self.root_positions[..., 2] > 5.0, ones, die)
        die = torch.where(self.orient_z < 0.0, ones, die)

        # NEW: gentle radial XY boundary (prevents unbounded slow drift)
        xy_radius = torch.norm(self.root_positions[..., :2], dim=-1)
        die = torch.where(xy_radius > 2.0, ones, die)   # 2 m radius; adjust as needed

        # Episode length timeout
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
