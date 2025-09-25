# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
# (license text unchanged)

import numpy as np
import torch

from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *  # quat_axis, quat_to_euler

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.crazyflie import Crazyflie
from omniisaacgymenvs.robots.articulations.views.crazyflie_view import CrazyflieView

EPS = 1e-6


def torch_rand_float(low, high, shape, device):
    return (high - low) * torch.rand(shape, device=device, dtype=torch.float32) + low


class CrazyflieTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self._num_observations = 18   # [pos_err(3), body_axes(9), lin_vel(3), ang_vel(3)]
        self._num_actions = 4         # 4 motors

        self._crazyflie_position = torch.tensor([0.0, 0.0, 0.20])
        self._ball_position = torch.tensor([0.0, 0.0, 1.0])

        super().__init__(name=name, env=env)

    # ----------------------------- config -------------------------------- #
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self.dt = self._task_cfg["sim"]["dt"]

        # near-Crazyflie params
        self.arm_length = 0.046
        self.mass = 0.027
        self.thrust_to_weight = 2.0

        # motor lag (slower → smoother) — slightly faster than before to allow corrections
        self.motor_damp_time_up = 0.08     # was 0.12
        self.motor_damp_time_down = 0.10   # was 0.15
        self.motor_tau_up = 4.0 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4.0 * self.dt / (self.motor_damp_time_down + EPS)

        # nominal motor asymmetry (sum=4)
        self.motor_assymetry = np.array([1.00, 0.98, 1.02, 1.00], dtype=np.float32)
        self.motor_assymetry = self.motor_assymetry * 4.0 / np.sum(self.motor_assymetry)

        self.grav_z = -1.0 * float(self._task_cfg["sim"]["gravity"][2])

        # rate limit (in raw [-1,1] space) — loosened a bit
        self.max_action_delta = 0.10  # was 0.06

        # output filter + HF penalty params
        self.filter_alpha = 0.5       # was 0.7 (less smoothing so it can correct down)
        self.hist_len = 16            # steps kept for HF penalty
        self.hf_penalty_enabled = False  # enable later after hover is stable

        # curriculum / randomization toggles
        self.enable_domain_rand = False  # enable after baseline hover success > 70%

    # ------------------------------ scene -------------------------------- #
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
        color = torch.tensor([1.0, 0.0, 0.0])  # RGB only
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

    # ---------------------------- observations --------------------------- #
    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        self.root_velocities = self._copters.get_velocities(clone=False)

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot

        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)

        # simple 1-step IMU delay + noise
        if not hasattr(self, "_obs_delay_buf"):
            self._obs_delay_buf = {
                "lin": torch.zeros_like(self.root_velocities[:, :3]),
                "ang": torch.zeros_like(self.root_velocities[:, 3:]),
            }
        lin_meas = self._obs_delay_buf["lin"]
        ang_meas = self._obs_delay_buf["ang"]
        self._obs_delay_buf["lin"] = self.root_velocities[:, :3] + 0.02 * torch.randn_like(self.root_velocities[:, :3])
        self._obs_delay_buf["ang"] = self.root_velocities[:, 3:] + 0.02 * torch.randn_like(self.root_velocities[:, 3:])

        self.obs_buf[..., 0:3] = self.target_positions - root_positions
        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z
        self.obs_buf[..., 12:15] = lin_meas
        self.obs_buf[..., 15:18] = ang_meas

        return {self._copters.name: {"obs_buf": self.obs_buf}}

    # -------------------------- action application ----------------------- #
    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)

        # rate limit in [-1,1]
        if not hasattr(self, "prev_raw_actions"):
            self.prev_raw_actions = torch.zeros_like(actions)
        raw = torch.clamp(actions, -1.0, 1.0)
        delta = torch.clamp(raw - self.prev_raw_actions,
                            -2 * self.max_action_delta, 2 * self.max_action_delta)
        raw = self.prev_raw_actions + delta
        self.prev_raw_actions = raw.clone()

        # smoothness terms use post-limited actions
        self.action_diff = (raw - getattr(self, "prev_actions", torch.zeros_like(raw)))
        self.prev_actions = raw.clone()
        self.actions = raw

        # --- Hover-referenced thrust mapping ---
        # policy outputs raw in [-1,1]; map to a delta around per-env hover
        if not hasattr(self, "u_hover_frac"):
            # Fallback before post_reset computes it
            self.u_hover_frac = torch.full((self._num_envs,), 0.5, device=self._device, dtype=torch.float32)
        u_hover = self.u_hover_frac.unsqueeze(1).expand(-1, 4)  # [E,4]
        delta = 0.25 * u_hover * raw                             # +/-25% of hover thrust
        thrust_cmds = torch.clamp(u_hover + delta, 0.0, 1.0)
        # --------------------------------------

        # first-order motor lag in sqrt space
        if not hasattr(self, "thrust_rot_damp"):
            self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        if not hasattr(self, "thrust_cmds_damp"):
            self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)

        motor_tau = self.motor_tau_up * torch.ones((self._num_envs, 4), dtype=torch.float32, device=self._device)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau = torch.clamp(motor_tau, 0.0, 1.0)

        thrust_rot = torch.sqrt(thrust_cmds)
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp ** 2

        # small noise
        self.thrust_cmds_damp = torch.clamp(
            self.thrust_cmds_damp + thrust_cmds * (0.01 * torch.randn_like(self.thrust_cmds_damp)), 0.0, 1.0
        )

        # low-pass filter on commands (reduces twitch)
        if not hasattr(self, "filtered_thrust"):
            self.filtered_thrust = self.thrust_cmds_damp.clone()
        self.filtered_thrust = self.filter_alpha * self.filtered_thrust + (1.0 - self.filter_alpha) * self.thrust_cmds_damp

        # 1-step actuation delay (use previous filtered command)
        if not hasattr(self, "applied_cmd"):
            self.applied_cmd = self.filtered_thrust.clone()
        cmd_for_step = self.applied_cmd
        self.applied_cmd = self.filtered_thrust.clone()

        # keep short history for HF penalty
        if not hasattr(self, "thrust_hist"):
            self.thrust_hist = torch.zeros((self.hist_len, self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.thrust_hist = torch.roll(self.thrust_hist, shifts=1, dims=0)
        self.thrust_hist[0] = cmd_for_step.detach()

        # thrusts [E,4]
        thrusts = self.thrust_max * cmd_for_step

        # rotate to world frame
        root_quats = self.root_rot
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        rot_matrix = torch.cat((rot_x, rot_y, rot_z), dim=1).reshape(-1, 3, 3)

        if not hasattr(self, "thrusts"):
            self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        force_xy = torch.zeros(self._num_envs, 4, 2, dtype=torch.float32, device=self._device)
        thrusts_3d = torch.cat((force_xy, thrusts.view(-1, 4, 1)), dim=2)

        for i in range(4):
            mod_thrust_i = torch.matmul(rot_matrix, thrusts_3d[:, i].unsqueeze(-1)).squeeze(-1)
            self.thrusts[:, i] = mod_thrust_i

        if len(reset_env_ids) > 0:
            self.thrusts[reset_env_ids] = 0.0

        # rotor visuals
        if not hasattr(self, "dof_vel"):
            self.dof_vel = self._copters.get_joint_velocities()
        prop_rot = cmd_for_step * 433.3
        self.dof_vel[:, 0] = prop_rot[:, 0]
        self.dof_vel[:, 1] = -prop_rot[:, 1]
        self.dof_vel[:, 2] = prop_rot[:, 2]
        self.dof_vel[:, 3] = -prop_rot[:, 3]
        self._copters.set_joint_velocities(self.dof_vel)

        # apply rotor forces
        for i in range(4):
            self._copters.physics_rotors[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)

        # light aero damping
        lin = self.root_velocities[:, :3]
        ang = self.root_velocities[:, 3:]
        lin_damp = -0.02 * lin
        ang_damp = -0.002 * ang
        if hasattr(self._copters, "apply_base_external_forces"):
            self._copters.apply_base_external_forces(lin_damp)
        if hasattr(self._copters, "apply_base_external_torques"):
            self._copters.apply_base_external_torques(ang_damp)

    # ---------------------------- reset & init --------------------------- #
    def post_reset(self):
        # per-env motor asymmetry / mass / T/W
        if self.enable_domain_rand:
            asym = 1.0 + 0.05 * torch.randn(self._num_envs, 4, device=self._device)
            asym = torch.clamp(asym, 0.8, 1.2)
            asym = 4.0 * (asym / (torch.sum(asym, dim=1, keepdim=True) + EPS))
            mass_env = self.mass * torch.clamp(1.0 + 0.10 * torch.randn(self._num_envs, device=self._device), 0.9, 1.1)
            ttw_env = self.thrust_to_weight * torch.clamp(1.0 + 0.15 * torch.randn(self._num_envs, device=self._device), 0.7, 1.3)
        else:
            asym = torch.ones(self._num_envs, 4, device=self._device)
            mass_env = torch.full((self._num_envs,), self.mass, device=self._device)
            ttw_env = torch.full((self._num_envs,), self.thrust_to_weight, device=self._device)
        self.motor_asym_env = asym

        thrust_max_env = (self.grav_z * mass_env * ttw_env).unsqueeze(-1) * (self.motor_asym_env / 4.0)
        self.thrust_max = thrust_max_env.to(torch.float32)

        # per-env hover fraction of max command (sum over motors)
        total_thrust_max = torch.sum(self.thrust_max, dim=1)  # [E,3] per motor vector? wait: thrust_max is scalar per motor along z
        # thrust_max is scalar magnitude per motor (not 3D). Sum over motor magnitudes:
        total_thrust_max_scalar = torch.sum(self.thrust_max.squeeze(-1), dim=1)  # [E]
        self.u_hover_frac = (self.mass * self.grav_z) / (total_thrust_max_scalar + 1e-6)
        self.u_hover_frac = torch.clamp(self.u_hover_frac, 0.35, 0.65)

        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.filtered_thrust = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.applied_cmd = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_hist = torch.zeros((self.hist_len, self._num_envs, 4), dtype=torch.float32, device=self._device)

        self.target_positions = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self._device)
        self.target_positions[:, 2] = 1.0
        self.actions = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.prev_actions = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.prev_raw_actions = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.action_diff_prev = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

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

        self.root_pos, self.root_rot = self._copters.get_world_poses()
        self.root_velocities = self._copters.get_velocities()
        self.dof_pos = self._copters.get_joint_positions()
        self.dof_vel = self._copters.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        self.hover_dwell = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

        self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()

        self.target_positions[envs_long, 0:2] = torch.zeros((num_sets, 2), device=self._device, dtype=torch.float32)
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device, dtype=torch.float32) * 1.0

        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
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

        self._copters.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._copters.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._copters.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._copters.set_velocities(root_velocities[env_ids], indices=env_ids)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # clear control/filters for these envs
        self.thrust_cmds_damp[env_ids] = 0.0
        self.thrust_rot_damp[env_ids] = 0.0
        self.filtered_thrust[env_ids] = 0.0
        self.applied_cmd[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_raw_actions[env_ids] = 0.0
        self.action_diff_prev[env_ids] = 0.0
        self.hover_dwell[env_ids] = 0.0
        self.thrust_hist[:, env_ids, :] = 0.0

        # re-randomize per reset (off for baseline hover)
        if self.enable_domain_rand:
            asym = 1.0 + 0.05 * torch.randn(len(env_ids), 4, device=self._device)
            asym = torch.clamp(asym, 0.8, 1.2)
            asym = 4.0 * (asym / (torch.sum(asym, dim=1, keepdim=True) + EPS))
            self.motor_asym_env[env_ids] = asym
            mass_env = self.mass * torch.clamp(1.0 + 0.10 * torch.randn(len(env_ids), device=self._device), 0.9, 1.1)
            ttw_env = self.thrust_to_weight * torch.clamp(1.0 + 0.15 * torch.randn(len(env_ids), device=self._device), 0.7, 1.3)
        else:
            asym = torch.ones(len(env_ids), 4, device=self._device)
            self.motor_asym_env[env_ids] = asym
            mass_env = torch.full((len(env_ids),), self.mass, device=self._device)
            ttw_env = torch.full((len(env_ids),), self.thrust_to_weight, device=self._device)

        thrust_max_env = (self.grav_z * mass_env * ttw_env).unsqueeze(-1) * (asym / 4.0)
        self.thrust_max[env_ids] = thrust_max_env.to(torch.float32)

        # update hover fraction for these envs
        total_thrust_max_scalar = torch.sum(self.thrust_max[env_ids].squeeze(-1), dim=1)  # [num_resets]
        self.u_hover_frac[env_ids] = (self.mass * self.grav_z) / (total_thrust_max_scalar + 1e-6)
        self.u_hover_frac[env_ids] = torch.clamp(self.u_hover_frac[env_ids], 0.35, 0.65)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids]) / float(self._max_episode_length)
            self.episode_sums[key][env_ids] = 0.0

    # ----------------------- rewards & terminations ---------------------- #
    def _attitude_error_reward(self, quats):
        up = quat_axis(quats, 2)     # body Z
        fwd = quat_axis(quats, 0)    # body X
        up_term = torch.clamp(up[..., 2], 0.0, 1.0)
        yaw_term = torch.clamp(fwd[..., 0], -1.0, 1.0)
        return 0.8 * up_term + 0.2 * (yaw_term + 1.0) * 0.5

    def calculate_metrics(self) -> None:
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_vels = self.root_velocities
        root_angvels = root_vels[:, 3:]

        pos_err = self.target_positions - root_positions
        err_xy = torch.norm(pos_err[:, :2], dim=-1)
        err_z = torch.abs(pos_err[:, 2])
        dist_3d = torch.sqrt(torch.square(pos_err).sum(-1))

        prog = torch.clamp(self.progress_buf.to(torch.float32) / float(max(self._max_episode_length, 1)), 0.0, 1.0)
        tol_xy = 0.50 - 0.40 * prog
        tol_z = 0.40 - 0.30 * prog

        pos3d_reward = 1.0 / (1.0 + dist_3d)
        pos_xy_reward = torch.exp(-2.0 * err_xy)
        pos_z_reward = torch.exp(-6.0 * err_z)

        up_reward = self._attitude_error_reward(root_quats)

        vel_xy = torch.norm(root_vels[:, :2], dim=-1)
        vz = root_vels[:, 2]
        effort = torch.square(self.actions).sum(-1)
        spin = torch.square(root_angvels).sum(-1)

        net_force_world = torch.sum(self.thrusts, dim=1)
        lateral_force = torch.norm(net_force_world[:, :2], dim=-1)
        lateral_force_norm = lateral_force / (self.mass * self.grav_z + 1e-6)

        # action smoothness + jerk (CAPS-style)
        act_smooth_pen = torch.square(self.action_diff).sum(-1) if hasattr(self, "action_diff") else 0.0
        if hasattr(self, "action_diff_prev"):
            jerk_pen = torch.square(self.action_diff - self.action_diff_prev).sum(-1)
        else:
            jerk_pen = 0.0
        self.action_diff_prev = getattr(self, "action_diff", torch.zeros_like(self.prev_actions))

        # high-frequency penalty from history (optionally disabled early)
        if hasattr(self, "thrust_hist") and self.hist_len > 1 and self.hf_penalty_enabled:
            diffs = self.thrust_hist[:-1] - self.thrust_hist[1:]            # [H-1,E,4]
            hf_energy = torch.square(diffs).sum(dim=(0, 2)) / float(self.hist_len - 1)  # [E]
        else:
            hf_energy = 0.0

        # signed z-error velocity shaping: drive toward target and settle
        z_err = pos_err[:, 2]  # z - z_tgt
        desired_vz = -0.8 * torch.tanh(z_err)          # cap approach speed ~0.8 m/s
        r_zdir = -torch.abs(vz - desired_vz)           # best when vz ≈ desired_vz

        z = root_positions[:, 2]

        within_xy = (err_xy < torch.clamp(tol_xy, min=0.08)).float()
        within_z = (err_z < torch.clamp(tol_z, min=0.08)).float()
        within_ang = (up_reward > 0.95).float()
        in_box = within_xy * within_z * within_ang
        self.hover_dwell = torch.where(in_box > 0.5, self.hover_dwell + 1.0, torch.zeros_like(in_box))
        dwell_bonus = torch.clamp(self.hover_dwell, 0.0, 200.0) / 200.0

        near_hover_gate = ((err_xy < 0.20) & (z > 0.8)).float()
        per_motor_mag = torch.norm(self.thrusts, dim=-1)
        thrust_imbalance_pen = near_hover_gate * torch.var(per_motor_mag, dim=-1)

        # weights
        w_pos3d, w_xy, w_z = 1.0, 1.2, 1.2
        w_att, w_spin = 0.9, 0.2
        w_velxy, w_lat = 0.03, 0.03
        w_eff, w_act = 0.01, 0.010
        w_jerk = 0.010
        w_hf = 0.020 if self.hf_penalty_enabled else 0.0
        w_zdir = 0.3
        w_dwell, w_imb = 0.10, 0.02

        self.rew_buf[:] = (
            w_pos3d * pos3d_reward
            + w_xy * pos_xy_reward
            + w_z * pos_z_reward
            + w_att * up_reward
            + w_spin * torch.exp(-1.0 * spin)
            - w_velxy * vel_xy
            - w_lat * lateral_force_norm
            - w_eff * effort
            - w_act * act_smooth_pen
            - w_jerk * jerk_pen
            - w_hf * hf_energy
            + w_zdir * r_zdir
            - w_imb * thrust_imbalance_pen
            + w_dwell * dwell_bonus
        )

        self.target_dist = dist_3d
        self.root_positions = root_positions
        self.orient_z = quat_axis(root_quats, 2)[..., 2]

        self.episode_sums["rew_pos"] += pos3d_reward
        self.episode_sums["rew_orient"] += up_reward
        self.episode_sums["rew_effort"] += torch.exp(-0.5 * effort)
        self.episode_sums["rew_spin"] += torch.exp(-1.0 * spin)
        self.episode_sums["raw_dist"] += dist_3d
        self.episode_sums["raw_orient"] += self.orient_z
        self.episode_sums["raw_effort"] += effort
        self.episode_sums["raw_spin"] += spin

    def is_done(self) -> None:
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        die = torch.where(self.target_dist > 5.0, ones, die)

        grace = self.progress_buf < int(1.5 / self.dt)
        low_floor_now = self.root_positions[..., 2] < 0.0
        low_floor_later = self.root_positions[..., 2] < 0.5
        die = torch.where(grace & low_floor_now, ones, die)
        die = torch.where(~grace & low_floor_later, ones, die)

        # ceiling tied to target to avoid extended climb bias
        too_high = self.root_positions[..., 2] > (self.target_positions[..., 2] + 0.8)
        die = torch.where(too_high, ones, die)

        die = torch.where(self.root_positions[..., 2] > 5.0, ones, die)
        die = torch.where(self.orient_z < 0.0, ones, die)

        xy_radius = torch.norm(self.root_positions[..., :2], dim=-1)
        die = torch.where(xy_radius > 2.0, ones, die)

        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
