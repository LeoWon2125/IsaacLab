from pxr import UsdGeom, Gf
from math import cos, sin
import random

from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.ladder.ladder_climbing_env_cfg import LocomotionLadderEnvCfg, RewardsCfg
from isaaclab_assets import G1_WITH_HAND_CFG

@configclass
class G1Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])}
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"])}
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        ])}
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"])}
    )


@configclass
class G1LadderRoughEnvCfg(LocomotionLadderEnvCfg):
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_WITH_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # --- Ladder Parameters ---
        # theta = random.uniform(0.0, 1.0)
        theta = 0.5
        print("theta : ", theta)
        # rung_spacing = random.uniform(0.25, 0.50)
        rung_spacing = 0.35
        print("rung_spacing : ", rung_spacing)

        ladder_theta = theta
        rung_radius = 0.025
        rung_length = 0.7
        rung_count = 15
        margin = 0.3
        ladder_height = (rung_count - 1) * rung_spacing + 2 * margin

        side_width = 0.05
        quat = (cos(ladder_theta / 2), 0.0, sin(ladder_theta / 2), 0.0)  # (w, x, y, z)
        side_offset = rung_length / 2 + side_width / 2

        base_x = -1.0
        center_x = base_x + ladder_height*sin(ladder_theta)

        dir_vec = (sin(ladder_theta), 0.0, cos(ladder_theta))
        pole_center = [h * (ladder_height / 2) for h in dir_vec[:3]]
        pole_center[0] = center_x
        # --- Left Pole ---
        self.scene.left_pole = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/left_pole",
            spawn=sim_utils.CuboidCfg(
                size=(side_width, side_width, ladder_height),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(center_x, -side_offset, pole_center[2]),
                rot=quat,
            )
        )

        # --- Right Pole ---
        self.scene.right_pole = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/right_pole",
            spawn=sim_utils.CuboidCfg(
                size=(side_width, side_width, ladder_height),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(center_x, side_offset, pole_center[2]),
                rot=quat,
            )
        )
        base_y = 0.0  # pole 사이 중앙

        for i in range(rung_count):
            z_offset = margin + i * rung_spacing
            local_offset = z_offset - ladder_height / 2  # 중심 기준으로 오프셋
            rung_pos = (
                center_x + dir_vec[0] * local_offset,  # x
                0.0,
                pole_center[2] + dir_vec[2] * local_offset,  # z
            )
            setattr(
                self.scene,
                f"rung_{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/rung_{i}",
                    spawn=sim_utils.CylinderCfg(
                        radius=rung_radius,
                        height=rung_length,
                        axis="Y",
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=True,
                            kinematic_enabled=True,
                            disable_gravity=True,
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=rung_pos,
                    ),
                )
            )

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1LadderEnvCfg(G1LadderRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])

        self.scene.env_spacing = 6.0
        self.scene.terrain.max_init_terrain_level = None
        self._add_ladders_after_scene_build = True


@configclass
class G1LadderEnvCfg_PLAY(G1LadderEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None