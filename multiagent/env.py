# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import sys
import os

# Get the current directory (main_directory/script.py's directory)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (main_directory)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
import gym
from gym import spaces
import numpy as np
import math
import carb
import rospy
from sensor_msgs.msg import LaserScan
import tf
from collections import deque
import time
from env_setup_utils.setup_simulation_app import setup_simulation_app
from env_setup_utils.env_setup import setup_action_graph
from collections import deque
import random


class JackalEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            args,
            seed=0,
    ) -> None:
        setup_simulation_app()

        from omni.isaac.core import World
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.sensor import RotatingLidarPhysX
        from env_setup_utils.differential_controller_multiagent import DifferentialController
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.objects import FixedCuboid, DynamicCuboid
        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
        from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
        from omni.isaac.core.articulations import ArticulationView
        import omni.graph.core as og
        from omni.isaac.core.prims import XFormPrimView, RigidPrimView
        import omni.replicator.isaac as dr
        import omni.replicator.core as rep

        self.args = args
        self.num_agents = self.args.num_agents
        physics_dt = 1.0 / self.args.physics_dt
        rendering_dt = 1.0 / self.args.rendering_dt
        self._skip_frame = self.args.skip_frame
        self._dt = physics_dt * self._skip_frame
        self._steps_after_reset = int(rendering_dt / physics_dt)

        # to disable gpu pipeline, add to World(...., sim_params=sim_params)

        # sim_params = {"use_gpu_pipeline":False,
        #               "use_gpu":False,"use_flatcache":False,
        #              }
        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)

        self._my_world.scene.add_default_ground_plane()

        setup_action_graph(self.num_agents)

        jackal_asset_path = self.args.jackal_asset_path

        cloner = GridCloner(spacing=self.args.grid_spacing)
        cloner.define_base_env("/envs")
        define_prim("/envs/env_0")
        VisualCuboid(
            prim_path="/envs/env_0/goal", name="visual_cube",
            position=np.array([5, 0, 0.05]), size=0.1,
            color=np.array([1.0, 0, 0]),
        )
        add_reference_to_stage(
            usd_path=jackal_asset_path, prim_path="/envs/env_0/jackal"
        )

        FixedCuboid(
            prim_path="/envs/env_0/obstacles_1",
            name="obstacles_cube1",
            scale=np.array([2, 2, 10]),
            position=np.array([2.5, 0, 0]),
            size=0.25,
            color=np.array([1.0, 1, 1])
        )

        # clone environments
        prim_paths = cloner.generate_paths("/envs/env", self.num_agents)
        self.env_pos = cloner.clone(source_prim_path="/envs/env_0", prim_paths=prim_paths)

        self.jackal_articulation = ArticulationView(prim_paths_expr="/envs/*/jackal", name="jackal_view")
        self._my_world.scene.add(self.jackal_articulation)
        self.jackal_prim = XFormPrimView("/envs/*/jackal")
        self.obstacle_prim = XFormPrimView("/envs/*/obstacles_1")
        self.goal_prim = XFormPrimView(prim_paths_expr="/envs/*/goal", name="goal_view")

        rospy.init_node("RL_agent")
        for i in range(self.num_agents):
            rospy.Subscriber(f"laser_scan_{i}", LaserScan, self.laser_scan_callback, i)

        self.jackal_controller = DifferentialController(name="simple_control", num_agents=self.num_agents,
                                                        wheel_radius=0.0975,
                                                        wheel_base=0.37559 * 1.35)

        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(self.num_agents, 2), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"),
                                            shape=(self.num_agents, self.args.num_stacks, 364), dtype=np.float32)

        self.last_action = [0, 0]
        self.spawn_distance = np.zeros((self.num_agents))
        self.last_distance = np.zeros((self.num_agents))
        self.scan = np.ones((self.num_agents, 360)) * 10
        self.local_goal = np.empty((self.num_agents, 2))
        self.num_steps = np.zeros((self.num_agents))  # counting number of iterations for each episode
        self.jackal_prev_linear_velo = 0
        self.jackal_prev_ang_velo = 0
        self.stacked_obs = np.zeros((self.num_agents, self.args.num_stacks, 364))
        self.rewards = np.zeros((self.num_agents,), dtype=object)
        for i in range(self.num_agents):
            self.rewards[i] = []
        self.dones = np.zeros((self.num_agents))  # [num_agents, true/False]
        self.infos = [{} for _ in range(self.num_agents)]
        self.ang_velos = deque(maxlen=50)
        self.reset()

        # can only register after reset, Isaac Sim DR framework

        # dr.physics_view.register_simulation_context(self._my_world)
        # # dr.physics_view.register_rigid_prim_view(object_view) #TODO goal prim
        # dr.physics_view.register_articulation_view(self.jackal_articulation)
        # with dr.trigger.on_rl_frame(num_envs=self.num_agents):

        #     with dr.gate.on_env_reset():
        #         dr.physics_view.randomize_articulation_view(
        #             view_name=self.jackal_articulation.name,
        #             operation="direct",
        #             stiffness=tuple([10000000] * self.jackal_articulation.num_dof)
        #             # body_inertias=[[5,5,0.4485],wheel_inertia,wheel_inertia,wheel_inertia,wheel_inertia],
        #         )
        #         print("reset gain:",self.jackal_articulation.get_gains())

        # rep.orchestrator.run()

        print("done init")
        return

    def laser_scan_callback(self, scan, idx):
        self.scan[idx] = np.array(scan.ranges)

    def get_dt(self):
        return self._dt

    def step(self, action: np.ndarray):
        jackal_linear_velo = self.jackal_articulation.get_linear_velocities()
        jackal_angular_velo = self.jackal_articulation.get_angular_velocities()
        self.ang_velos.append(jackal_angular_velo[0][2])
        self.jackal_prev_linear_velo = jackal_linear_velo
        self.jackal_prev_ang_velo = jackal_angular_velo
        self.last_action = action

        for i in range(self.num_agents):
            action[i] = np.clip(action[i], [0, -self.args.w_max], [self.args.v_max, self.args.w_max])

        # we apply our actions to the jackal
        for i in range(self._skip_frame):
            self.jackal_articulation.set_joint_velocities(self.jackal_controller.forward(command=action))
            self._my_world.step(render=False)

        observations = self.get_observations()
        reward, done_info = self.compute_reward(self.local_goal, self.scan)

        self.dones = done_info[:, 0]
        for i in range(self.num_agents):
            self.rewards[i].append(reward[i])
            if self.dones[i]:
                self.infos[i]["done_reason"] = done_info[i]
                ep_rew = sum(self.rewards[i])
                ep_len = len(self.rewards[i])
                ep_info = {"r": round(ep_rew, 6), "l": ep_len}

                self.infos[i]["episode"] = ep_info

                self.rewards[i] = []

        return observations, reward, self.dones, self.infos

    def compute_reward(self, local_goal, scan):
        reward = np.zeros((self.num_agents), dtype=np.float32)
        done_info = np.zeros((self.num_agents, 2))

        for i in range(self.num_agents):
            goal_reward = 0
            collision_reward = 0
            timeout_reward = 0
            distance_reward = 0
            social_reward = 0

            dist_to_goal = math.hypot(local_goal[i][0], local_goal[i][1])

            if dist_to_goal < self.args.goal_radius:
                print(f'Agent_{i} Reahced goal')
                done_info[i] = [True, 1]  # Reach Goalum_steps
            elif np.amin(scan[i]) < self.args.collision_dist:
                done_info[i] = [True, 2]  # Collision
            elif self.num_steps[i] > self.args.timeout:
                print("Timeout")
                done_info[i] = [True, 3]  # Timeout
            else:
                done_info[i] = [False, 0]  # Nothing

            if self.args.final_dist_reward:
                if done_info[i][0]:  # from spawn to goal
                    distance_reward = self.spawn_distance - dist_to_goal

            else:
                distance_reward = (self.last_distance[i] - dist_to_goal) * 5
                if abs(distance_reward) > 0.5:
                    distance_reward = 0  # To prevent init bug

            self.last_distance[i] = dist_to_goal

            if dist_to_goal < self.args.goal_radius:
                goal_reward = 15  # Reach Goal
            if np.amin(scan[i]) < self.args.collision_dist:
                collision_reward = -15  # Collision
            if self.num_steps[i] > self.args.timeout:
                timeout_reward = -15  # Timeout
            if np.amin(scan[i]) < 1.0:
                social_reward = -self.args.social_penalty  # Collision for getting to near

                # big rotation? can just ignore for now

            reward[i] = distance_reward + collision_reward + goal_reward + timeout_reward + social_reward

            self.num_steps[i] += 1
        return reward, done_info

    def reset(self, reset_idx=None):
        if reset_idx == None:
            self.rewards = np.empty((self.num_agents,), dtype=object)
            for i in range(self.num_agents):
                self.rewards[i] = []
            self.dones = np.zeros((self.num_agents))  # [num_agents, true/False]
            self.infos = [{} for _ in range(self.num_agents)]

            self._my_world.reset()
            self.scan = np.ones((self.args.num_agents, 360)) * 10
            print("-------------------reset----------------------------.")
            self.num_steps = np.zeros((self.num_agents))
        else:
            self.dones[reset_idx] = 0

            # uncomment this to enable dr framework
            import omni.replicator.isaac as dr
            # dr.physics_view.step_randomization(reset_inds=[reset_idx])
            reset_indices = list()
            reset_indices.append(reset_idx)

            jackal_reset_pos = np.zeros((self.num_agents, 3)) + np.array(self.env_pos)
            self.jackal_articulation.set_world_poses(positions=jackal_reset_pos[reset_idx].reshape((-1, 3)),
                                                     orientations=np.array([1, 0, 0, 0]).reshape((-1, 4)),
                                                     indices=np.array(reset_idx))

            alpha = 2 * math.pi * np.random.rand()
            r = np.random.uniform(low=4.5, high=5.0)

            # randomize goal position
            goal_reset_pos = np.multiply(np.sin(alpha), r).reshape((1, -1))
            goal_reset_pos = np.hstack((goal_reset_pos, np.multiply(np.cos(alpha), r).reshape((1, -1))))
            goal_reset_pos = np.hstack((goal_reset_pos, np.ones(1).reshape((1, -1)) * 0.05))
            goal_reset_pos = goal_reset_pos + np.array(self.env_pos[reset_idx])
            self.goal_prim.set_world_poses(goal_reset_pos, orientations=np.array([1, 0, 0, 0]).reshape((-1, 4)),
                                           indices=np.array([reset_idx]))

            # random obstacles position
            obstacles_r = np.random.uniform(low=2.0, high=3)
            obstacles_alpha = alpha + np.random.uniform(low=-30.0, high=30.0) / 180 * math.pi
            obstacles_reset_pos = np.multiply(np.sin(obstacles_alpha), obstacles_r).reshape((1, -1))
            obstacles_reset_pos = np.hstack(
                (obstacles_reset_pos, np.multiply(np.cos(obstacles_alpha), obstacles_r).reshape((1, -1))))
            obstacles_reset_pos = np.hstack((obstacles_reset_pos, np.ones(1).reshape((1, -1)) * 0.05))
            obstacles_reset_pos = obstacles_reset_pos + np.array(self.env_pos[reset_idx])

            self.obstacle_prim.set_world_poses(obstacles_reset_pos,
                                               orientations=np.array([1, 0, 0, 0]).reshape((-1, 4)),
                                               indices=np.array([reset_idx]))

            self.scan[reset_idx] = np.ones((360)) * 10
            self.num_steps[reset_idx] = 0
            self.stacked_obs[reset_idx] = np.zeros((self.args.num_stacks, 364))
            self.rewards[reset_idx] = []
            self.infos[reset_idx] = {}

            self.last_distance[reset_idx] = math.hypot(self.local_goal[reset_idx][0], self.local_goal[reset_idx][1])
            self.spawn_distance[reset_idx] = math.hypot(self.local_goal[reset_idx][0], self.local_goal[reset_idx][1])
        observations = self.get_observations()
        return observations

    def get_local_goal(self):
        # robot_odom = ([caretsian(x,y,z)], [quaternion(w,x,y,z])
        robot_odom = self.jackal_articulation.get_world_poses()  # [2,num_envs]
        goal_pose, _ = self.goal_prim.get_world_poses()
        local_goal = []
        for i in range(self.num_agents):
            Quaternions = robot_odom[1][i]
            robot_linear_velo = robot_odom[0][i]
            Euler = tf.transformations.euler_from_quaternion(
                [Quaternions[1], Quaternions[2], Quaternions[3], Quaternions[0]]
            )
            x, y, theta = robot_linear_velo[0], robot_linear_velo[1], Euler[2]
            goal_x, goal_y = goal_pose[i][0], goal_pose[i][1]
            local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
            local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
            local_goal.append((local_x, local_y))
        return np.array(local_goal)

    def get_observations(self):
        self._my_world.render()
        jackal_linear_velocity = self.jackal_articulation.get_linear_velocities()[:, 0].reshape((self.num_agents, -1))
        jackal_angular_velocity = self.jackal_articulation.get_angular_velocities()[:, 2].reshape((self.num_agents, -1))
        self.local_goal = self.get_local_goal()  # relative goalpose to robot

        scan = self.scan / 10 - 0.5
        state = np.concatenate(
            (
                scan,
                self.local_goal,
                # self.last_action, can be ignore for now, can be add in later stage
                jackal_linear_velocity,
                jackal_angular_velocity,
            ),
            axis=1,
            dtype=np.float32,
        )
        self.stacked_obs = np.roll(self.stacked_obs, 1, axis=1)
        self.stacked_obs[:, 0] = state
        return self.stacked_obs

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
