import time
from typing import Dict

import robosuite as suite
from robosuite.controllers import controller_factory
from robosuite.utils.input_utils import *
try:
    from env_robosuite import EnvRobosuite
except ImportError:
    from .env_robosuite import EnvRobosuite
from libero.libero import benchmark
from libero.libero import get_libero_path

import networkx as nx
import os
import json
import numpy as np
from collections import namedtuple

ts_tuple = namedtuple("ts_tuple", ["observation", "reward", "done", "info"])


def to_camel_case(snake_str):
    """Convert snake_case string to CamelCase"""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def get_sg(hdf5_group, sg_name):
    sg_json = hdf5_group[sg_name][()] if sg_name in hdf5_group else None
    if sg_json is None:
        return None
    sg_str = sg_json.decode('utf-8')
    sg = nx.node_link_graph(json.loads(sg_str))
    return sg


def refactor_controller_config(robot, controller_config):
    controller_config["robot_name"] = robot.name
    controller_config["sim"] = robot.sim
    controller_config["eef_name"] = robot.gripper.important_sites["grip_site"]
    controller_config["eef_rot_offset"] = robot.eef_rot_offset
    controller_config["joint_indexes"] = {
        "joints": robot.joint_indexes,
        "qpos": robot._ref_joint_pos_indexes,
        "qvel": robot._ref_joint_vel_indexes,
    }
    controller_config["actuator_range"] = robot.torque_limits
    controller_config["policy_freq"] = robot.control_freq
    controller_config["ndim"] = len(robot.robot_joints)
    return controller_config



class Libero_env_switchable(EnvRobosuite):
    def __init__(self, task_suite_name, task_name,
                 cam_names = ["agentview", "robot0_eye_in_hand"],\
                  W = 128, H = 128, controller_name = "OSC_POSE", abs_action = False,
                  postprocess_visual_obs = True, max_framerate = 25, max_timesteps = 500):
        
        robosuite_version_id = int(suite.__version__.split(".")[1])
        assert robosuite_version_id < 5, "Libero is only compatible to V1.4"
        
        self.task_suite_name = task_suite_name
        self.task_name = task_name
        
        self.max_timesteps = max_timesteps
        self.max_framerate = max_framerate
        self.options = {}
        # self.options["env_name"] = env_name
        self.options["camera_names"] = cam_names
        self.options["camera_heights"] = H
        self.options["camera_widths"] = W
        self.options["camera_segmentations"] = "instance"

        self.env = None 

        default_controller_configs = self.init_controller_configs(controller_name, abs_action)
        # self.controller_name = controller_name
        # default_controller_configs = suite.load_controller_config(default_controller=controller_name)
        self.options["controller_configs"] = default_controller_configs
        self.controller_configs = default_controller_configs

        ## setup libero env
        task_suite = benchmark.get_benchmark_dict()[self.task_suite_name]()
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if task.name in self.task_name:
                self.task_id = task_id
                break
        # task = task_suite.get_task(self.task_id)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {self.task_id} from suite {self.task_suite_name},  \
             and the bddl file is {task_bddl_file}")
        self.options['bddl_file_name'] = task_bddl_file
        self.options['robots'] = ['Panda']

        super().__init__(
            env_name = "Libero_Tabletop_Manipulation",
            render = True,
            render_offscreen = True,
            use_image_obs = True,
            # use_depth_obs = True,
            postprocess_visual_obs = postprocess_visual_obs,
            # env_lang = None,
            **self.options
        )

        self.raw_obs = self.env.reset()
        
    # def reset_ts(self, with_planning = False):
    #     raise NotImplementedError("Resetting is not implemented yet")
    
    # def step_ts(self, action):
    #     raise NotImplementedError("Stepping is not implemented yet")

    def reset_ts(self, with_planning = False):
        self.raw_obs = self.env.reset()
        # self.obs = self.get_observation(self.raw_obs)
        init_ts = ts_tuple(self.raw_obs, 0, False, {})
        return init_ts

    def step_ts(self, action):
        self.raw_obs, reward, done, info = self.env.step(action)
        # self.obs = self.get_observation(self.raw_obs)
        info["is_success"] = self.is_success()
        return ts_tuple(self.raw_obs, reward, done, info)

    def init_controller_configs(self, controller_name="OSC_POSE", abs_action=False):
        # config the JOINT_POSITION controller
        self.abs_joint_controller_config = {
            "type": "JOINT_POSITION",
            "input_max": 3.14,
            "input_min": -3.14,
            # "output_max": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            # "output_min": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
            "output_max": [1, 1, 1, 1, 1, 1, 1],
            "output_min": [-1, -1, -1, -1, -1, -1, -1],
            "kp": 50,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": False,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }

        # config the OSC_POSE controller, input_type is delta by default
        self.relative_osc_pose_controller_config = suite.load_controller_config(default_controller="OSC_POSE")

        # config the abs OSC_POSE controller
        self.absolute_osc_pose_controller_config = suite.load_controller_config(default_controller="OSC_POSE")
        self.absolute_osc_pose_controller_config["control_delta"] = True  # use absolute actions

        default_controller_configs = self.update_controller_configs(
            controller_name=controller_name, abs_action=abs_action
        )
        return default_controller_configs


    def update_controller_configs(self, controller_name="OSC_POSE", abs_action=False):
        """Update the robot's controller configuration.
        
        Args:
            controller_name (str): Name of the controller type to use (e.g. "OSC_POSE", "OSC_POSITION", etc.)
            abs_action (bool): If True, use absolute actions for OSC_POSE controller. If False, use delta actions.
        """
        self.controller_name = controller_name
        self.abs_action = abs_action

        # Load the base controller config for the specified controller type
        if controller_name == "OSC_POSE":
            if abs_action:
                arm_controller_config = self.absolute_osc_pose_controller_config
            else:
                arm_controller_config = self.relative_osc_pose_controller_config
        elif controller_name == "JOINT_POSITION":
            arm_controller_config = self.abs_joint_controller_config
        else:
            raise ValueError(f"Unsupported controller name: {controller_name}")
        
        # robot = self.options["robots"][0] if isinstance(self.options["robots"], list) else self.options["robots"]
        # # Convert to composite controller config format
        if self.env is not None:
            updated_controller_configs = refactor_controller_config(self.env.robots[0], arm_controller_config)
            return updated_controller_configs
        else:
            return arm_controller_config

        
        
    
    def update_controllers(self, controller_name="OSC_POSE", abs_action=False):

        self.controller_configs = self.update_controller_configs(
            controller_name=controller_name, abs_action=abs_action
        )
        ## do partial reset following _reset_internal()
        self.env._action_dim = 0
        for robot in self.env.robots:
            robot.controller = controller_factory(controller_name, self.controller_configs)
            robot.controller.update_base_pose(robot.base_pos, robot.base_ori)
            robot.controller.reset_goal()

            ## important: reset env._action_dim
            self.env._action_dim += robot.action_dim


        #     # Get the switchable controller instance
        #     controller = robot.controller
            
        #     # Create a unique name for this configuration
        #     config_name = f"{controller_name}_{'abs' if abs_action else 'delta'}"
            
        #     # Add or update the configuration
        #     controller.add_configuration(
        #         name=config_name,
        #         part_controller_config=self.controller_configs["body_parts"],
        #         composite_controller_specific_config=self.controller_configs
        #     )
            
        #     # Switch to the new configuration
        #     controller.switch_configuration(config_name)
        
        # self.env.reset_controller(self.controller_configs)
        # Log the change
        print(
            f"Switched to {controller_name} controller with {'absolute' if abs_action else 'delta'} actions"
        )


    def  handle_rewards(self):
        return self.env.reward() >=1
    
    def exit(self):
        self.env.close()

    def replay_tamp_step(self, total_action):
        start = time.time()

        ts = self.step_ts(total_action)
        self.env.render()
        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / self.max_framerate - elapsed
        if diff > 0:
            time.sleep(diff)
        return ts

    def get_cur_jpose(self):
        cur_obs = self.env._get_observations(force_update = True)
        robot_jposes = []
        for robot in self.env.robots:
            robot_nick_name = f'robot{robot.idn}'
            jpose = cur_obs[f"{robot_nick_name}_joint_pos"].reshape(1, -1)
            robot_jposes.append(jpose)
        robot_jposes = np.concatenate(robot_jposes, axis=0)
        return robot_jposes

    def test_controller(self, controller_name="OSC_POSE", abs_action=False):

        self.update_controllers(
            controller_name=controller_name, 
            abs_action=abs_action
        )
        joint_dim = 7    
        # Define the pre-defined controller actions to use (action_dim, num_test_steps, test_value)
        controller_settings = {
            "OSC_POSE": [6, 6, 0.1],
            "OSC_POSITION": [3, 3, 0.1],
            "IK_POSE": [6, 6, 0.01],
            "JOINT_POSITION": [joint_dim, joint_dim, 1],
            "JOINT_VELOCITY": [joint_dim, joint_dim, -0.1],
            "JOINT_TORQUE": [joint_dim, joint_dim, 0.25],
        }

        # Define variables for each controller test
        action_dim = controller_settings[self.controller_name][0]
        num_test_steps = controller_settings[self.controller_name][1]
        test_value = controller_settings[self.controller_name][2]

        # Define the number of timesteps to use per controller action as well as timesteps in between actions
        steps_per_action = 75
        steps_per_rest = 75


        # To accommodate for multi-arm settings (e.g.: Baxter), we need to make sure to fill any extra action space
        # Get total number of arms being controlled
        n = 0
        gripper_dim = 0
        for robot in self.env.robots:
            gripper_dim = robot.gripper["right"].dof if isinstance(robot, Bimanual) else robot.gripper.dof
            n += int(robot.action_dim / (action_dim + gripper_dim))
        neutral = np.zeros(action_dim + gripper_dim)
        
        count = 0
        # Loop through controller space
        while count < num_test_steps:
            action = neutral.copy()
            for i in range(steps_per_action):
                start = time.time()
                # action = np.array([-0.876209485918424, 1.2561013815928623, 1.3308805441681182, -1.9343220594253552, -1.3210136402291082, 1.8487462748068064, -0.927032436745828, 1 ])
                action[count] = test_value
                # total_action = np.tile(action, n)
                self.env.step(action)
                # self.env.step(total_action)
                self.env.render()

                # limit frame rate if necessary
                elapsed = time.time() - start
                diff = 1 / self.max_framerate - elapsed
                if diff > 0:
                    time.sleep(diff)
            for i in range(steps_per_rest):
                start = time.time()
                # total_action = np.tile(neutral, n)
                self.env.step(neutral)
                # self.env.step(total_action)
                self.env.render()

                # limit frame rate if necessary
                elapsed = time.time() - start
                diff = 1 / self.max_framerate - elapsed
                if diff > 0:
                    time.sleep(diff)
            count += 1


if __name__ == "__main__":
    # env_name = to_camel_case("two_arm_three_piece_assembly")
    task_name  = 'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate'
    dmg_wrapper = Libero_env_switchable(task_suite_name='libero_spatial', task_name= task_name,  controller_name = "OSC_POSE", abs_action=False)
    dmg_wrapper.test_controller("JOINT_POSITION", abs_action = True)