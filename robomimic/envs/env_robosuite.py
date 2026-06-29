"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy
import open3d as o3d

import robosuite
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

try:
    # this is needed for ensuring robosuite can find the additional mimicgen environments (see https://mimicgen.github.io)
    import mimicgen
except ImportError:
    pass
try:
    # this is needed for ensuring robosuite can find the additional mimicgen environments (see https://mimicgen.github.io)
    import mimicgen_envs
except ImportError:
    pass

try:
    # try to import mimicgen environments
    import dexmimicgen
except ImportError:
    print("WARNING: could not import dexmimicgen envs")

try:
    # try to import LIBERO environments
    from libero.libero.envs import *
except ImportError:
    print("WARNING: could not import LIBERO envs")
    
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB

# protect against missing mujoco-py module, since robosuite might be using mujoco-py or DM backend
try:
    import mujoco_py
    MUJOCO_EXCEPTIONS = [mujoco_py.builder.MujocoException]
except ImportError:
    MUJOCO_EXCEPTIONS = []

## depth cameras that used to construct instance pc
# D_CAM = ['agentview', 'birdview', 'frontview'] 
# OBJ_PC_SIZE = 256

def get_name2id(env):
    """
    Creates a mapping from instance names to their corresponding IDs.
    
    Args:
        env: The environment object containing instance information.
        
    Returns:
        dict: A dictionary mapping instance names to their IDs.
    """
    return {inst: (i+1) for i, inst in enumerate(list(env.model.instances_to_ids.keys()))}


def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None and color.shape[0] > 0:
        try:
            assert pcd.shape[0] == color.shape[0]
            assert color.max() <= 1
            assert color.min() >= 0
        except:
            raise ValueError(f"Point cloud and color shape mismatch or color values out of range: {pcd.shape}, {color.shape}")
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def get_interested_objects(env, env_name):
        ## set interested objects
    name2id = get_name2id(env)
    if env_name == 'TwoArmThreePieceAssembly':
        interested_objects = ['base', 'piece_1', 'piece_2']
    elif env_name == 'TwoArmThreading':
        interested_objects = ['tripod_obj', 'needle_obj']
    elif env_name == 'TwoArmDrawerCleanup':
        interested_objects = ['DrawerObject', 'cleanup_object']
    elif env_name.startswith('Libero_'):
        interested_objects = env.obj_of_interest
    else:
        raise NotImplementedError(f"Unsupported environment: {env_name}")
    
    return interested_objects

def get_d_cams(env_name):
    """Get depth cameras used to construct instance point cloud."""
    if env_name.startswith('Libero_'):
        return ['agentview', 'birdview']
    else:
        return ['agentview']

# def is_cam_used(env, cam_name):
#     if 'Libero_' in env._env_name:
#         return True
    
#     d_cams = get_d_cams(env._env_name)
#     if cam_name in d_cams:
#         return True
#     return False

    
class EnvRobosuite(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        # robosuite version check
        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"

        kwargs = deepcopy(kwargs)

        # update kwargs based on passed arguments
        update_kwargs = dict(
            has_renderer=render,
            has_offscreen_renderer=(render_offscreen or use_image_obs),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            camera_depths=True,
        )
        kwargs.update(update_kwargs)

        if self._is_v1:
            if kwargs["has_offscreen_renderer"]:
                # ensure that we select the correct GPU device for rendering by testing for EGL rendering
                # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
                import egl_probe
                valid_gpu_devices = egl_probe.get_available_devices()
                if len(valid_gpu_devices) > 0:
                    kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
        else:
            # make sure gripper visualization is turned off (we almost always want this for learning)
            kwargs["gripper_visualization"] = False
            del kwargs["camera_depths"]
            kwargs["camera_depth"] = False # rename kwarg
        ## unsupported kwargs, which is for robosuite 1.5.1
        if 'env_lang' in kwargs:
            del kwargs['env_lang']
        ## for 'object-state'
        kwargs['use_object_obs'] = True


        self.output_all_pcds = False
        if "output_all_pcds" in kwargs:
            self.output_all_pcds = kwargs["output_all_pcds"]               
            del kwargs["output_all_pcds"]

        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.env = robosuite.make(self._env_name, **kwargs)

        if self._is_v1:
            # Make sure joint position observations and eef vel observations are active
            for ob_name in self.env.observation_names:
                if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                    self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

        self.ws_size = 0.6

        if env_name.startswith('Libero_'):
            # Tight workspace for libero — narrow z range to filter gripper.
            voxel_center = np.array([0, 0, 0.03])
            pc_center = np.array([0, 0, 0.03])
            if hasattr(self.env, 'workspace_offset'):
                pc_center = np.array(self.env.workspace_offset)
            elif hasattr(self.env, 'table_offset'):
                pc_center = np.array(self.env.table_offset)
            elif hasattr(self.env, 'table_offsets'):
                pc_center = np.mean(self.env.table_offsets, axis=0)
            pc_center[2] = pc_center[2] + 0.02
            voxel_center = pc_center

            self.voxel_workspace = np.array([
                [voxel_center[0] - self.ws_size/2, voxel_center[0] + self.ws_size/2],
                [voxel_center[1] - self.ws_size/2, voxel_center[1] + self.ws_size/2],
                [voxel_center[2], voxel_center[2] + self.ws_size/2],
            ])
            self.pc_workspace = np.array([
                [pc_center[0] - self.ws_size/2, pc_center[0] + self.ws_size/2],
                [pc_center[1] - self.ws_size/2, pc_center[1] + self.ws_size/2],
                [pc_center[2], pc_center[2] + self.ws_size*0.4],
            ])
        else:
            # Dmg + other envs — wide workspace (original range).
            voxel_center = np.array([0, 0, 0.7])
            pc_center = np.array([0, 0, 0.7])
            if hasattr(self.env, 'table_offset'):
                voxel_center[:2] = self.env.table_offset[:2]
                pc_center = np.array(self.env.table_offset)
                pc_center[2] = pc_center[2] + 0.02
            if env_name.startswith('Kitchen_'):
                self.ws_size = 0.7
                pc_center = np.array(self.env.table_offset)
                pc_center[2] = pc_center[2] + 0.02
            elif env_name.startswith('PickPlace_'):
                pc_center = np.array([0, 0, 0.83])
                self.ws_size = 1.1

            self.voxel_workspace = np.array([
                [voxel_center[0] - self.ws_size/2, voxel_center[0] + self.ws_size/2],
                [voxel_center[1] - self.ws_size/2, voxel_center[1] + self.ws_size/2],
                [voxel_center[2], voxel_center[2] + self.ws_size],
            ])
            self.pc_workspace = np.array([
                [pc_center[0] - self.ws_size/2, pc_center[0] + self.ws_size/2],
                [pc_center[1] - self.ws_size/2, pc_center[1] + self.ws_size/2],
                [pc_center[2], pc_center[2] + self.ws_size],
            ])

        self.obj_pc_size= 512
        self.all_pc_size = 1024
        self.obstacle_pc_size = 1024
        self.d_cams = get_d_cams(env_name)

        if env_name.startswith('Libero_'):
            self.is_libero = True
        else:
            self.is_libero = False

        ## if there is interested objects and segmentation is enabled, output instance pcd.
        if "camera_segmentations" in kwargs and kwargs["camera_segmentations"] == "instance":
            self.interested_objects = get_interested_objects(self.env, env_name)
            assert len(self.interested_objects) > 0, f"no interested objects found for environment {env_name}"
            self.output_instance_pcd = True
        else:
            self.output_instance_pcd = False



    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state and not self.is_libero:
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise tget_d_camsd_camshe observations will be garbage
            return self.get_observation()
        return None

    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_instance_pcd(self, di, require_downsample = True):
        instance_pcds = {k:o3d.geometry.PointCloud() for k in self.interested_objects}
        visible_dicts = {f'{k}_visible': True for k in self.interested_objects}
        name2id = get_name2id(self.env)
        for cam_idx, camera_name in enumerate(self.env.camera_names):
            if camera_name not in self.d_cams:
                continue
            # if not is_cam_used(self.env, camera_name):
            #     continue
            
            cam_height = self.env.camera_heights[cam_idx]
            cam_width = self.env.camera_widths[cam_idx]
            ext_mat = get_camera_extrinsic_matrix(self.env.sim, camera_name)
            int_mat = get_camera_intrinsic_matrix(self.env.sim, camera_name, cam_height, cam_width)
            cam_param = [int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2]]
            depth = di[f'{camera_name}_depth'][::-1]
            depth = np.clip(depth, 0, 1)
            depth = get_real_depth_map(self.env.sim, depth)
            depth = depth[:, :, 0]
            color = di[f'{camera_name}_image'][::-1]

            for obj_name in self.interested_objects:
                seg_id = name2id[obj_name]
                if seg_id is None:
                    continue
                binary_mask = (di[f"{camera_name}_segmentation_instance"][:, :, 0] == seg_id).astype(np.uint8)
                # Flip binary_mask to match OpenCV convention (consistent with flipped RGB/depth)
                binary_mask = binary_mask[::-1]  # Flip vertically                     
                # rgb_mask = np.stack([binary_mask]*3, axis=-1)
                obj_pcd_c =  depth2fgpcd(depth, binary_mask, cam_param)

                obj_pcd_w = ext_mat @ np.concatenate([obj_pcd_c.T, np.ones((1, obj_pcd_c.shape[0]))], axis=0)
                obj_pcd_w = obj_pcd_w[:3, :].T

                masked_indices = np.where(binary_mask > 0)
                masked_color = color[masked_indices].astype(np.float64) / 255
                # masked_color_raw = (color*rgb_mask).reshape(-1, 3).astype(np.float64) / 255
                # masked_color = masked_color_raw[masked_color_raw[:, 0] > 0]
                obj_pcd_o3d = np2o3d(obj_pcd_w, masked_color)

                instance_pcds[obj_name] += obj_pcd_o3d

        ## process the instance pc, get kdtree
        from scipy.spatial import KDTree
        obj_pc_size = self.obj_pc_size
        pc_instance_dict = {}
        kdtree_instance_dict = {}
        for obj_name, obj_pcd_raw in instance_pcds.items():

            ## filter pc
            obj_pcd_rad, ind_rad = obj_pcd_raw.remove_radius_outlier(nb_points=10, radius=0.05)
            obj_pcd, ind = obj_pcd_rad.remove_statistical_outlier(nb_neighbors=5, std_ratio=7.0)
            # o3d.io.write_point_cloud(f'{obj_name}_128.ply', obj_pcd)

            # Visibility floor is an ABSOLUTE point count, not a fraction of the sample
            # target obj_pc_size: a small/flat object (e.g. butter) yields only ~30 real
            # points at this camera resolution, which is enough to attempt a grasp but
            # well under obj_pc_size*0.1. Below this floor the cloud is too sparse to
            # trust, so it is zeroed out and flagged invisible.
            min_visible_points = 10
            if len(obj_pcd.points) < min_visible_points:
                ## too few real points: zero out and mark invisible
                obj_pcd.points = o3d.utility.Vector3dVector(np.zeros((obj_pc_size,3)))
                obj_pcd.colors = o3d.utility.Vector3dVector(np.zeros((obj_pc_size,3)))
                visible_dicts[f'{obj_name}_visible'] = False
                
            elif len(obj_pcd.points) < obj_pc_size:
                # random upsample to obj_pc_size
                num_pad = obj_pc_size - len(obj_pcd.points)
                indices = np.random.choice(len(obj_pcd.points), num_pad)
                padded_xyz = np.asarray(obj_pcd.points)[indices]
                padded_color = np.asarray(obj_pcd.colors)[indices]
                xyz = np.concatenate([np.asarray(obj_pcd.points), padded_xyz], 0)
                color = np.concatenate([np.asarray(obj_pcd.colors), padded_color], 0)
                obj_pcd = o3d.geometry.PointCloud()
                obj_pcd.points = o3d.utility.Vector3dVector(xyz)
                obj_pcd.colors = o3d.utility.Vector3dVector(color)
            
            elif require_downsample:
                obj_pcd = obj_pcd.farthest_point_down_sample(obj_pc_size)

            obj_xyz = np.asarray(obj_pcd.points)
            obj_color = np.asarray(obj_pcd.colors)
            pc_instance_dict[f'{obj_name}_point_cloud'] = np.concatenate([obj_xyz, obj_color], 1)

            if visible_dicts[f'{obj_name}_visible']:
                kdtree_instance_dict[f'{obj_name}_kdtree'] = KDTree(obj_xyz)
            else:
                kdtree_instance_dict[f'{obj_name}_kdtree'] = None

        return pc_instance_dict, kdtree_instance_dict, visible_dicts

    def in_rbt_body(self, di, point, rbt_radius = 0.2):
        robot_pose = self.env.robots[0].base_pos
        
        if np.linalg.norm(point[:2] - robot_pose[:2]) < rbt_radius:
            return True
        
        eef_pos = di['robot0_eef_pos']
        if np.linalg.norm(point - eef_pos) < 0.1:
            return True
        return False

    def get_obstacle_pcd(self, di, all_points_6d, kdtree_instance_dict):
        # Compute obstacle point cloud by filtering out points that lie in any interested objs
        obstacle_pcd = o3d.geometry.PointCloud()
        
        # Get all points and colors from sampled_pcds
        all_points = all_points_6d[:, :3]  # Extract xyz coordinates
        all_colors = all_points_6d[:, 3:6]  # Extract color information

        ## filter out points that near the robot base and eef
        not_rbt_mask = np.array([not self.in_rbt_body(di, p) for p in all_points])
        all_points = all_points[not_rbt_mask]
        all_colors = all_colors[not_rbt_mask]

        far_mask = np.ones(len(all_points), dtype=bool)
        for obj_kdt_name, instance_kdt in kdtree_instance_dict.items():
            if instance_kdt is None:
                continue
            distances, _ = instance_kdt.query(all_points, distance_upper_bound=0.02)

            # distances < inf means there's a neighbor within 2cm
            mask = distances == np.inf
            far_mask = far_mask & mask  
           
        
        obstacle_points = all_points[far_mask]
        obstacle_colors = all_colors[far_mask]

        if len(obstacle_points) > 0:
            obstacle_pcd.points = o3d.utility.Vector3dVector(np.array(obstacle_points))
            obstacle_pcd.colors = o3d.utility.Vector3dVector(np.array(obstacle_colors))
        else:
            # Create fake points if no obstacle points found
            obstacle_pcd.points = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
            obstacle_pcd.colors = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
        
        if len(obstacle_pcd.points) > self.obstacle_pc_size:
            obstacle_pcd = obstacle_pcd.farthest_point_down_sample(self.obstacle_pc_size)
        elif len(obstacle_pcd.points) < self.obstacle_pc_size* 0.1:
            print(f"Warning: Obstacle point cloud has too few points ({len(obstacle_pcd.points)}), not enough to sample {self.obstacle_pc_size}.")
            return None
        
        obstacle_xyz = np.asarray(obstacle_pcd.points)
        obstacle_color = np.asarray(obstacle_pcd.colors)
        
        return np.concatenate([obstacle_xyz, obstacle_color], 1)
    
    def _point_in_oriented_bbox(self, point, oobb):
        """
        Check if a point lies inside an oriented bounding box.
        
        Args:
            point (np.array): 3D point coordinates
            oobb (o3d.geometry.OrientedBoundingBox): Oriented bounding box
            
        Returns:
            bool: True if point is inside the OOBB, False otherwise
        """
        # Get the center and rotation of the OOBB
        center = oobb.center
        R = oobb.R  # Rotation matrix (3x3)
        extent = oobb.extent  # Half-lengths of the box
        
        # Transform point to OOBB's local coordinate system
        # Subtract center and apply inverse rotation
        local_point = R.T @ (point - center)
        
        # Check if point is within the box bounds in local coordinates
        # extent contains half-lengths, so we check against [-extent, +extent]
        return (np.abs(local_point[0]) <= extent[0] and 
                np.abs(local_point[1]) <= extent[1] and 
                np.abs(local_point[2]) <= extent[2])

    def get_all_pcd(self, di):
        workspace = self.voxel_workspace

        voxel_bound = workspace.T
        voxel_size = 64

        all_pcds = o3d.geometry.PointCloud()
        for cam_idx, camera_name in enumerate(self.env.camera_names):
            cam_height = self.env.camera_heights[cam_idx]
            cam_width = self.env.camera_widths[cam_idx]
            ext_mat = get_camera_extrinsic_matrix(self.env.sim, camera_name)
            int_mat = get_camera_intrinsic_matrix(self.env.sim, camera_name, cam_height, cam_width)
            depth = di[f'{camera_name}_depth'][::-1]
            depth = np.clip(depth, 0, 1)
            depth = get_real_depth_map(self.env.sim, depth)
            depth = depth[:, :, 0]
            color = di[f'{camera_name}_image'][::-1]

            cam_param = [int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2]]
            mask = np.ones_like(depth, dtype=bool)
            pcd = depth2fgpcd(depth, mask, cam_param)

            # pose = np.linalg.inv(ext_mat)
            pose = ext_mat
            
            trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
            trans_pcd = trans_pcd[:3, :].T

            mask = (trans_pcd[:, 0] > workspace[0, 0]) * (trans_pcd[:, 0] < workspace[0, 1]) * (trans_pcd[:, 1] > workspace[1, 0]) * (trans_pcd[:, 1] < workspace[1, 1]) * (trans_pcd[:, 2] > workspace[2, 0]) * (trans_pcd[:, 2] < workspace[2, 1])

            pcd_o3d = np2o3d(trans_pcd[mask], color.reshape(-1, 3)[mask].astype(np.float64) / 255)

            all_pcds += pcd_o3d

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(all_pcds, voxel_size=self.ws_size/voxel_size+1e-4, min_bound=voxel_bound[0], max_bound=voxel_bound[1])
        voxels = voxel_grid.get_voxels()  # returns list of voxels
        if len(voxels) == 0:
            np_voxels = np.zeros([4, voxel_size, voxel_size, voxel_size], dtype=np.uint8)
        else:
            indices = np.stack(list(vx.grid_index for vx in voxels))
            colors = np.stack(list(vx.color for vx in voxels))

            mask = (indices > 0) * (indices < voxel_size)
            indices = indices[mask.all(axis=1)]
            colors = colors[mask.all(axis=1)]

            np_voxels = np.zeros([4, voxel_size, voxel_size, voxel_size], dtype=np.uint8)
            np_voxels[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]] = colors.T * 255

        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.pc_workspace.T[0], self.pc_workspace.T[1])
        cropped_pcd = all_pcds.crop(bounding_box)
        if len(cropped_pcd.points) == 0:
            # create fake points
            cropped_pcd.points = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
            cropped_pcd.colors = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))
        if len(cropped_pcd.points) < self.all_pc_size:
            # random upsample to 1024
            num_pad = self.all_pc_size - len(cropped_pcd.points)
            indices = np.random.choice(len(cropped_pcd.points), num_pad)
            padded_xyz = np.asarray(cropped_pcd.points)[indices]
            padded_color = np.asarray(cropped_pcd.colors)[indices]
            xyz = np.concatenate([np.asarray(cropped_pcd.points), padded_xyz], 0)
            color = np.concatenate([np.asarray(cropped_pcd.colors), padded_color], 0)
            cropped_pcd = o3d.geometry.PointCloud()
            cropped_pcd.points = o3d.utility.Vector3dVector(xyz)
            cropped_pcd.colors = o3d.utility.Vector3dVector(color)
        sampled_pcds = cropped_pcd.farthest_point_down_sample(self.all_pc_size)
        xyz = np.asarray(sampled_pcds.points)
        color = np.asarray(sampled_pcds.colors)

        return {'voxels': np_voxels, 'point_cloud': np.concatenate([xyz, color], 1)}

    def get_instance_and_obstacles_pcd(self, di, require_downsample=True, get_obstacle = True):
        pc_instance_dict, kdtree_instance_dict, visible_dicts = self.get_instance_pcd(di, require_downsample=require_downsample)
        output_dict = pc_instance_dict.copy()

        if get_obstacle:
            all_pcd_voxels = self.get_all_pcd(di)

            obstacle_pcd = self.get_obstacle_pcd(di, all_pcd_voxels['point_cloud'], kdtree_instance_dict)
            if obstacle_pcd is not None:
                output_dict['obstacle_pcd'] = obstacle_pcd
        return output_dict, visible_dicts

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                ret[k] = di[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
                depth_map = di[k][::-1]
                depth_map = np.clip(depth_map, 0, 1)
                ret[k] = get_real_depth_map(self.env.sim, depth_map)
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # "object" key contains object information
        if "object-state" in di:
            ret["object"] = np.array(di["object-state"])

        if self.env.use_camera_obs:

            if self.output_instance_pcd:
                pc_instance_dict, _, visible_dict = self.get_instance_pcd(di)
                ret.update(pc_instance_dict)
                ret.update(visible_dict)


            ## TODO: output obstacle pcds (all pcds that does not lie in OOBB of interested objects, and below gripper range)
            if self.output_all_pcds:
                all_pcd_voxels = self.get_all_pcd(di)
                ret.update(all_pcd_voxels)

                # obstacle_pcd = self.get_obstacle_pcd(all_pcd_voxels['point_cloud'], oobb_instance_dict)
                # ret['obstacle_pcd'] = obstacle_pcd

        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and \
                            (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])
        ## note that segmentation is not in modalities. And robot0_segemntation is added to obs as k.startswith(pf)
        if self.output_instance_pcd:
            ret['agentview_segmentation_instance'] = di['agentview_segmentation_instance']
        return ret

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return self.get_observation(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_spec[0].shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ROBOSUITE_TYPE

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return robosuite.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
            depth_modalities = ["{}_depth".format(cn) for cn in camera_names]
        elif has_camera:
            # v0.3 only had support for one image, and it was named "rgb"
            assert len(image_modalities) == 1
            image_modalities = ["rgb"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
                "depth": depth_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=has_camera, 
            use_image_obs=has_camera, 
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return tuple(MUJOCO_EXCEPTIONS)

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
