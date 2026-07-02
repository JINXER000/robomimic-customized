


import time

import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import torch
from argparse import Namespace
from scipy.spatial.transform import Rotation as R

class RerunLogger():
    def __init__(self, log_name="libero_env", intrinsic=None, extrinsic=None):

        self.log_name = log_name
        self.frame_count = 0
        # Store trajectory points for each robot
        self.trajectories = {}


        # Create default args for rr.script_setup
        default_args = Namespace(
            stdout=False,
            serve=False,
            connect=False,
            url=None,
            save=None,
            headless=False
        )

        # Define the primary camera entity path
        self.primary_camera_entity = "world/agentview_image"

        if extrinsic is not None and intrinsic is not None:

            blueprint = rrb.Horizontal(
                rrb.Spatial3DView(name="3D"),
                rrb.Vertical(
                    rrb.Tabs(
                        # Note that we re-project the annotations into the 2D views:
                        # For this to work, the origin of the 2D views has to be a pinhole camera,
                        # this way the viewer knows how to project the 3D annotations into the 2D views.
                        rrb.Spatial2DView(
                            name="BGR",
                            origin=self.primary_camera_entity,  # Use the camera entity as origin
                            contents=["$origin/bgr", "/world/annotations/**"],  # Reference annotations from world
                        ),
                        name="2D",
                    ),
                    rrb.TextDocumentView(name="Readme"),
                    row_shares=[2, 1],
                ),
            )

            # rr.init("libero_env", spawn=True)
            rr.script_setup(default_args, log_name, default_blueprint=blueprint)

            # Set the world coordinate system
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
            # Log a transform; the viewer will render an axis gizmo of the given length
            rr.log("world/origin", rr.Transform3D(axis_length=10.0))  # 1 meter

            # Set up the camera entity with a transform
            # Convert rotation matrix to quaternion and ensure it's in the right format
            ext_quat = R.from_matrix(extrinsic[:3, :3]).as_quat()
            # Convert to list to avoid numpy array compatibility issues
            ext_quat_list = ext_quat.tolist()
            rr.log(self.primary_camera_entity, rr.Transform3D(translation=extrinsic[:3, 3], rotation=rr.Quaternion(xyzw=ext_quat_list)), static=True)
            # You'll need to set the camera intrinsics when you have them
            rr.log(self.primary_camera_entity, rr.Pinhole(image_from_camera=intrinsic, resolution=[84, 84]))
        else:
            # Initialize without camera setup
            rr.script_setup(default_args, log_name)
        
    def set_frame_time(self):
        """Set the time for the current frame."""
        rr.set_time_seconds("log_time", time.time())
        rr.set_time_sequence("frame", self.frame_count)
        # rr.set_time('frame', self.frame_count)
        # Log time explicitly
        # rr.log("world/time", rr.Scalar(self.frame_count)) 
    
    def update_eef_pose(self, robot_name, eef_xyz, eef_quat):
        """Update the end-effector pose in Rerun as a transform."""

        eef_quat_tensor = torch.tensor(eef_quat)
        rr.log(
            f"{robot_name}/eef_pose",
            rr.Transform3D(
                translation=eef_xyz,
                rotation=rr.Quaternion(xyzw=eef_quat_tensor),
                relation=rr.TransformRelation.ChildFromParent,
            ),
        )
        
        # Accumulate trajectory points
        if robot_name not in self.trajectories:
            self.trajectories[robot_name] = []
        
        # Add current position to trajectory
        self.trajectories[robot_name].append(eef_xyz)
        
        # Log the full trajectory so far
        rr.log(
            f"world/annotations/{robot_name}/eef_trajectory",
            rr.Points3D(
                positions=self.trajectories[robot_name],
                colors=[250, 70, 100] * len(self.trajectories[robot_name]),  
                radii=[0.01] * len(self.trajectories[robot_name])  # Same radius for all points
            ),
        )
    
    def log_3d_annotations(self, annotations):
        """Log 3D annotations that will be reprojected to 2D views."""
        for i, annotation in enumerate(annotations):
            rr.log(
                f"world/annotations/point-{i}",
                rr.Points3D(
                    positions=[annotation['position']],
                    colors=[annotation.get('color', [255, 0, 0])],
                    radii=[0.01]
                ),
                static=True,
            )

    def update_img_obs(self, img, img_name):
        """Update image observations in Rerun."""
        if img is not None:
            # Ensure image is in the right format (H, W, C) with uint8 values
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            ## upside-down
            img = img[::-1]
            # Handle different image formats
            if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
                rr.log(
                    # f"{self.primary_camera_entity}/bgr/{img_name}",
                    f"/world/{img_name}/bgr",
                    rr.Image(img),
                )
            elif len(img.shape) == 2:  # Grayscale
                rr.log(
                    f"/world/{img_name}/bgr",
                    rr.Image(img),
                )
            else:
                print(f"Warning: Unsupported image format for {img_name}: shape={img.shape}")

    def update_pc_obs(self, pc, pc_name):
        """Update the point cloud observation in Rerun."""
        if pc is not None and len(pc) > 0:
            rr.log(
                f"{self.log_name}/{pc_name}",
                rr.Points3D(
                    positions=pc,
                    colors=None,
                    radii=0.01
                ),
                
            )
    
    def close(self):
        """Close the rerun logger."""
        rr.close()