import sys
from pathlib import Path
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent.parent 
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import numpy as np
import pickle
import decord
from typing import Optional
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import MANOOutput, to_tensor
from smplx.vertex_ids import vertex_ids
import json
import utils.rotation_conversions as geometry


class MANO(smplx.MANOLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official MANO implementation to support more joints.
        Args:
            Same as MANOLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(MANO, self).__init__(*args, **kwargs)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        #2, 3, 5, 4, 1
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('extra_joints_idxs', to_tensor(list(vertex_ids['mano'].values()), dtype=torch.long))
        self.register_buffer('joint_map', torch.tensor(mano_to_openpose, dtype=torch.long))
        self.pose2rot = kwargs.get('pose2rot')

    def forward(self, *args, **kwargs) -> MANOOutput:
        """
        Run forward pass. Same as MANO and also append an extra set of joints if joint_regressor_extra is specified.
        """
        mano_output = super(MANO, self).forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output


if __name__ == "__main__":
    mano_cfg = {
        'data_dir': '/home/zvc/Project/VHand/_DATA/data/',
        'model_path': '/home/zvc/Project/VHand/_DATA/data/mano',
        'gender': 'neutral',
        'num_hand_joints': '15',
        'mean_params': 'data/mano_mean_params.npz',
        'create_body_pose': False,
    }
    mano = MANO(pose2rot=False, **mano_cfg)

    data_path = '/home/zvc/Project/VHand/test_dataset/GigaHands/vhand/hamer_out/p001-folder_017_brics-odroid-002_cam0/results/track_500.0/track_2.pkl'
    mano_path = '/home/zvc/Data/GigaHands/hand_poses/p001-folder/params/017.json'
    video_path = '/home/zvc/Data/GigaHands/symlinks/p001-folder_017_brics-odroid-002_cam0/brics-odroid-002_cam0.mp4'
    # frame_indices = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

    
    with open(data_path, 'rb') as f:
        y_data = pickle.load(f)
    with open(mano_path, 'r') as f:
        x_data = json.load(f)

    frame_indices = y_data['frame_indices']
    start_idx = frame_indices[0]
    end_idx = frame_indices[-1]

    # y
    y_betas = torch.from_numpy(np.asarray([mano['betas'] for mano in y_data['mano']])).mean(dim=0).reshape(-1, 10)
    y_hand_pose = torch.from_numpy(np.asarray([mano['hand_pose'] for mano in y_data['mano']]))
    y_global_orient = torch.from_numpy(np.asarray([mano['global_orient'] for mano in y_data['mano']]))
    y_pose_rotmat = torch.cat([y_global_orient, y_hand_pose], dim=1) # (N, 16, 3, 3)
    y_pose_rotvec = geometry.matrix_to_axis_angle(y_pose_rotmat) # (N, 16, 3)

    # x
    x_betas = torch.tensor(x_data["right"]["shapes"], dtype=torch.float32).reshape(-1, 10)
    x_poses = torch.tensor(x_data["right"]["poses"], dtype=torch.float32)[frame_indices].reshape(-1, 16, 3)
    x_Rh = torch.tensor(x_data["right"]["Rh"], dtype=torch.float32)[frame_indices].reshape(-1, 1, 3)
    x_pose_rotvec = torch.cat([x_Rh, x_poses[:, 1:]], dim=1) # (N, 16, 3, 3)
    x_pose_rotmat = geometry.axis_angle_to_matrix(x_pose_rotvec) # (N, 16, 3)

    
    # y Canonical on first frame
    y_pose_rotvec_can = y_pose_rotvec.clone()
    first_frame_root_pose_matrix_y = geometry.axis_angle_to_matrix(y_pose_rotvec_can[0][0])
    all_root_poses_matrix_y = geometry.axis_angle_to_matrix(y_pose_rotvec_can[:, 0, :])
    aligned_root_poses_matrix_y = torch.matmul(torch.transpose(first_frame_root_pose_matrix_y, 0, 1), all_root_poses_matrix_y)
    y_pose_rotvec_can[:, 0, :] = geometry.matrix_to_axis_angle(aligned_root_poses_matrix_y)
    y_pose_rotmat_can = geometry.axis_angle_to_matrix(y_pose_rotvec_can)
    y_pose_rot6d_can = geometry.matrix_to_rotation_6d(y_pose_rotmat_can)
    


    # x Canonical on first frame
    x_pose_rotvec_can = x_pose_rotvec.clone()
    first_frame_root_pose_matrix_x = geometry.axis_angle_to_matrix(x_pose_rotvec_can[0][0])
    all_root_poses_matrix_x = geometry.axis_angle_to_matrix(x_pose_rotvec_can[:, 0, :])
    aligned_root_poses_matrix_x= torch.matmul(torch.transpose(first_frame_root_pose_matrix_x, 0, 1), all_root_poses_matrix_x)
    x_pose_rotvec_can[:, 0, :] = geometry.matrix_to_axis_angle(aligned_root_poses_matrix_x)
    x_pose_rotmat_can = geometry.axis_angle_to_matrix(x_pose_rotvec_can)
    x_pose_rot6d_can = geometry.matrix_to_rotation_6d(x_pose_rotmat_can)

    
    # y mano rec
    y_input = {
        'global_orient': y_pose_rotmat_can[:, 0],
        'hand_pose': y_pose_rotmat_can[:, 1:],
        'betas': y_betas,
    }




