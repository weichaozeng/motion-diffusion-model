import torch
import utils_hand.rotation_conversions as geometry
from model_hand.smpl import MANO

class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.hand_model = MANO(device=device, num_pca_comps=6, use_pca=False, use_flat_mean=False,).eval().to(device)

    def __call__(self, pose, pose_rep, beta, translation=None, ff_rotmat=None, return_vertices=False, **kwargs):

        x_rotations = pose

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations)
        elif pose_rep == "rotmat":
            rotations = x_rotations.view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations)
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations)
        else:
            raise NotImplementedError("No geometry for this one.")

        if ff_rotmat is not None:
            all_root_pose_mat = rotations[:, 0]
            all_root_pose_mat = torch.matmul(ff_rotmat, all_root_pose_mat)
            rotations[:, 0] = all_root_pose_mat


        global_orient = rotations[:, 0]
        rotations = rotations[:, 1:]

            # import ipdb; ipdb.set_trace()
        vertices, joints = self.hand_model(pose=rotations, Rh=global_orient, shapes=beta, Th=translation)

    
        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=pose.device, dtype=pose.dtype)
        x_xyz[:] = joints
        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        if return_vertices:
            return x_xyz, vertices
        else:
            return x_xyz
