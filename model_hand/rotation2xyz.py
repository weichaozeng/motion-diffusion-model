import torch
import utils_hand.rotation_conversions as geometry
from model_hand.smpl import MANO

class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.hand_model = MANO(device=device, num_pca_comps=6, use_pca=False, use_flat_mean=False,).eval().to(device)

    def __call__(self, pose, pose_rep, beta, translation=None, root_translation=None, ff_rotmat=None, return_vertices=False, R_cam2world=None, R_adj=None, **kwargs):

        x_rotations = pose

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, nframes, njoints, feats = x_rotations.shape

        # Compute rotations
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations)
        elif pose_rep == "rotmat":
            rotations = x_rotations.view(nsamples, nframes, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations)
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations)
        else:
            raise NotImplementedError("No geometry for this one.")

        if len(ff_rotmat.shape) == 3:
            ff_rotmat = ff_rotmat.unsqueeze(1).repeat(1, nframes, 1, 1)

        # pose
        if ff_rotmat is not None:
            all_root_pose_mat = rotations[:, :, 0]
            all_root_pose_mat = torch.matmul(ff_rotmat, all_root_pose_mat)
            if R_cam2world is not None and R_adj is not None:
                # R_total_T (B, F, 3, 3) = (R_c2w @ R_adj)^T = R_adj^T @ R_c2w^T
                if len(R_cam2world.shape) == 3:
                    R_cam2world = R_cam2world.unsqueeze(1).repeat(1, nframes, 1, 1)
                # R_adj (B, F, 3, 3)
                R_total = torch.matmul(R_cam2world, R_adj)
                R_total_inv = R_total.transpose(-1, -2) 
                all_root_pose_mat = torch.matmul(R_total_inv, all_root_pose_mat)
            rotations[:, :, 0] = all_root_pose_mat
        B, F, J, _, _ = rotations.shape
        rotmat_flat = rotations.view(-1, J, 3, 3)
        rotvec_flat = geometry.matrix_to_axis_angle(rotmat_flat)

        global_orient = rotvec_flat[:, 0, :].reshape(-1, 3)
        hand_pose = rotvec_flat[:, 1:, :].reshape(-1, 45)
        hand_pose = torch.cat([torch.zeros_like(global_orient), hand_pose], dim=1)


        # trans
        if translation is not None:
            if ff_rotmat is not None:
                translation_world_rel = torch.matmul(ff_rotmat, translation.unsqueeze(-1)).squeeze(-1)
                translation_world = translation_world_rel + root_translation.unsqueeze(1)
                if R_cam2world is not None and R_adj is not None:
                    translation = torch.matmul(R_total_inv, translation_world_rel.unsqueeze(-1)).squeeze(-1)
                else:
                    translation = translation_world
            translation = translation.reshape(-1, 3)

        # shapes
        shapes = beta.unsqueeze(1).repeat(1, F, 1).view(-1, 10)
        
        # import ipdb; ipdb.set_trace()
        vertices, joints = self.hand_model(poses=hand_pose, Rh=global_orient, Th=translation, shapes=shapes, pose2rot=True)

        vertices = vertices.view(B, F, -1, 3)
        joints = joints.view(B, F, -1, 3)
        joints = joints.permute(0, 2, 3, 1).contiguous()

        if return_vertices:
            return joints, vertices
        else:
            return joints
