import torch
import utils_hand.rotation_conversions as geometry
from model_hand.smpl import MANO

class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.hand_model = MANO(device=device).eval().to(device)

    def __call__(self, pose, pose_rep, beta, glob, translation=None, 
                 glob_rot=None, return_vertices=False, **kwargs):



        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")
        

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

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
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
