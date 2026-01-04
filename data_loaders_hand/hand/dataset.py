import sys
from pathlib import Path
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent
sys.path.append(str(root_dir))
import torch
import numpy as np
from utils.misc import to_torch
import utils.rotation_conversions as geometry
import random
import pickle
import os
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=150, sampling="conseq", sampling_step=1, split="train", pose_rep="rot6d", translation=False, glob=True, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.sampling = sampling 
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob

        self.align_pose_frontview = kwargs.get('align_pose_frontview', False)

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"{self.split} is not a valid split")
        
        # to remove shuffling
        self._original_train = None
        self._original_test = None

    # def get_pose_data(self, data_index, frame_ix, is_right, y_data, mano_data):
    #     pose, beta, ref_motion, inpain_mask, orig_root, orig_root_y,first_frame_root_pose_matrix, first_frame_root_pose_matrix_y, trans, ref_trans  = self._load(data_index, frame_ix, is_right, y_data, mano_data)
    #     return pose, beta, ref_motion, inpain_mask, orig_root, orig_root_y,first_frame_root_pose_matrix, first_frame_root_pose_matrix_y, trans, ref_trans
    
    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        elif self.split == 'val':
            data_index = self._val[index]
        return self._get_item_data_index(data_index)
    
    def _load(self, ind, frame_ix, is_right, y_data, x_data, cam):
        pose_rep = self.pose_rep
        assert pose_rep != "xyz"

        if getattr(self, "_load_rotvec_x", None) is None or getattr(self, "_load_rotvec_y", None) is None :
            raise ValueError("This representation is not possible.")
        else:
            # x
            x_pose, x_beta = self._load_rotvec_x(ind, frame_ix, x_data, is_right)
            if not self.glob:
                x_pose = x_pose[:, 1:, :]
            x_pose = to_torch(x_pose)
            if self.align_pose_frontview:
                x_first_frame_root_pose_matrix = geometry.axis_angle_to_matrix(x_pose[0][0])
                x_all_root_poses_matrix = geometry.axis_angle_to_matrix(x_pose[:, 0, :])
                x_aligned_root_poses_matrix = torch.matmul(torch.transpose(x_first_frame_root_pose_matrix, 0, 1), x_all_root_poses_matrix)
                x_pose[:, 0, :] = geometry.matrix_to_axis_angle(x_aligned_root_poses_matrix)
            else:
                x_first_frame_root_pose_matrix = torch.eye(3).float()
                # if self.translation:
                #     ret_tr = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1).float(),
                #                         torch.transpose(ret_tr, 0, 1))
                #     ret_tr = torch.transpose(ret_tr, 0, 1)
            # y
            y_pose, inpaint_mask, R_c2w, R_adj = self._load_rotvec_y(ind, frame_ix, y_data, cam)
            if not self.glob:
                y_pose = y_pose[:, 1:, :]
            y_pose = to_torch(y_pose)
            if self.align_pose_frontview:
                y_first_frame_root_pose_matrix = geometry.axis_angle_to_matrix(y_pose[0][0])
                y_all_root_poses_matrix = geometry.axis_angle_to_matrix(y_pose[:, 0, :])
                y_aligned_root_poses_matrix = torch.matmul(torch.transpose(y_first_frame_root_pose_matrix, 0, 1), y_all_root_poses_matrix)
                y_pose[:, 0, :] = geometry.matrix_to_axis_angle(y_aligned_root_poses_matrix)
            else:
                y_first_frame_root_pose_matrix = torch.eye(3).float()
                # if self.translation:
                #     ret_tr_y = torch.matmul(torch.transpose(first_frame_root_pose_matrix_y, 0, 1).float(),
                #                         torch.transpose(ret_tr_y, 0, 1))
                #     ret_tr_y = torch.transpose(ret_tr_y, 0, 1)

            if pose_rep == "rotvec":
                x_pose = x_pose
                y_pose = y_pose
            elif pose_rep == "rotmat":
                x_pose = geometry.axis_angle_to_matrix(x_pose).view(*x_pose.shape[:2], 9)
                y_pose = geometry.axis_angle_to_matrix(y_pose).view(*y_pose.shape[:2], 9)
            elif pose_rep == "rotquat":
                x_pose = geometry.axis_angle_to_quaternion(x_pose)
                y_pose = geometry.axis_angle_to_quaternion(y_pose)
            elif pose_rep == "rot6d":
                x_pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x_pose))
                y_pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(y_pose))
        
        if self.translation:
            # x
            if getattr(self, "_load_translation_x") is None:
                raise ValueError("Can't extract translations x.")
            x_trans = self._load_translation_x(ind, frame_ix, x_data, is_right)
            x_orig_root = to_torch(x_trans[0]).clone()
            x_trans = to_torch(x_trans - x_trans[0])
            # y
            if getattr(self, "_load_translation_y") is None:
                raise ValueError("Can't extract translations y.")
            y_trans, _= self._load_translation_y(ind, frame_ix, y_data)
            y_orig_root = to_torch(y_trans[0]).clone()
            y_trans = to_torch(y_trans - y_trans[0])
        else:
            x_trans = torch.zeros((x_pose.shape[0], 3), dtype=x_pose.dtype)
            x_orig_root = x_trans[0].clone()
            y_trans = torch.zeros((y_pose.shape[0], 3), dtype=y_pose.dtype)
            y_orig_root = y_trans[0].clone()
        # if pose_rep != "xyz" and not self.translation:
        #     ret_tr = torch.zeros((ret.shape[0], 3), dtype=ret.dtype)
        #     # padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
        #     # padded_tr[:, :3] = ret_tr
        #     # ret = torch.cat((ret, padded_tr[:, None]), 1)
        #     # # y 
        #     ret_tr_y = torch.zeros((ret.shape[0], 3), dtype=ret.dtype)
            # padded_tr_y = torch.zeros((ret_y.shape[0], ret_y.shape[2]), dtype=ret_y.dtype)
            # padded_tr_y[:, :3] = ret_tr_y
            # ret_y = torch.cat((ret_y, padded_tr_y[:, None]), 1)
        
        x_pose = x_pose.permute(1, 2, 0).contiguous()         # J, 6, T
        x_trans = x_trans.permute(1, 0).contiguous()      # 3, T
        y_pose = y_pose.permute(1, 2, 0).contiguous()     # J, 6, T
        y_trans = y_trans.permute(1, 0).contiguous()  # 3, T
        inpaint_mask = torch.from_numpy(inpaint_mask)
        
        data_dict = {
            'inpaint_mask': inpaint_mask.float(),
            # x
            'x_pose': x_pose.float(),
            'x_beta': x_beta.float(),
            'x_root_trans': x_orig_root.float(),
            'x_trans': x_trans.float(),
            'x_ff_root_orient_rotmat': x_first_frame_root_pose_matrix.float(),
            # y
            'y_pose': x_pose.float(),
            'y_root_trans': y_orig_root.float(),
            'y_trans': y_trans.float(),
            'y_ff_root_orient_rotmat': y_first_frame_root_pose_matrix.float(),
            # corr
            'R_c2w': R_c2w.float(),
            'R_adj': R_adj.float(),
        }

        return data_dict

    def _get_item_data_index(self, data_index):
        # skip left
        while True:
            seq_y = self.seqs_y[data_index]
            with open(seq_y, 'rb') as f:
                temp_data = pickle.load(f)
            if temp_data["handedness"][0] != 0:
                y_data = temp_data
                break
            if self.split == 'train':
                data_index = random.choice(self._train)
            else:
                data_index = random.choice(self._test)
        # cam
        if getattr(self, "_load_cam", None) is None:
            raise ValueError("No cam params.")
        else:
            cam = self._load_cam(data_index)
        # anno
        seq_mano = self.seqs_mano[data_index]
        with open(seq_mano, 'r') as f:
            x_data = json.load(f)

        # frame length
        max_nframe = min(y_data['frame_indices'][-1], len(x_data["right"]["Th"])-1)
        min_nframe = max(0, y_data['frame_indices'][0])
        nframes = max_nframe - min_nframe + 1
        # handedness
        is_right = y_data["handedness"][0]
        is_right = torch.from_numpy(np.asarray(is_right))

        assert self.num_frames > 0
        num_frames = self.num_frames if self.num_frames != -1 else self.max_len
        suffix_mask = torch.ones(num_frames, dtype=torch.bool)

        if num_frames > nframes:
            # adding the last frame until done
            ntoadd = max(0, num_frames - nframes)
            # lastframe = nframes - 1
            lastframe = max_nframe
            padding = lastframe * np.ones(ntoadd, dtype=int)
            frame_indices = np.concatenate((np.arange(min_nframe, max_nframe + 1), padding))
            suffix_mask[:nframes] = False

        elif self.sampling in ["conseq"]:
            step_max = (nframes - 1) // (num_frames - 1)
            if self.sampling == "conseq":
                if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                    step = step_max
                else:
                    step = self.sampling_step
            else:
                raise NotImplementedError()
            lastone = step * (num_frames - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            frame_indices = shift + np.arange(0, lastone + 1, step) + min_nframe
            suffix_mask[:] = False

        data_dict = self._load(data_index, frame_indices, is_right, y_data, x_data, cam)

        output = {
            'name': os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(seq_y)))), 
            'video_path': self.seqs_video[data_index],
            'frame_indices': frame_indices,
            'inpaint_mask': data_dict['inpaint_mask'], 
            'suffix_mask': suffix_mask, 
            'is_right': is_right, 
            'cam': cam, 
            # x
            'x_pose': data_dict['x_pose'], 
            'x_beta': data_dict['x_beta'], 
            'x_trans': data_dict['x_trans'],
            'x_root_trans': data_dict['x_root_trans'], 
            'x_ff_root_orient_rotmat': data_dict['x_ff_root_orient_rotmat'], 
            # y
            'y_pose': data_dict['y_pose'], 
            'y_trans': data_dict['y_trans'],
            'y_root_trans': data_dict['y_root_trans'], 
            'y_ff_root_orient_rotmat': data_dict['y_ff_root_orient_rotmat'],
        }
        
        return output
    
    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train), num_seq_max)
        else:
            return min(len(self._test), num_seq_max)

    def shuffle(self):
        if self.split == 'train':
            random.shuffle(self._train)
        else:
            random.shuffle(self._test)

    def reset_shuffle(self):
        if self.split == 'train':
            if self._original_train is None:
                self._original_train = self._train
            else:
                self._train = self._original_train
        else:
            if self._original_test is None:
                self._original_test = self._test
            else:
                self._test = self._original_test