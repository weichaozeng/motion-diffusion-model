import sys
from pathlib import Path
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent
sys.path.append(str(root_dir))
import torch
import numpy as np
from utils_hand.misc import to_torch
import utils_hand.rotation_conversions as geometry
import random
import pickle
import os
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=150, sampling="conseq", sampling_step=1, split="train", pose_rep="rot6d", translation=True, glob=True, align_pose_frontview=True, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.sampling = sampling 
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.align_pose_frontview = align_pose_frontview

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

        # 1. pose
        if getattr(self, "_load_rotvec_x", None) is None or getattr(self, "_load_rotvec_y", None) is None :
            raise ValueError("This representation is not possible.")
        else:
            # x
            x_pose, x_beta = self._load_rotvec_x(ind, frame_ix, x_data, is_right)
            x_pose = to_torch(x_pose)
            x_raw_root_rotmat = geometry.axis_angle_to_matrix(x_pose[:, 0, :])
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

            # y
            y_pose, inpaint_mask, R_c2w = self._load_rotvec_y(ind, frame_ix, y_data, cam)
            y_pose = to_torch(y_pose)
            R_c2w = to_torch(R_c2w)
            if not self.glob:
                y_pose = y_pose[:, 1:, :]
            y_pose = to_torch(y_pose)
            R_c2w = to_torch(R_c2w)
            if self.align_pose_frontview:
                y_first_frame_root_pose_matrix = geometry.axis_angle_to_matrix(y_pose[0][0])
                y_all_root_poses_matrix = geometry.axis_angle_to_matrix(y_pose[:, 0, :])
                y_aligned_root_poses_matrix = torch.matmul(torch.transpose(y_first_frame_root_pose_matrix, 0, 1), y_all_root_poses_matrix)
                y_pose[:, 0, :] = geometry.matrix_to_axis_angle(y_aligned_root_poses_matrix)
            else:
                y_first_frame_root_pose_matrix = torch.eye(3).float()


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
        
        # 2. trans
        if self.translation:
            J0_offset = torch.tensor([0.0957, 0.0064, 0.0062], device=x_pose.device, dtype=x_pose.dtype)
            # x
            if getattr(self, "_load_translation_x") is None:
                raise ValueError("Can't extract translations x.")
            x_trans = to_torch(self._load_translation_x(ind, frame_ix, x_data, is_right))
            # x_wrist = x_trans + Rot_J0
            rot_J0_x = torch.matmul(x_raw_root_rotmat, J0_offset.unsqueeze(-1)).squeeze(-1)
            x_wrist_world = x_trans + rot_J0_x
            x_orig_root = x_wrist_world[0].clone()
            x_trans = x_wrist_world - x_wrist_world[0]
            if self.align_pose_frontview:
                x_trans = torch.matmul(x_trans, x_first_frame_root_pose_matrix)
            # y
            if getattr(self, "_load_translation_y") is None:
                raise ValueError("Can't extract translations y.")
            y_trans_cam, _= self._load_translation_y(ind, frame_ix, y_data, cam)
            # y_wrist = R_c2w * (y_trans_cam + J0) - R_c2w * T_w2c
            y_wrist_cam = y_trans_cam + J0_offset
            # cam2world      
            R_w2c = to_torch(cam['R'][0]) # [3, 3]
            T_w2c = to_torch(cam['T'][0]) # [3]
            C_world = -torch.matmul(R_w2c.t(), T_w2c)
            y_wrist_world = torch.matmul(R_c2w.unsqueeze(0), y_wrist_cam.unsqueeze(-1)).squeeze(-1) + C_world.unsqueeze(0)
            y_orig_root = to_torch(y_wrist_world[0]).clone()
            y_trans = to_torch(y_wrist_world - y_wrist_world[0])
            if self.align_pose_frontview:
                y_trans = torch.matmul(y_trans, y_first_frame_root_pose_matrix)
        else:
            x_trans = torch.zeros((x_pose.shape[0], 3), dtype=x_pose.dtype)
            x_orig_root = x_trans[0].clone()
            y_trans = torch.zeros((y_pose.shape[0], 3), dtype=y_pose.dtype)
            y_orig_root = y_trans[0].clone()
        

        # print(f"Movement distance (Meters): X_GT={x_trans[-1].norm():.3f}, Y_HaMeR={y_trans[-1].norm():.3f}")
        # p0_gt = x_orig_root  
        # p0_pred = y_orig_root  
        # bias = p0_gt - p0_pred
        # print(f"Constant Bias (meters): {bias}")

        # 3. padding pose with trans
        # x
        x_padded_tr = torch.zeros((x_pose.shape[0], x_pose.shape[2]), dtype=x_pose.dtype)
        x_padded_tr[:, :3] = x_trans
        x_ret = torch.cat((x_pose, x_padded_tr[:, None]), 1)
        # y
        y_padded_tr = torch.zeros((y_pose.shape[0], y_pose.shape[2]), dtype=y_pose.dtype)
        y_padded_tr[:, :3] = y_trans
        y_ret = torch.cat((y_pose, y_padded_tr[:, None]), 1)

        # 4. permute        
        x_pose = x_pose.permute(1, 2, 0).contiguous()     # J, 6, T
        x_trans = x_trans.permute(1, 0).contiguous()      # 3, T
        x_ret = x_ret.permute(1, 2, 0).contiguous()       # J+1, 6, T
        y_pose = y_pose.permute(1, 2, 0).contiguous()     # J, 6, T
        y_trans = y_trans.permute(1, 0).contiguous()      # 3, T
        y_ret = y_ret.permute(1, 2, 0).contiguous()       # J+1, 6, T
        inpaint_mask = torch.from_numpy(inpaint_mask)

        # extra. pose 2d
        y_2d = self._load_2d_pose_y(ind, frame_ix, y_data)
        y_2d = y_2d.permute(1, 2, 0).contiguous()         # 21, 3, T

        # ==========================================================
        # [NEW] 速度与姿态异常检测 & 缺失/异常帧强制置零
        # ==========================================================
        valid_idx = torch.where(inpaint_mask > 0)[0]
        
        # 将时间维度提到最前面，方便按帧切片
        y_trans_t = y_trans.permute(1, 0)       # [T, 3]
        y_pose_t = y_pose.permute(2, 0, 1)      # [T, J, 6]
        
        if len(valid_idx) > 1:
            valid_trans = y_trans_t[valid_idx]
            valid_pose = y_pose_t[valid_idx]
            
            # 1. 计算平移差异 (L2 Norm)
            trans_dists = torch.norm(valid_trans[1:] - valid_trans[:-1], dim=-1) # [N-1]
            
            # 2. 计算姿态差异 (使用 6D 表征的 L2 距离作为快速近似)
            # 先计算每个关节的 6D 差异 [N-1, J]，再取当前帧所有关节中最大的跳变幅度 [N-1]
            pose_diffs = torch.norm(valid_pose[1:] - valid_pose[:-1], dim=-1)
            max_pose_dists = torch.max(pose_diffs, dim=-1)[0]
            
            # 设置物理极限阈值 (需要根据你的数据集尺度微调)
            velocity_threshold = 0.10  # 平移阈值 (例: 0.1 米)
            pose_threshold = 0.5       # 姿态阈值 (经验值: 6D向量的L2突变超过0.5通常意味着严重的关节翻折)
            
            # 找出平移或姿态超过阈值的异常跳变索引 (取并集)
            outlier_local_idx = torch.where(
                (trans_dists > velocity_threshold) | (max_pose_dists > pose_threshold)
            )[0]
            
            for local_idx in outlier_local_idx:
                # 将发生跳变的两帧都标记为无效 (交由模型去 Inpaint 脑补)
                idx_a = valid_idx[local_idx]
                idx_b = valid_idx[local_idx + 1]
                inpaint_mask[idx_a] = 0
                inpaint_mask[idx_b] = 0
                
        # 基于更新后的 mask，将 y_init 相关的所有缺失帧和异常帧的特征暴力清零
        mask_bool = (inpaint_mask > 0).view(-1)
        y_trans[:, ~mask_bool] = 0.0
        y_pose[:, :, ~mask_bool] = 0.0
        y_ret[:, :, ~mask_bool] = 0.0
        y_2d[:, :, ~mask_bool] = 0.0
        # ==========================================================
                
        # 5, return
        data_dict = {
            'inpaint_mask': inpaint_mask.float(),
            # x
            'x_ret': x_ret.float(),
            'x_pose': x_pose.float(),
            'x_beta': x_beta.float(),
            'x_root_trans': x_orig_root.float(),
            'x_trans': x_trans.float(),
            'x_ff_root_orient_rotmat': x_first_frame_root_pose_matrix.float(),
            # y
            'y_ret': y_ret.float(),
            'y_pose': y_pose.float(),
            'y_root_trans': y_orig_root.float(),
            'y_trans': y_trans.float(),
            'y_ff_root_orient_rotmat': y_first_frame_root_pose_matrix.float(),
            # corr
            'R_c2w': R_c2w.float(),
            'C_world': C_world.float(),
            # extra
            'y_2d': y_2d.float(),
        }

        return data_dict

    def _get_item_data_index(self, data_index):
        # skip left
        while True:
            seq_y = self.seqs_y[data_index]
            with open(seq_y, 'rb') as f:
                temp_data = pickle.load(f)
            if temp_data["handedness"][0] != 0 and len(temp_data['frame_indices']) >= 10:
                y_data = temp_data
                break
            if self.split == 'train':
                data_index = random.choice(self._train)
            elif self.split == 'val':
                data_index = random.choice(self._val)
            else:
                raise NotImplementedError()
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
            'video_path': str(self.seqs_video[data_index]),
            'frame_indices': frame_indices,
            'inpaint_mask': data_dict['inpaint_mask'], 
            'suffix_mask': suffix_mask, 
            'is_right': is_right, 
            'cam': cam, 
            # x
            'x_ret': data_dict['x_ret'],
            'x_pose': data_dict['x_pose'], 
            'x_beta': data_dict['x_beta'], 
            'x_trans': data_dict['x_trans'],
            'x_root_trans': data_dict['x_root_trans'], 
            'x_ff_root_orient_rotmat': data_dict['x_ff_root_orient_rotmat'], 
            # y
            'y_ret': data_dict['y_ret'],
            'y_pose': data_dict['y_pose'], 
            'y_trans': data_dict['y_trans'],
            'y_root_trans': data_dict['y_root_trans'], 
            'y_ff_root_orient_rotmat': data_dict['y_ff_root_orient_rotmat'],
            # corr
            'R_c2w': data_dict['R_c2w'].float(),
            'C_world': data_dict['C_world'].float(),
            'y_2d': data_dict['y_2d'],
        }
        
        return output
    
    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train), num_seq_max)
        elif self.split == 'val':
            return min(len(self._val), num_seq_max)
        else:
            raise NotImplementedError()

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