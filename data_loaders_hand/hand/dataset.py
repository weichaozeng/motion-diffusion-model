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
    def __init__(self, num_frames=150, sampling_step=1, split="train", pose_rep="rot6d", translation=True, glob=True, align_pose_frontview=True, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.align_pose_frontview = align_pose_frontview

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"{self.split} is not a valid split")
        
    def __len__(self):
        if self.split == 'train':
            return len(self._train)
        elif self.split == 'val':
            return len(self._val)
        else:
            raise NotImplementedError()
        
    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        elif self.split == 'val':
            data_index = self._val[index]
        return self._get_item_data_index(data_index)
    
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

        # random for anno degradation or init prediction
        if random.random() < 0.0: # or self.split != 'train':
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

            else:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                    step = step_max
                else:
                    step = self.sampling_step

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_indices = shift + np.arange(0, lastone + 1, step) + min_nframe
                suffix_mask[:] = False

            data_dict = self._load_x_y(data_index, frame_indices, is_right, y_data, x_data, cam)
        
        else:
            # frame length
            max_nframe = len(x_data["right"]["Th"])-1
            min_nframe = 0
            nframes = max_nframe - min_nframe + 1
            # handedness
            is_right = torch.from_numpy(np.asarray(1))

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

            else:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                    step = step_max
                else:
                    step = self.sampling_step

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_indices = shift + np.arange(0, lastone + 1, step) + min_nframe
                suffix_mask[:] = False
            
            data_dict = self._load_x_x(data_index, frame_indices, is_right, x_data, cam)

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
            'j_2d': data_dict['j_2d'],
        }
        
        return output
    

    def _load_x_y(self, ind, frame_ix, is_right, y_data, x_data, cam):
        pose_rep = self.pose_rep
        assert pose_rep != "xyz"

        # 0. cam
        R_w2c = to_torch(cam['R'][0]) # [3, 3]
        R_w2c = R_w2c[:3, :3]
        R_c2w = R_w2c.t() 
        R_c2w = to_torch(R_c2w)
        T_w2c = to_torch(cam['T'][0]) # [3]
        C_world = -torch.matmul(R_w2c.t(), T_w2c)

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
            y_pose, inpaint_mask = self._load_rotvec_y(ind, frame_ix, y_data, cam)
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
            y_wrist_cam = y_trans_cam + J0_offset
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

        # 3. padding pose with trans
        trans_scale = 15.0
        x_trans = x_trans * trans_scale
        y_trans = y_trans * trans_scale
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
        j_2d = self._load_2d_joint(ind, frame_ix, x_data, cam, is_right)
        j_2d = j_2d.permute(1, 2, 0).contiguous()         # 21, 2, T
        

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
            'j_2d': j_2d.float(),
        }

        return data_dict
    
    def _load_x_x(self, ind, frame_ix, is_right, x_data, cam):
        pose_rep = self.pose_rep
        assert pose_rep != "xyz"

        # 0. cam
        R_w2c = to_torch(cam['R'][0]) # [3, 3]
        R_w2c = R_w2c[:3, :3]
        R_c2w = R_w2c.t() 
        R_c2w = to_torch(R_c2w)
        T_w2c = to_torch(cam['T'][0]) # [3]
        C_world = -torch.matmul(R_w2c.t(), T_w2c)

        # 1. pose
        if getattr(self, "_load_rotvec_x", None) is None:
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

            # x degrade to y
            y_pose = x_pose.clone()
            T_len = y_pose.shape[0]
            inpaint_mask = torch.ones(T_len, dtype=torch.float32)
            occupied = torch.zeros(T_len, dtype=torch.bool)
            occupied[0] = True
            def get_free_segment(min_len, max_len, max_tries=15):
                    for _ in range(max_tries):
                        length = random.randint(min_len, max_len)
                        if T_len <= length: 
                            continue
                        start = random.randint(0, T_len - length)
                        end = start + length
                        # 如果这段区间全是 False (未被占用)
                        if not occupied[start:end].any():
                            occupied[start:end] = True
                            return start, end
                    return None, None
            # Frame Drop
            if random.random() < 1.0:  # 50% 概率触发丢帧
                    num_drops = random.randint(1, 3)
                    for _ in range(num_drops):
                        s, e = get_free_segment(min_len=5, max_len=15)
                        if s is not None:
                            inpaint_mask[s:e] = 0.0
            
            # Pose Jitter
            if random.random() < 0.5:
                num_jitter = random.randint(1, 3) # 采样 1~3 个片段
                for _ in range(num_jitter):
                    s, e = get_free_segment(min_len=2, max_len=5)
                    if s is not None:
                        y_pose[s:e] += torch.randn_like(y_pose[s:e]) * 0.1
            # Hand Flip
            if random.random() < 1.0:  # 40% 概率触发手腕翻转
                    num_flip = random.randint(1, 2)
                    for _ in range(num_flip):
                        s, e = get_free_segment(min_len=3, max_len=8)
                        if s is not None:
                            import math
                            # 获取这段时间 Root 关节的旋转矩阵
                            root_pose = y_pose[s:e, 0, :] # [L, 3]
                            root_mat = geometry.axis_angle_to_matrix(root_pose) # [L, 3, 3]
                            
                            # 随机选择一个轴 (X, Y, 或 Z) 翻转 180 度 (pi)
                            axis_idx = random.choice([0, 1, 2])
                            flip_vec = [0.0, 0.0, 0.0]
                            flip_vec[axis_idx] = math.pi
                            
                            # 生成翻转矩阵并与原本的旋转相乘 (R_new = R_old @ R_flip)
                            flip_axis = torch.tensor(flip_vec, device=y_pose.device)
                            flip_mat = geometry.axis_angle_to_matrix(flip_axis)
                            
                            flipped_mat = torch.matmul(root_mat, flip_mat)
                            # 转回 rotvec 格式写回
                            y_pose[s:e, 0, :] = geometry.matrix_to_axis_angle(flipped_mat)
            
            

            mask_bool_pose = (inpaint_mask > 0).view(T_len, 1, 1)
            y_pose = y_pose * mask_bool_pose
            y_first_frame_root_pose_matrix = x_first_frame_root_pose_matrix.clone()

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
            # x degrade to y
            y_trans = x_trans.clone()
            occupied_trans = torch.zeros(T_len, dtype=torch.bool)
            occupied_trans[0] = True
            def get_free_segment_trans(min_len, max_len, max_tries=15):
                    for _ in range(max_tries):
                        length = random.randint(min_len, max_len)
                        if T_len <= length: 
                            continue
                        start = random.randint(0, T_len - length)
                        end = start + length
                        if not occupied_trans[start:end].any():
                            occupied_trans[start:end] = True
                            return start, end
                    return None, None
            # Local Jitte
            if random.random() < 0.0:  # 60% 概率触发
                    num_jitter = random.randint(1, 3)
                    for _ in range(num_jitter):
                        s, e = get_free_segment_trans(min_len=2, max_len=8)
                        if s is not None:
                            # 局部高频抖动，标准差 1cm (0.01米)
                            y_trans[s:e] += torch.randn_like(y_trans[s:e]) * 0.01
            # Depth Jump
            if random.random() < 0.0:  # 40% 概率触发
                    num_jumps = random.randint(1, 2)
                    for _ in range(num_jumps):
                        s, e = get_free_segment_trans(min_len=5, max_len=15)
                        if s is not None:
                            # a. 在相机坐标系下产生深度 Z 突变 (例如 +/- 5cm)
                            z_shift = (random.random() - 0.5) * 0.10
                            delta_cam = torch.tensor([0.0, 0.0, z_shift], dtype=y_trans.dtype, device=y_trans.device)
                            
                            # b. 转换到世界坐标系 (World Space)
                            # R_c2w 将向量从相机系转到世界系
                            delta_world = torch.matmul(R_c2w, delta_cam)
                            
                            # c. 转换到当前 y_trans 的对齐坐标系 (Aligned Space)
                            # 因为之前的 x_trans 是通过 torch.matmul(x_trans, x_first_frame_root_pose_matrix) 得到的
                            if self.align_pose_frontview:
                                delta_aligned = torch.matmul(delta_world, x_first_frame_root_pose_matrix)
                            else:
                                delta_aligned = delta_world
                                
                            # d. 将符合相机光轴深度的偏移量加到局部的平移序列上
                            y_trans[s:e] += delta_aligned
            mask_bool_trans = (inpaint_mask > 0).view(T_len, 1)
            y_trans = y_trans * mask_bool_trans
            y_orig_root = x_orig_root.clone()

        else:
            x_trans = torch.zeros((x_pose.shape[0], 3), dtype=x_pose.dtype)
            x_orig_root = x_trans[0].clone()
            y_trans = torch.zeros((y_pose.shape[0], 3), dtype=y_pose.dtype)
            y_orig_root = y_trans[0].clone()

        # 3. padding pose with trans
        trans_scale = 15.0
        x_trans = x_trans * trans_scale
        y_trans = y_trans * trans_scale
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
        inpaint_mask = to_torch(inpaint_mask)

        # extra. pose 2d
        j_2d = self._load_2d_joint(ind, frame_ix, x_data, cam, is_right)
        j_2d = j_2d.permute(1, 2, 0).contiguous()         # 21, 2, T
        

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
            'j_2d': j_2d.float(),
        }

        return data_dict
    

    #----------------------------------------------
    # add for statistics
    #----------------------------------------------
    def compute_statistics(self, num_samples=500):
        """
        随机抽样计算数据集中关键张量的均值和标准差，用于指导特征标准化 (Normalization/Scaling)。
        建议在实例化 Dataset 后，手动调用一次此函数查看输出。
        """
        print(f"🚀 开始对 {self.split} 数据集进行 {num_samples} 个样本的均值方差统计...")
        
        # 收集容器
        trans_list = []
        pose_list = []
        j2d_list = []
        
        # 随机采样索引，防止数据完全同质化
        total_len = len(self)
        sample_indices = random.sample(range(total_len), min(num_samples, total_len))
        
        for idx in sample_indices:
            # 直接调用 __getitem__ 获取处理好的完整数据
            data = self[idx]
            
            # 提取 GT 数据 (抛弃时间维度以便全局统计)
            # x_trans: [3, T] -> [3, T]
            trans_list.append(data['x_trans'])
            # x_pose: [J, 6, T] -> [J*6, T]
            pose_list.append(data['x_pose'].view(-1, data['x_pose'].shape[-1]))
            # j_2d: [21, 2, T] -> [42, T]
            j2d_list.append(data['j_2d'].view(-1, data['j_2d'].shape[-1]))

        # 在时间维度上拼接所有样本
        all_trans = torch.cat(trans_list, dim=1)  # [3, N*T]
        all_pose = torch.cat(pose_list, dim=1)    # [J*6, N*T]
        all_j2d = torch.cat(j2d_list, dim=1)      # [42, N*T]

        print("\n" + "="*50)
        print(f"数据集 {self.split} 统计结果 (基于 {num_samples} 个视频序列)")
        print("="*50)
        
        # 1. 统计平移 (Translation: X, Y, Z)
        trans_mean = all_trans.mean(dim=1)
        trans_std = all_trans.std(dim=1)
        print(f"[平移 Trans] (单位: 米)")
        print(f"   X轴 - 均值: {trans_mean[0]:.4f}, 标准差: {trans_std[0]:.4f}")
        print(f"   Y轴 - 均值: {trans_mean[1]:.4f}, 标准差: {trans_std[1]:.4f}")
        print(f"   Z轴 - 均值: {trans_mean[2]:.4f}, 标准差: {trans_std[2]:.4f}")
        print(f"   全局标准差均值: {trans_std.mean():.4f} (扩散模型最佳方差为 1.0, 建议缩放倍数: {1.0 / (trans_std.mean().item() + 1e-6):.1f}x)")
        
        # 2. 统计姿态 (6D Pose)
        pose_mean = all_pose.mean()
        pose_std = all_pose.std()
        print(f"\n[姿态 Pose 6D] (范围应在 [-1, 1] 之间)")
        print(f"   全局均值: {pose_mean:.4f}, 全局标准差: {pose_std:.4f}")
        
        # 3. 统计 2D 关键点 (Pixel Coordinates)
        j2d_mean = all_j2d.mean()
        j2d_std = all_j2d.std()
        print(f"\n[2D 投影 Pixel] (画面尺寸范围)")
        print(f"   全局均值: {j2d_mean:.1f}, 全局标准差: {j2d_std:.1f}")
        print("="*50 + "\n")