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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=1, sampling="conseq", sampling_step=1, split="train", pose_rep="rot6d", translation=False, glob=True, max_len=-1, min_len=-1, num_seq_max=-1, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.sampling = sampling
        self.sampling_step = sampling_step
        self.split = split
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.max_len = max_len
        self.min_len = min_len
        self.num_seq_max = num_seq_max

        self.align_pose_frontview = kwargs.get('align_pose_frontview', False)

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"{self.split} is not a valid split")
        
        # to remove shuffling
        self._original_train = None
        self._original_test = None

    def get_pose_data(self, data_index, frame_ix, is_right, y_data):
        pose, beta, ref_motion, inpain_mask = self._load(data_index, frame_ix, is_right, y_data)
        return pose, beta, ref_motion, inpain_mask
    
    def __getitem__(self, index):
        if self.split == 'train':
            data_index = self._train[index]
        else:
            data_index = self._test[index]
        return self._get_item_data_index(data_index)
    
    def _load(self, ind, frame_ix, is_right, y_data):
        pose_rep = self.pose_rep
        if pose_rep == "xyz" or self.translation:
            if getattr(self, "_load_joints3D", None) is not None:
                # Locate the root joint of initial pose at origin
                joints3D = self._load_joints3D(ind, frame_ix, is_right)
                joints3D = joints3D - joints3D[0, 0, :]
                ret = to_torch(joints3D)
                if self.translation:
                    ret_tr = ret[:, 0, :]
            else:
                if pose_rep == "xyz":
                    raise ValueError("This representation is not possible.")
                if getattr(self, "_load_translation") is None:
                    raise ValueError("Can't extract translations.")
                ret_tr = self._load_translation(ind, frame_ix, is_right)
                ret_tr = to_torch(ret_tr - ret_tr[0])
            # y
            if getattr(self, "_load_joints3D_y", None) is not None:
                joints3D_y, inpaint_mask = self._load_joints3D_y(ind, frame_ix, y_data)
                joints3D_y = joints3D_y - joints3D_y[0, 0, :]
                ret_y = to_torch(joints3D)
                if self.translation:
                    ret_tr_y = ret_y[:, 0, :]
            else:
                if pose_rep == "xyz":
                    raise ValueError("This representation is not possible.")
                if getattr(self, "_load_translation_y") is None:
                    raise ValueError("Can't extract translations.")
                ret_tr_y, inpaint_mask = self._load_translation_y(ind, frame_ix, y_data)
                ret_tr_y = to_torch(ret_tr_y - ret_tr_y[0])
                
        
        if pose_rep != "xyz":
            if getattr(self, "_load_rotvec", None) is None or getattr(self, "_load_rotvec_y", None) is None :
                raise ValueError("This representation is not possible.")
            else:
                pose, beta = self._load_rotvec(ind, frame_ix, is_right)
                if not self.glob:
                    pose = pose[:, 1:, :]
                pose = to_torch(pose)
                if self.align_pose_frontview:
                    first_frame_root_pose_matrix = geometry.axis_angle_to_matrix(pose[0][0])
                    all_root_poses_matrix = geometry.axis_angle_to_matrix(pose[:, 0, :])
                    aligned_root_poses_matrix = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1),
                                                            all_root_poses_matrix)
                    pose[:, 0, :] = geometry.matrix_to_axis_angle(aligned_root_poses_matrix)

                    if self.translation:
                        ret_tr = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1).float(),
                                            torch.transpose(ret_tr, 0, 1))
                        ret_tr = torch.transpose(ret_tr, 0, 1)
                # y
                pose_y, inpaint_mask = self._load_rotvec_y(ind, frame_ix, y_data)
                if not self.glob:
                    pose_y = pose_y[:, 1:, :]
                pose_y = to_torch(pose_y)
                if self.align_pose_frontview:
                    first_frame_root_pose_matrix = geometry.axis_angle_to_matrix(pose_y[0][0])
                    all_root_poses_matrix = geometry.axis_angle_to_matrix(pose_y[:, 0, :])
                    aligned_root_poses_matrix = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1),
                                                            all_root_poses_matrix)
                    pose_y[:, 0, :] = geometry.matrix_to_axis_angle(aligned_root_poses_matrix)

                    if self.translation:
                        ret_tr_y = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1).float(),
                                            torch.transpose(ret_tr_y, 0, 1))
                        ret_tr_y = torch.transpose(ret_tr_y, 0, 1)

                if pose_rep == "rotvec":
                    ret = pose
                    ret_y = pose_y
                elif pose_rep == "rotmat":
                    ret = geometry.axis_angle_to_matrix(pose).view(*pose.shape[:2], 9)
                    ret_y = geometry.axis_angle_to_matrix(pose_y).view(*pose_y.shape[:2], 9)
                elif pose_rep == "rotquat":
                    ret = geometry.axis_angle_to_quaternion(pose)
                    ret_y = geometry.axis_angle_to_quaternion(pose_y)
                elif pose_rep == "rot6d":
                    ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
                    ret_y = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose_y))

        if pose_rep != "xyz" and self.translation:
            padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
            padded_tr[:, :3] = ret_tr
            ret = torch.cat((ret, padded_tr[:, None]), 1)
            # y 
            padded_tr_y = torch.zeros((ret_y.shape[0], ret_y.shape[2]), dtype=ret_y.dtype)
            padded_tr_y[:, :3] = ret_tr_y
            ret_y = torch.cat((ret_y, padded_tr_y[:, None]), 1)
        
        ret = ret.permute(1, 2, 0).contiguous()  # J or J + 1, 6, T
        ret_y = ret_y.permute(1, 2, 0).contiguous() # J or J + 1, 6, T
        inpaint_mask = torch.from_numpy(inpaint_mask)

        return ret.float(), beta.float(), ret_y.float(), inpaint_mask.float()
    

    def _get_item_data_index(self, data_index):
        if getattr(self, "_load_cam", None) is None:
            raise ValueError("No cam params.")
        else:
            cam = self._load_cam(data_index)
        seq_y = self.seqs_y[data_index]
        with open(seq_y, 'rb') as f:
            data = pickle.load(f)
        nframes = data['frame_indices'][-1] - data['frame_indices'][0] + 1
        is_right = data["handedness"][0]

        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            # frame_ix = np.arange(nframes)
            frame_ix = np.arange(data['frame_indices'][0], data['frame_indices'][-1] + 1)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len

            mask = torch.zeros(num_frames, dtype=torch.bool)
            if num_frames > nframes:
                # adding the last frame until done
                ntoadd = max(0, num_frames - nframes)
                # lastframe = nframes - 1
                lastframe = data['frame_indices'][-1]
                padding = lastframe * np.ones(ntoadd, dtype=int)
                frame_ix = np.concatenate((np.arange(data['frame_indices'][0], data['frame_indices'][-1] + 1), padding))
                mask[:nframes] = True

            elif self.sampling in ["conseq", "random_conseq"]:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step) + data['frame_indices'][0]
                mask[:] = True

            # elif self.sampling == "random":
            #     choices = np.random.choice(range(nframes),
            #                                num_frames,
            #                                replace=False)
            #     frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")

        x0, x0_beta, y, inpaint_mask = self.get_pose_data(data_index, frame_ix, is_right, data)
        is_right = torch.from_numpy(np.asarray(is_right))
        

        output = {'inp': x0, 'beta': x0_beta, 'ref_motion': y, 'inpaint_mask': inpaint_mask, 'mask': mask, 'is_right': is_right, 'cam': cam}
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