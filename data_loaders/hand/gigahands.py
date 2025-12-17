from dataset import Dataset
import torch
import os
import json
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
# class GigaHands(Dataset):
#     dataname = "gigahands"

#     def __init__(self, datapath="dataset/gigahands", **kwargs):
#         super().__init__(**kwargs)
#         self.datapath = datapath

#         self.scenes = sorted(os.listdir(self.datapath))[:2]
#         self.seqs_kp3d = []
#         self.seqs_mano = []
#         self._num_frames_in_video = []
#         for scene in self.scenes:
#             dir_kp3d = os.path.join(self.datapath, scene, "keypoints_3d_mano_align")
#             dir_mano = os.path.join(self.datapath, scene, "params")
#             for seq in sorted(os.listdir(dir_kp3d)):
#                 with open(os.path.join(dir_mano, seq), 'r') as f:
#                     _data = json.load(f)
#                 # left hand
#                 self.seqs_kp3d.append(os.path.join(dir_kp3d, seq))
#                 self.seqs_mano.append(os.path.join(dir_mano, seq))
#                 self._num_frames_in_video.append(len(_data["left"]["poses"]))
#                 # right hand
#                 self.seqs_kp3d.append(os.path.join(dir_kp3d, seq))
#                 self.seqs_mano.append(os.path.join(dir_mano, seq))
#                 self._num_frames_in_video.append(len(_data["right"]["poses"]))
        
#         self._train = list(range(100, len(self.seqs_kp3d)))
#         self._test = list(range(0, 100))


#     def _load_rotvec(self, ind, frame_ix, flip_left=True):
#         mano_params_path = self.seqs_mano[ind]
#         with open(mano_params_path, 'r') as f:
#             mano_data = json.load(f)
#         is_left_hand = (ind % 2 == 0)
#         if is_left_hand:
#             full_poses = torch.tensor(mano_data["left"]["poses"], dtype=torch.float32) 
#             Rh = torch.tensor(mano_data["left"]["Rh"], dtype=torch.float32)
#         else:
#             full_poses = torch.tensor(mano_data["right"]["poses"], dtype=torch.float32)
#             Rh = torch.tensor(mano_data["right"]["Rh"], dtype=torch.float32)
#         full_poses = torch.cat([Rh, full_poses[:, 3:]], dim=1) # (num_frames, 16*3)
#         poses = full_poses[frame_ix].reshape(-1, 16, 3)  # (num_frames, 16, 3)
#         if is_left_hand and flip_left:
#             poses[:, :, 1] *= -1
#             poses[:, :, 2] *= -1
#         return poses        
    
#     def _load_joints3D(self, ind, frame_ix, flip_left=True):
#         joints_path = self.seqs_kp3d[ind]
#         with open(joints_path, 'r') as f:
#             joints_data = json.load(f)
#         full_joints = torch.tensor(joints_data, dtype=torch.float32)
#         batch_joints = full_joints[frame_ix]  # (num_frames, 42, 3)
#         is_left_hand = (ind % 2 == 0) 
#         if is_left_hand:
#             joints3D = batch_joints[:, :21]  # (num_frames, 21, 3)
#         else:
#             joints3D = batch_joints[:, 21:]  # (num_frames, 21, 3)
#         if is_left_hand and flip_left:
#             joints3D[:, 0] *= -1
#         return joints3D

class GigaHands(Dataset):
    dataname = "gigahands"

    def __init__(self, datapath="/home/zvc/Project/VHand/test_dataset/GigaHands/vhand/hamer_out", **kwargs):
        super().__init__(**kwargs)
        self.datapath = datapath
        self.seqs_y = []
        self.seqs_kp3d = []
        self.seqs_mano = []
        
        rgb_seq_json = "/home/zvc/Data/GigaHands/multiview_rgb_seq_info.json"
        rgb_root = "/home/zvc/Data/GigaHands/multiview_rgb_vids"
        with open(rgb_seq_json, 'r') as f:
            rgb_seq_data = json.load(f)        
        outs = sorted(os.listdir(self.datapath))[-200:]

        for out in outs:
            scene = out.split('_')[0]
            cam_id = out.split('_')[1] + '_' + out.split('_')[2]
            vid_id = out.split('_')[3] + '_' + out.split('_')[4] + '_' + out.split('_')[5] + '.mp4'
            video_path = os.path.join(rgb_root, scene, cam_id, vid_id)
            mano_path = rgb_seq_data[video_path]['mano_param_path']
            kp3d_path = rgb_seq_data[video_path]['kp_3d_path']
            for track in os.listdir(os.path.join(self.datapath, out, 'results', 'track_500.0')):
                self.seqs_y.append(os.path.join(self.datapath, out, 'results', 'track_500.0', track))
                self.seqs_kp3d.append(kp3d_path)
                self.seqs_mano.append(mano_path)
    
        
        self._train = list(range(100, len(self.seqs_y)))
        self._test = list(range(0, 100))


    def _load_rotvec(self, ind, frame_ix, is_right, flip_left=True):
        mano_params_path = self.seqs_mano[ind]
        with open(mano_params_path, 'r') as f:
            mano_data = json.load(f)

        if is_right:
            full_poses = torch.tensor(mano_data["right"]["poses"], dtype=torch.float32)
            Rh = torch.tensor(mano_data["right"]["Rh"], dtype=torch.float32)
            beta = torch.tensor(mano_data["right"]["shapes"], dtype=torch.float32)
        else:
            full_poses = torch.tensor(mano_data["left"]["poses"], dtype=torch.float32) 
            Rh = torch.tensor(mano_data["left"]["Rh"], dtype=torch.float32)
            beta = torch.tensor(mano_data["left"]["shapes"], dtype=torch.float32)
        full_poses = torch.cat([Rh, full_poses[:, 3:]], dim=1) # (num_frames, 16*3)
        poses = full_poses[frame_ix].reshape(-1, 16, 3)  # (num_frames, 16, 3)
        if not is_right and flip_left:
            poses[:, :, 1] *= -1
            poses[:, :, 2] *= -1
    
        return poses, beta        
    
    def _load_joints3D(self, ind, frame_ix, is_right, flip_left=True):
        joints_path = self.seqs_kp3d[ind]
        with open(joints_path, 'r') as f:
            joints_data = json.load(f)
        full_joints = torch.tensor(joints_data, dtype=torch.float32)
        batch_joints = full_joints[frame_ix]  # (num_frames, 42, 3)

        if is_right:
            joints3D = batch_joints[:, 21:]  # (num_frames, 21, 3)
        else:
            joints3D = batch_joints[:, :21]  # (num_frames, 21, 3)
           
        if not is_right and flip_left:
            joints3D[:, 0] *= -1
        return joints3D
    
    def _load_translation(self, ind, frame_ix, is_right, flip_left=True):
        mano_params_path = self.seqs_mano[ind]
        with open(mano_params_path, 'r') as f:
            mano_data = json.load(f)

        if is_right:
            Th = torch.tensor(mano_data["right"]["Th"], dtype=torch.float32)
        else:
            Th = torch.tensor(mano_data["left"]["Th"], dtype=torch.float32)
        if not is_right and flip_left:
            Th[:, 0] *= -1
        return Th[frame_ix]  
    
    def _load_rotvec_y(self, ind, frame_ix, y_data):
        # seq_y_path = self.seqs_y[ind]
        frame_indices = y_data['frame_indices']
        hand_pose = [mano['hand_pose'] for mano in y_data['mano']]
        global_orient = [mano['global_orient'] for mano in y_data['mano']]

        indices = np.searchsorted(frame_indices, [frame_ix[0], frame_ix[-1]])
        start_idx = indices[0]
        end_idx = indices[1]

        full_pose_rotvec, inpaint_mask = self._slerp_y(frame_indices[start_idx:end_idx+1], global_orient[start_idx:end_idx+1], hand_pose[start_idx:end_idx+1])

        relative_indices = frame_ix - frame_indices[start_idx]
        max_relative_idx = full_pose_rotvec.shape[0] - 1
        relative_indices = np.clip(relative_indices, 0, max_relative_idx)

        return full_pose_rotvec[relative_indices], inpaint_mask[relative_indices]

    def _load_joints3D_y(self, ind, frame_ix, y_data):
        raise NotImplementedError


    def _load_translation_y(self, ind, frame_ix, y_data):
        frame_indices = y_data['frame_indices']
        cam_trans = y_data['cam_trans']

        indices = np.searchsorted(frame_indices, [frame_ix[0], frame_ix[-1]])
        start_idx = indices[0]
        end_idx = indices[1]

        full_trans, inpaint_mask = self._interpolate_y(frame_indices[start_idx:end_idx+1], cam_trans[start_idx:end_idx+1])

        relative_indices = frame_ix - frame_indices[start_idx]
        max_relative_idx = full_trans.shape[0] - 1
        relative_indices = np.clip(relative_indices, 0, max_relative_idx)

        return full_trans[relative_indices], inpaint_mask[relative_indices]

    
    def _interpolate_y(self, frame_indices, cam_trans):
        """
        Args:
            frame_indices: (N,)
            cam_trans: (N, 3)
        
        Returns:
            full_trans: (Total_N, 3)
            mask: (Total_N, )
        """
        frame_indices = np.asarray(frame_indices)
        cam_trans = np.asarray(cam_trans)

        min_t, max_t = frame_indices[0], frame_indices[-1]
        total_N = max_t - min_t + 1
        target_times = np.arange(min_t, max_t + 1)

        mask = np.zeros(total_N, dtype=np.float32)
        relative_indices = frame_indices - min_t
        mask[relative_indices] = 1.0

        idx_right = np.searchsorted(frame_indices, target_times, side='left')
        idx_right = np.clip(idx_right, 1, len(frame_indices) - 1)
        idx_left = idx_right - 1
        t_left = frame_indices[idx_left]
        t_right = frame_indices[idx_right]
        val_left = cam_trans[idx_left]
        val_right = cam_trans[idx_right]
        dt = t_right - t_left
        alpha = (target_times - t_left) / dt
        alpha = alpha[:, np.newaxis]

        full_trans = val_left + alpha * (val_right - val_left)    
        return full_trans.astype(np.float32), mask.astype(np.bool_)


    def _slerp_y(self, frame_indices, global_orient, hand_pose):
        """
        Args:
            frame_indices: (N,)
            global_orient: (N, 1, 3, 3)
            hand_pose: (N, 15, 3, 3)
        
        Returns:
            full_pose : (Total_N, 16, 3)
            mask: (Total_N,)
        """
        frame_indices = np.asarray(frame_indices)
        global_orient = np.asarray(global_orient)
        hand_pose = np.asarray(hand_pose)

        N = len(frame_indices)
        full_pose_mat = np.concatenate([global_orient, hand_pose], axis=1) # (N, 16, 3, 3)
        flat_mat = full_pose_mat.reshape(-1, 3, 3)
        quats = R.from_matrix(flat_mat).as_quat()
        quats = quats.reshape(N, 16, 4)

        min_t, max_t = frame_indices[0], frame_indices[-1]
        total_N = max_t - min_t + 1
        target_times = np.arange(min_t, max_t + 1)
        mask = np.zeros(total_N, dtype=np.float32)
        relative_indices = frame_indices - min_t
        mask[relative_indices] = 1.0

        idx_right = np.searchsorted(frame_indices, target_times, side='left')
        idx_right = np.clip(idx_right, 1, N-1)
        idx_left = idx_right - 1
        t_left = frame_indices[idx_left]
        t_right = frame_indices[idx_right]
        dt = t_right - t_left
        alpha = (target_times - t_left) / dt
        alpha = alpha[:, np.newaxis, np.newaxis]
        
        # q(t) = (sin((1-a)θ)/sinθ) * q0 + (sin(aθ)/sinθ) * q1
        q0 = quats[idx_left]
        q1 = quats[idx_right]
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        dot = np.abs(dot)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        epsilon = 1e-6
        use_slerp = sin_theta > epsilon
        scale0 = 1.0 - alpha
        scale1 = alpha
        sin_theta_safe = np.where(use_slerp, sin_theta, 1.0)
        k0 = np.sin((1 - alpha) * theta) / sin_theta_safe
        k1 = np.sin(alpha * theta) / sin_theta_safe
        scale0 = np.where(use_slerp, k0, scale0)
        scale1 = np.where(use_slerp, k1, scale1)

        q_interp = scale0 * q0 + scale1 * q1
        q_interp = q_interp / np.linalg.norm(q_interp, axis=-1, keepdims=True)
        q_final_flat = q_interp.reshape(-1, 4)
        r_final = R.from_quat(q_final_flat)
        full_pose = r_final.as_rotvec().reshape(total_N, 16, 3)

        return full_pose.astype(np.float32), mask.astype(np.bool_)


# vis
def visualize_batch(dataset_item):
    inp = dataset_item['inp']
    ref_motion = dataset_item['ref_motion']
    inpaint_mask = dataset_item['inpaint_mask'] # 1=Real, 0=Interp
    padding_mask = dataset_item['mask']

    T = ref_motion.shape[-1]
    frames = np.arange(T)


if __name__ == "__main__":
    dataset = GigaHands(split="train", num_frames=128, sampling="conseq", pose_rep="rot6d")
    print(len(dataset))
    sample = dataset[0]
    

    # import utils.rotation_conversions as geometry
    # pose = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(data.permute(2, 0, 1)))
    x0 = sample['inp']
    y = sample['ref_motion']
    inpaint_mask = sample['inpaint_mask']
    mask = sample['mask']
    beta = sample['beta']

    print("debug")