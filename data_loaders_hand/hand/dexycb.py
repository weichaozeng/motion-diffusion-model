from .dataset import Dataset
from pathlib import Path
import torch
import os
import json
import random
import numpy as np
import pickle
from model_hand.rotation2xyz import Rotation2xyz
import yaml
from utils_hand.misc import to_torch 
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import utils_hand.rotation_conversions as geometry


beta_dir = {
    '01': '/home/zvc/Data/DexYCB/calibration/mano_20200709_140042_subject-01_right',
    '02': '/home/zvc/Data/DexYCB/calibration/mano_20200813_143449_subject-02_right',
    '03': '/home/zvc/Data/DexYCB/calibration/mano_20200820_133405_subject-03_right',
    '04': '/home/zvc/Data/DexYCB/calibration/mano_20200903_101911_subject-04_right',
    '05': '/home/zvc/Data/DexYCB/calibration/mano_20200908_140650_subject-05_right',
    '06': '/home/zvc/Data/DexYCB/calibration/mano_20200918_110920_subject-06_right',
    '07': '/home/zvc/Data/DexYCB/calibration/mano_20200807_132210_subject-07_right',
    '08': '/home/zvc/Data/DexYCB/calibration/mano_20201002_103251_subject-08_right',
    '09': '/home/zvc/Data/DexYCB/calibration/mano_20200514_142106_subject-09_right',
    '10': '/home/zvc/Data/DexYCB/calibration/mano_20201022_105224_subject-10_right',
}
MANO_HANDS_MEAN = [
    [ 0.1117, -0.0429,  0.4164],
    [ 0.1088,  0.0660,  0.7562],
    [-0.0964,  0.0909,  0.1885],
    [-0.1181, -0.0509,  0.5296],
    [-0.1437, -0.0552,  0.7049],
    [-0.0192,  0.0923,  0.3379],
    [-0.4570,  0.1963,  0.6255],
    [-0.2147,  0.0660,  0.5069],
    [-0.3697,  0.0603,  0.0795],
    [-0.1419,  0.0859,  0.6355],
    [-0.3033,  0.0579,  0.6314],
    [-0.1761,  0.1321,  0.3734],
    [ 0.8510, -0.2769,  0.0915],
    [-0.4998, -0.0266, -0.0529],
    [ 0.5356, -0.0460,  0.2774]
    ]


def read_anno_from_dir(anno_dir):
    dir_path = Path(anno_dir)
    npz_files = sorted(dir_path.glob("*.npz"))
    all_pose_m = []
    all_kp_2d = []
    all_kp_3d = []
    for file_path in npz_files:
        with np.load(file_path) as data:
            pose_m = data['pose_m']  
            kp_2d = data['joint_2d']   
            kp_3d = data['joint_3d']   
            all_pose_m.append(pose_m)
            all_kp_2d.append(kp_2d)
            all_kp_3d.append(kp_3d)
    return np.array(all_pose_m), np.array(all_kp_2d), np.array(all_kp_3d)

def read_beta(beta_path):
    with open(beta_path, 'r') as f:
        beta_data = yaml.load(f, Loader=yaml.FullLoader)
    beta = np.array(beta_data['betas'])
    return beta


def read_cam(cam_path, n_Frames=1):
    with open(cam_path, 'r') as f:
        cam_data = yaml.load(f, Loader=yaml.FullLoader)
    # intrinsics
    intrs = np.zeros((3, 3))
    intrs[0, 0] = cam_data['color']['fx']
    intrs[1, 1] = cam_data['color']['fy']
    intrs[0, 2] = cam_data['color']['ppx']
    intrs[1, 2] = cam_data['color']['ppy']
    intrs[2, 2] = 1.0
    # rot & trans
    extrinsics_matrix = np.array(cam_data['extrinsics']).reshape(3, 4)
    rot = extrinsics_matrix[:, :3]
    trans = extrinsics_matrix[:, 3]
    # 
    cameras = {
        'K': np.repeat(intrs[None, ...], n_Frames, axis=0),
        'R': np.repeat(rot[None, ...], n_Frames, axis=0),
        'T': np.repeat(trans[None, ...], n_Frames, axis=0),
    }
    return cameras

class DexYCB(Dataset):
    dataname = "dexycb"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seqs_y = []
        self.seqs_kp3d = []
        self.seqs_mano = []
        self.seqs_cam = []
        self.seqs_video = []
        
        data_path = Path("/home/zvc/Project/VHand/test_dataset/DexYCB/vhand/hamer_out")
        anno_root = Path("/home/zvc/Data/DexYCB/s0_train") 
        rgb_root = Path("/home/zvc/Data/DexYCB/s0_train_rgb")   
        cam_root = Path("/home/zvc/Data/DexYCB/calibration") 

        outs = sorted(os.listdir(anno_root))[:300]

        for out in outs:
            beta_name = out.split('_')[0].split('-')[-1]
            cam = out.split('_')[-1]
            # video
            video_path = rgb_root / out
            # anno
            all_pose_m, all_kp_2d, all_kp_3d = read_anno_from_dir(anno_root / out)
            # beta
            beta_path = Path(beta_dir[beta_name]) / 'mano.yml'
            beta = read_beta(beta_path)
            # cam
            cam_path = cam_root / 'intrinsics' / f'{cam}_640x480.yml'
            cam = read_cam(cam_path)
            
            track_dir = data_path / out / 'results' / 'track'
            track_files = list(track_dir.glob('*.pkl'))
            num_tracks = len(track_files)

            if num_tracks > 2:
                print(f"Warning: {num_tracks} tracks found in {track_dir}, skip.")
                continue
            elif num_tracks == 0:
                print(f"Warning: no tracks found in {track_dir}, skip.")
                continue
            elif num_tracks == 2:
                target_track = None
                for track_file in track_files:
                    with open(track_file, 'rb') as f:
                        temp_data = pickle.load(f)
                    if "handedness" in temp_data and temp_data["handedness"][0] != 0:
                        target_track = track_file
                        break
                if target_track is None:
                    print(f"Warning: 2 tracks found but no right hand in {track_dir}, skip.")
                    continue
            else:
                target_track = track_files[0]
            self.seqs_y.append(target_track)
            self.seqs_kp3d.append(all_kp_3d)
            self.seqs_mano.append({
                'beta': beta,
                'pose_m': all_pose_m,
            })
            self.seqs_cam.append(cam)
            self.seqs_video.append(video_path)
        
        _all_indices = list(range(len(self.seqs_y)))
        random.seed(42)
        random.shuffle(_all_indices)

        val_size = 128
        self._val = _all_indices[:val_size]
        self._train = _all_indices[val_size:]

        self.rot2xyz = Rotation2xyz(device='cpu')

        prior_path = '/home/zvc/Project/motion-diffusion-model/body_models/mano_right_pca_prior.npz'
        prior_data = np.load(prior_path)
        self.pca_basis = torch.from_numpy(prior_data['pca_basis']).float().detach()
        self.mean_pose = torch.from_numpy(prior_data['mean_pose']).float().detach()


    def _load_cam(self, ind):
        return self.seqs_cam[ind]
    

    def _load_rotvec_x(self, ind, frame_ix, x_data, cam, is_right, flip_left=True):
        if is_right:
            full_poses = torch.tensor(x_data['pose_m'][:, 0, :48])
            beta = torch.tensor(x_data['beta'])
        else:
            raise NotImplementedError("Left hand PCA inverse not implemented yet.")

        poses_subset = full_poses[frame_ix] 
        poses_subset = to_torch(poses_subset).float()

        root_pose_cam = poses_subset[:, :3]      # shape: (num_frames, 3)
        pca_coeffs = poses_subset[:, 3:48]   # shape: (num_frames, 45)
        device = pca_coeffs.device

        cam_R = to_torch(cam['R'][0]).float().to(device)
        R_root_cam = geometry.axis_angle_to_matrix(root_pose_cam)
        R_root_world = torch.matmul(cam_R.t(), R_root_cam) 
        root_pose_world = geometry.matrix_to_axis_angle(R_root_world)

        
        pca_basis = self.pca_basis.to(device)
        mean_pose = self.mean_pose.to(device)

        # joint_angles = PCA_coeffs @ PCA_basis + Mean_pose
        joint_pose = torch.matmul(pca_coeffs, pca_basis) + mean_pose # shape: (num_frames, 45)

        full_axis_angle = torch.cat([root_pose_world, joint_pose], dim=1)  # shape: (num_frames, 48)
        poses = full_axis_angle.reshape(-1, 16, 3)  # shape: (num_frames, 16, 3)
        if not is_right and flip_left:
            raise NotImplementedError()
        return poses, to_torch(beta).squeeze(0).float()


    def _load_translation_x(self, ind, frame_ix, x_data, cam, is_right, flip_left=True):
        if is_right:
            Th_cam = torch.tensor(x_data['pose_m'][:, 0, 48:51])[frame_ix].float()
        else:
            raise NotImplementedError
        if not is_right and flip_left:
            raise NotImplementedError
        # cam_R = to_torch(cam['R'][0]).float().to(Th_cam.device)
        # cam_T = to_torch(cam['T'][0]).float().to(Th_cam.device)
        
        # Th_world = torch.matmul(Th_cam - cam_T, cam_R)
        return Th_cam


    def _load_rotvec_y(self, ind, frame_ix, y_data, cam):
        frame_indices = y_data['frame_indices']
        hand_pose = torch.from_numpy(np.asarray([mano['hand_pose'] for mano in y_data['mano']]))
        global_orient = torch.from_numpy(np.asarray([mano['global_orient'] for mano in y_data['mano']]))
        # corret
        boxes = torch.from_numpy(np.asarray(y_data['boxes']))
        crop_centers = boxes[:, 0:2] # [cx, cy, bbox_size, bbox_size]
        
        global_orient_corrected = self.get_hamer_to_world_orient(
            global_orient, 
            cam['R'][0],
            # virtual_R, 
            crop_centers,  
            cam['K'][0],
            # virtual_K,      
        )
        pose_mean = torch.tensor(np.asarray(MANO_HANDS_MEAN), dtype=torch.float32)
        hand_pose_rotvec = geometry.matrix_to_axis_angle(hand_pose)
        hand_pose_corrected = hand_pose_rotvec - pose_mean
        hand_pose_corrected = geometry.axis_angle_to_matrix(hand_pose_corrected)

        global_orient_corrected = global_orient_corrected.numpy()
        hand_pose_corrected = hand_pose_corrected.numpy()

        # slerp
        indices = np.searchsorted(frame_indices, [frame_ix[0], frame_ix[-1]])
        start_idx = max(0, indices[0] - 1)
        end_idx = indices[1]

        chunk_indices = frame_indices[start_idx:end_idx+1]
        target_times = np.arange(chunk_indices[0], chunk_indices[-1] + 1)

        full_pose, inpaint_mask = self._slerp_y(
            chunk_indices, 
            global_orient_corrected[start_idx:end_idx+1],
            hand_pose_corrected[start_idx:end_idx+1]
        )

        relative_indices = frame_ix - chunk_indices[0]
        target_idx = relative_indices


        return full_pose[target_idx], inpaint_mask[target_idx]


    def _load_translation_y(self, ind, frame_ix, y_data, cam):
        frame_indices = y_data['frame_indices']
        # pred_cam: [s, tx_local, ty_local]
        # boxes: [cx, cy, box_size]
        pred_cam = np.asarray(y_data['pred_cam'])
        boxes = np.asarray(y_data['boxes'])

        indices = np.searchsorted(frame_indices, [frame_ix[0], frame_ix[-1]])
        start_idx = max(0, indices[0] - 1)
        end_idx = indices[1]

        full_pred_cam, inpaint_mask = self._interpolate_y(frame_indices[start_idx:end_idx+1], pred_cam[start_idx:end_idx+1])
        full_boxes, _ = self._interpolate_y(frame_indices[start_idx:end_idx+1], boxes[start_idx:end_idx+1])

        relative_indices = frame_ix - frame_indices[start_idx]
        target_idx = relative_indices

        target_pred_cam = to_torch(full_pred_cam[target_idx])
        target_boxes = to_torch(full_boxes[target_idx])
        mask = inpaint_mask[target_idx]

        fx_real = cam['K'][0][0, 0]
        fy_real = cam['K'][0][1, 1]
        f_real = (fx_real + fy_real) / 2.0
        cx_real = cam['K'][0][0, 2]
        cy_real = cam['K'][0][1, 2]

        s = target_pred_cam[:, 0]
        tx_local = target_pred_cam[:, 1]
        ty_local = target_pred_cam[:, 2]

        bx = target_boxes[:, 0]
        by = target_boxes[:, 1]
        w = target_boxes[:, 2]
        h = target_boxes[:, 3]
        b = torch.max(w, h)

        # Z_real = 2 * f_real / (box_size * scale)
        z_real = (2.0 * f_real) / (b * s + 1e-6)

        u = bx + tx_local * (b * s / 2.0) 
        v = by + ty_local * (b * s / 2.0)

        x_real = (u - cx_real) * z_real / fx_real
        y_real = (v - cy_real) * z_real / fy_real

        y_trans_cam_real = torch.stack([x_real, y_real, z_real], dim=-1)

        return y_trans_cam_real, mask



    def _interpolate_y(self, frame_indices, cam_trans):
        frame_indices = np.asarray(frame_indices)
        cam_trans = np.asarray(cam_trans)

        min_t, max_t = frame_indices[0], frame_indices[-1]
        total_N = max_t - min_t + 1
        target_times = np.arange(min_t, max_t + 1)
        target_times_clipped = np.clip(target_times, min_t, max_t)

        mask = np.zeros(total_N, dtype=np.bool_)
        mask[frame_indices - min_t] = True

        # --- [OLD] Linear Interplate ---
        full_trans = np.stack([
            np.interp(target_times_clipped, frame_indices, cam_trans[:, i]) 
            for i in range(cam_trans.shape[-1])
        ], axis=-1)

        # --- [NEW] Padding with Zero ---
        # full_trans = np.zeros((total_N, cam_trans.shape[-1]), dtype=np.float32)
        # full_trans[frame_indices - min_t] = cam_trans

        return full_trans.astype(np.float32), mask


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
        # global_orient: (N, 1, 3, 3), hand_pose: (N, 15, 3, 3)
        full_pose_mat = np.concatenate([global_orient, hand_pose], axis=1) # (N, 16, 3, 3)
        N, J = full_pose_mat.shape[:2]

        min_t, max_t = frame_indices[0], frame_indices[-1]
        total_N = max_t - min_t + 1
        target_times = np.arange(min_t, max_t + 1)

        mask = np.zeros(total_N, dtype=np.bool_)
        mask[frame_indices - min_t] = True

        # --- [OLD] Slerp ---
        full_pose_rotvecs = []
        for j in range(J):
            rots = R.from_matrix(full_pose_mat[:, j])
            slerp = Slerp(frame_indices, rots)
            full_pose_rotvecs.append(slerp(target_times).as_rotvec())
        full_pose = np.stack(full_pose_rotvecs, axis=1)

        # --- [NEW] Padding with Zero ---
        # valid_pose_rotvecs = []
        # for j in range(J):
        #     rots = R.from_matrix(full_pose_mat[:, j]).as_rotvec()
        #     valid_pose_rotvecs.append(rots)
        # valid_pose_rotvecs = np.stack(valid_pose_rotvecs, axis=1) # (N, 16, 3)
        
        # full_pose = np.zeros((total_N, J, 3), dtype=np.float32)
        # full_pose[frame_indices - min_t] = valid_pose_rotvecs

        return full_pose.astype(np.float32), mask


    def _load_2d_joint(self, ind, frame_ix, x_data, cam, is_right):
        from utils_hand.misc import to_torch
        if not is_right:
            raise NotImplementedError("Left hand 2D joint projection not implemented.")

        poses, beta = self._load_rotvec_x(ind, frame_ix, x_data, cam, is_right)
        
        # Th: (T, 3)
        Th = self._load_translation_x(ind, frame_ix, x_data, cam, is_right)

        pose_input = poses.permute(1, 2, 0).unsqueeze(0).contiguous()
        trans_input = Th.unsqueeze(0)
        beta_input = beta.unsqueeze(0) if beta.dim() == 1 else beta

        j_3d = self.rot2xyz(
            pose=pose_input,
            pose_rep='rotvec',
            beta=beta_input,
            translation=trans_input,
            root_revise=False,
        ).squeeze(0)     

        cam_K = to_torch(cam['K'][0]).float()
        cam_R = to_torch(cam['R'][0]).float()
        cam_T = to_torch(cam['T'][0]).float()
        
        # (21, 3, T) -> (T, 21, 3) 
        j_3d = j_3d.permute(2, 0, 1).contiguous()
        j_cam = torch.matmul(j_3d, cam_R.t()) + cam_T
        j_img = torch.matmul(j_cam, cam_K.t())
        
        z = j_img[..., 2:3] + 1e-6 
        j_2d = j_img[..., :2] / z

        return j_2d

    
    def get_hamer_to_world_orient(self, y_global_orient, cam_extrinsic, crop_center, cam_intrinsics):
        N = y_global_orient.shape[0]
        device = y_global_orient.device

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device).float()
            return torch.from_numpy(x).to(device).float()
        
        R_w2c = to_tensor(cam_extrinsic)
        R_w2c = R_w2c[:3, :3]
        R_c2w = R_w2c.t() 

        y_global_orient_world = R_c2w.unsqueeze(0) @ y_global_orient.squeeze(1)
        
        return y_global_orient_world.unsqueeze(1)
    
    # def _interpolate_linear(self, source_indices, values, target_indices):
    #     v_np = values.numpy() if torch.is_tensor(values) else values
    #     t_np = source_indices.numpy() if torch.is_tensor(source_indices) else source_indices
    #     target_np = target_indices.numpy() if torch.is_tensor(target_indices) else target_indices

    #     f = interp1d(t_np, v_np, axis=0, kind='linear', fill_value="extrapolate")
    #     interpolated_values = f(target_np)
        
    #     return torch.from_numpy(interpolated_values).float()


if __name__ == "__main__":
    dataset = DexYCB()



