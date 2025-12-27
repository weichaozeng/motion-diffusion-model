from dataset import Dataset
import torch
import os
import json
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import vis.camparam_utils as param_utils
from pathlib import Path
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


def read_params(params_path):
    """Reads camera intrinsics and extrinsics from a formatted txt file."""
    params = np.loadtxt(
        params_path,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ]
    )
    params = np.sort(params, order="cam_name")
    return params

def get_projections(params, cam_names, n_Frames=1):
    """Returns camera intrinsics, extrinsics, projections, and distortion parameters for the named camera."""
    projs, intrs, dists, rot, trans = [], [], [], [], []
    for param in params:
        if param["cam_name"] == cam_names:
            extr = param_utils.get_extr(param)
            intr, dist = param_utils.get_intr(param)
            r, t = param_utils.get_rot_trans(param)
            rot.append(r)
            trans.append(t)
            intrs.append(intr.copy())
            projs.append(intr @ extr)
            dists.append(dist)
    cameras = {
        'K': np.repeat(np.asarray(intrs), n_Frames, axis=0),
        'R': np.repeat(np.asarray(rot), n_Frames, axis=0),
        'T': np.repeat(np.asarray(trans), n_Frames, axis=0),
        'dist': np.repeat(np.asarray(dists), n_Frames, axis=0),
        'P': np.repeat(np.asarray(projs), n_Frames, axis=0)
    }
    return cameras


class GigaHands(Dataset):
    dataname = "gigahands"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seqs_y = []
        self.seqs_kp3d = []
        self.seqs_mano = []
        self.seqs_cam = []
        self.seqs_video = []
        
        data_path = Path("/home/zvc/Project/VHand/test_dataset/GigaHands/vhand/hamer_out")
        anno_root = Path("/home/zvc/Data/GigaHands/hand_poses") 
        rgb_root = Path("/home/zvc/Data/GigaHands/video_aligned")    
        # outs = sorted(os.listdir(data_path))[:200]
        outs = [
            "p001-folder_001_brics-odroid-002_cam0", "p001-folder_001_brics-odroid-006_cam0",
            "p001-packing_008_brics-odroid-008_cam0",
            "p001-packing_008_brics-odroid-022_cam0",
        ]

        for out in outs:
            session = out.split('_')[0]
            seq = out.split('_')[1]
            cam = out.split('_')[2] + '_' + out.split('_')[3]
            # video
            video_path = rgb_root / session / 'aligned_video' / seq /cam / f'{cam}.mp4'
            # anno
            mano_path = anno_root / session / 'params' / f'{seq}.json'
            kp3d_path = anno_root / session / 'keypoints_3d_mano_align' / f'{seq}.json'
            # cam
            camera_path = os.path.join(os.path.dirname(os.path.dirname(mano_path)), 'optim_params.txt')
            cam_params = read_params(camera_path)
            cam = get_projections(cam_params, cam, n_Frames=1)

            for track in os.listdir(data_path / out / 'results' / 'track_500.0'):
                self.seqs_y.append(data_path / out / 'results' / 'track_500.0' / track)
                self.seqs_kp3d.append(kp3d_path)
                self.seqs_mano.append(mano_path)
                self.seqs_cam.append(cam)
                self.seqs_video.append(video_path)
                
    
        
        self._train = list(range(0, len(self.seqs_y)))
        # self._test = list(range(0, 100))

    def _load_cam(self, ind):
        return self.seqs_cam[ind]

    def _load_rotvec(self, ind, frame_ix, mano_data, is_right, flip_left=True):
        # mano_params_path = self.seqs_mano[ind]
        # with open(mano_params_path, 'r') as f:
        #     mano_data = json.load(f)

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
    
    # def _load_joints3D(self, ind, frame_ix, is_right, flip_left=True):
    #     joints_path = self.seqs_kp3d[ind]
    #     with open(joints_path, 'r') as f:
    #         joints_data = json.load(f)
    #     full_joints = torch.tensor(joints_data, dtype=torch.float32)
    #     batch_joints = full_joints[frame_ix]  # (num_frames, 42, 3)

    #     if is_right:
    #         joints3D = batch_joints[:, 21:]  # (num_frames, 21, 3)
    #     else:
    #         joints3D = batch_joints[:, :21]  # (num_frames, 21, 3)
           
    #     if not is_right and flip_left:
    #         joints3D[:, 0] *= -1
    #     return joints3D
    
    def _load_translation(self, ind, frame_ix, mano_data, is_right, flip_left=True):

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

    # def _load_joints3D_y(self, ind, frame_ix, y_data):
    #     raise NotImplementedError


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


# # vis
# def visualize_batch(dataset_item):
#     inp = dataset_item['inp']
#     ref_motion = dataset_item['ref_motion']
#     inpaint_mask = dataset_item['inpaint_mask'] # 1=Real, 0=Interp
#     padding_mask = dataset_item['mask']

#     T = ref_motion.shape[-1]
#     frames = np.arange(T)


if __name__ == "__main__":
    from tqdm import tqdm
    import decord
    from decord import VideoReader, cpu

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dataset = GigaHands(split="train", num_frames=128, sampling="conseq", sampling_step=1, pose_rep="rot6d", translation=False, align_pose_frontview=True)
    print(len(dataset))
    for sample_idx in range(10):
        sample = dataset[sample_idx]
        import utils.rotation_conversions as geometry
        # translation
        if dataset.translation:
            x0_trans = sample['inp'].permute(2, 0, 1)[:, -1, :3]
            x0_trans = torch.matmul(sample['inp_ff_root_pose_mat'], torch.transpose(x0_trans, 0, 1))
            x0_trans = torch.transpose(x0_trans, 0, 1)
            x0 = sample['inp'].permute(2, 0, 1)[:, :-1, :]
            x0 = geometry.rotation_6d_to_matrix(x0)
            #x0 = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(x0))
            x0_root = sample['inp_root']
            x0_trans += x0_root

            # y_trans = sample['ref_motion'].permute(2, 0, 1)[:, -1, :3]
            # y_trans = torch.matmul(sample['inp_ff_root_pose_mat'], torch.transpose(y_trans, 0, 1))     
            # y_trans = torch.transpose(y_trans, 0, 1)     
            y = sample['ref_motion'].permute(2, 0, 1)[:, :-1, :]
            y = geometry.rotation_6d_to_matrix(y)
            # y = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(y))
            y_root = sample['ref_motion_root']
            y_trans += y_root
            y_trans = x0_trans 
        else:
            x0_trans = torch.zeros_like(sample['inp'].permute(2, 0, 1)[:, -1, :3])
            x0 = sample['inp'].permute(2, 0, 1)
            x0 = geometry.rotation_6d_to_matrix(x0)
            #x0 = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(x0))
            
            y_trans = torch.zeros_like(sample['ref_motion'].permute(2, 0, 1)[:, -1, :3])
            y = sample['ref_motion'].permute(2, 0, 1)
            y = geometry.rotation_6d_to_matrix(y)
            # y = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(y))
        
        #align_pose_frontview[
        x0_all_root_pose_mat = x0[:, 0, :]
        x0_all_root_pose_mat = torch.matmul(sample['inp_ff_root_pose_mat'], x0_all_root_pose_mat)
        x0[:, 0, :] = x0_all_root_pose_mat
        x0 = geometry.matrix_to_axis_angle(x0)

        y_all_root_pose_mat = y[:, 0, :]
        # y_all_root_pose_mat = torch.matmul(sample['ref_motion_ff_root_pose_mat'], y_all_root_pose_mat)
        y_all_root_pose_mat = torch.matmul(sample['inp_ff_root_pose_mat'], y_all_root_pose_mat)
        y[:, 0, :] = y_all_root_pose_mat
        y = geometry.matrix_to_axis_angle(y)


        inpaint_mask = sample['inpaint_mask']
        mask = sample['mask']
        beta = sample['beta']
        is_right = sample['is_right']
        cam = sample['cam']
        frame_ix = sample['frame_ix']

        # temporal mask
        valid_indices = torch.nonzero(mask).squeeze()
        if valid_indices.numel() > 0:
            start_idx = valid_indices[0].item() 
            end_idx = valid_indices[-1].item()

        # model
        from data_loaders.hand.vis import load_model, Renderer
        render = Renderer(height=720, width=1280, faces=None, extra_mesh=[])
        model_path = '/home/zvc/Project/motion-diffusion-model/body_models'
        if is_right:
            hand_model = load_model(
            gender='neutral', model_type='manor', model_path=model_path,
            num_pca_comps=6, use_pose_blending=True, use_shape_blending=True,
            use_pca=False, use_flat_mean=False)
        else:
            hand_model = load_model(
            gender='neutral', model_type='manol', model_path=model_path,
            num_pca_comps=6, use_pose_blending=True, use_shape_blending=True,
            use_pca=False, use_flat_mean=False)
            x0_trans[:, 0] *= -1
            y_trans[:, 0] *= -1
            x0[:, :, 1] *= -1
            x0[:, :, 2] *= -1
            y[:, :, 1] *= -1
            y[:, :, 2] *= -1

        # video frame
        # video_path = os.path.join("/home/zvc/Data/GigaHands/symlinks", sample['name'], 'rgb', sample['name'] + '.mp4')
        # vr = VideoReader(video_path, ctx=cpu(0))
        # frames_rgb = [frame.asnumpy() for frame in vr]
        from glob import glob
        import cv2
        video_path = str(sample['video_path'])
        vr = VideoReader(video_path, ctx=cpu(0))
        frames_rgb = [frame.asnumpy() for frame in vr]

        # param
        frames_gt = []
        frames_ref = []
        for idx in tqdm(range(start_idx, end_idx+1)):
            # gt
            hand_param_gt = {
                "poses": torch.cat([torch.zeros_like(x0[idx, 0, :]).unsqueeze(0), x0[idx, 1:, :]], dim=0).unsqueeze(0).reshape(1, -1).to(device), 
                "Rh": x0[idx, 0, :].unsqueeze(0).to(device),
                "Th": x0_trans[idx].unsqueeze(0).to(device),
                "shapes": beta.to(device),
            }
            vertices_gt = hand_model(return_verts=True, return_tensor=False, **hand_param_gt)[0]
            faces = hand_model.faces
            image_gt = np.zeros((720, 1280, 3))
            render_data_gt = {
                0: {'vertices': vertices_gt, 'faces': faces, 'vid': 1 if is_right else 4, 'name': f'gt_{idx}'},
            }
            render_results_gt = render.render(render_data_gt, cam, [image_gt], add_back=False)
            image_vis_gt = render_results_gt[0][:, :, [2, 1, 0, 3]]
            render_rgb_gt = image_vis_gt[:, :, :3]
            alpha_gt = image_vis_gt[:, :, 3] / 255.0
            alpha_gt = alpha_gt[:, :, np.newaxis]
            combined_image_gt = (render_rgb_gt * alpha_gt + frames_rgb[sample['frame_ix'][idx]] * (1 - alpha_gt)).astype(np.uint8)
            frames_gt.append(combined_image_gt.astype(np.uint8))

            # ref
            hand_param_ref = {
                "poses": torch.cat([torch.zeros_like(y[idx, 0, :]).unsqueeze(0), y[idx, 1:, :]], dim=0).unsqueeze(0).reshape(1, -1).to(device), 
                "Rh": y[idx, 0, :].unsqueeze(0).to(device),
                "Th": y_trans[idx].unsqueeze(0).to(device),
                "shapes": beta.to(device),
            }
            vertices_ref = hand_model(return_verts=True, return_tensor=False, **hand_param_ref)[0]
            faces = hand_model.faces
            image_ref = np.zeros((720, 1280, 3))
            render_data_ref = {
                0: {'vertices': vertices_ref, 'faces': faces, 'vid': 1 if is_right else 4, 'name': f'ref_{idx}'},
            }
            render_results_ref = render.render(render_data_ref, cam, [image_ref], add_back=False)
            image_vis_ref = render_results_ref[0][:, :, [2, 1, 0, 3]]
            render_rgb_ref = image_vis_ref[:, :, :3]
            alpha_ref = image_vis_ref[:, :, 3] / 255.0
            alpha_ref = alpha_ref[:, :, np.newaxis]
            combined_image_ref = (render_rgb_ref * alpha_ref + frames_rgb[sample['frame_ix'][idx]] * (1 - alpha_ref)).astype(np.uint8)
            frames_ref.append(combined_image_ref.astype(np.uint8))

        # Save video
        import imageio
        save_path = '/home/zvc/Project/motion-diffusion-model/_vis'
        os.makedirs(save_path, exist_ok=True)

        output_video_gt = os.path.join(save_path, f'output_gt_{sample_idx}.mp4')
        imageio.mimsave(str(output_video_gt), frames_gt, fps=30)
        print(f"Saved output video to {output_video_gt}")

        output_video_ref = os.path.join(save_path, f'output_ref_{sample_idx}.mp4')
        imageio.mimsave(str(output_video_ref), frames_ref, fps=30)
        print(f"Saved output video to {output_video_ref}")

        print(f"data name {sample['name']}")