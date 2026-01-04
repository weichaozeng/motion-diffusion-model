from dataset import Dataset
import torch
import os
import json
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import vis.camparam_utils as param_utils
from pathlib import Path
import utils_hand.rotation_conversions as geometry

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
        outs = sorted(os.listdir(data_path))

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
                
    
        
        self._train = list(range(100, len(self.seqs_y)))
        self._val = list(range(0, 100))

    def _load_cam(self, ind):
        return self.seqs_cam[ind]

    def _load_rotvec_x(self, ind, frame_ix, x_data, is_right, flip_left=True):
        # mano_params_path = self.seqs_mano[ind]
        # with open(mano_params_path, 'r') as f:
        #     mano_data = json.load(f)

        if is_right:
            full_poses = torch.tensor(x_data["right"]["poses"], dtype=torch.float32)
            Rh = torch.tensor(x_data["right"]["Rh"], dtype=torch.float32)
            beta = torch.tensor(x_data["right"]["shapes"], dtype=torch.float32)
        else:
            full_poses = torch.tensor(x_data["left"]["poses"], dtype=torch.float32) 
            Rh = torch.tensor(x_data["left"]["Rh"], dtype=torch.float32)
            beta = torch.tensor(x_data["left"]["shapes"], dtype=torch.float32)
        full_poses = torch.cat([Rh, full_poses[:, 3:]], dim=1) # (num_frames, 16*3)
        poses = full_poses[frame_ix].reshape(-1, 16, 3)  # (num_frames, 16, 3)
        if not is_right and flip_left:
            raise NotImplementedError
            # poses[:, :, 1] *= -1
            # poses[:, :, 2] *= -1
    
        return poses, beta.squeeze(0)        
    
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
    
    def _load_translation_x(self, ind, frame_ix, x_data, is_right, flip_left=True):

        if is_right:
            Th = torch.tensor(x_data["right"]["Th"], dtype=torch.float32)
        else:
            Th = torch.tensor(x_data["left"]["Th"], dtype=torch.float32)
        if not is_right and flip_left:
            raise NotImplementedError
            # Th[:, 0] *= -1
        return Th[frame_ix]  
    
    def _load_rotvec_y(self, ind, frame_ix, y_data, cam):
        # seq_y_path = self.seqs_y[ind]
        frame_indices = y_data['frame_indices']
        hand_pose = torch.from_numpy(np.asarray([mano['hand_pose'] for mano in y_data['mano']]))
        global_orient = torch.from_numpy(np.asarray([mano['global_orient'] for mano in y_data['mano']]))
        # corret
        boxes = torch.from_numpy(np.asarray(y_data['boxes']))
        crop_centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        global_orient_corrected, R_c2w, R_adj = self.get_hamer_to_world_orient(
            global_orient, 
            cam['R'][0], 
            crop_centers,        
            cam['K'][0]
        )
        pose_mean = torch.tensor(np.asarray(MANO_HANDS_MEAN), dtype=torch.float32)
        hand_pose_rotvec = geometry.matrix_to_axis_angle(hand_pose)
        hand_pose_corrected = hand_pose_rotvec - pose_mean
        hand_pose_corrected = geometry.axis_angle_to_matrix(hand_pose_corrected)

        global_orient_corrected = global_orient_corrected.numpy()
        hand_pose_corrected = hand_pose_corrected.numpy()

        # slerp
        indices = np.searchsorted(frame_indices, [frame_ix[0], frame_ix[-1]])
        start_idx = indices[0]
        end_idx = indices[1]
        full_pose_rotvec, inpaint_mask = self._slerp_y(frame_indices[start_idx:end_idx+1], global_orient_corrected[start_idx:end_idx+1], hand_pose_corrected[start_idx:end_idx+1])

        relative_indices = frame_ix - frame_indices[start_idx]
        max_relative_idx = full_pose_rotvec.shape[0] - 1
        relative_indices = np.clip(relative_indices, 0, max_relative_idx)

        return full_pose_rotvec[relative_indices], inpaint_mask[relative_indices], R_c2w, R_adj



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

    def get_hamer_to_world_orient(self, y_global_orient, cam_extrinsic, crop_center, cam_intrinsics):
        N = y_global_orient.shape[0]
        device = y_global_orient.device

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device).float()
            return torch.from_numpy(x).to(device).float()
        
        R_w2c = to_tensor(cam_extrinsic)
        K = to_tensor(cam_intrinsics)
        centers = to_tensor(crop_center)

        R_w2c = R_w2c[:3, :3]
        R_c2w = R_w2c.t() 

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ux, uy = centers[:, 0], centers[:, 1]
        theta_y = torch.atan((ux - cx) / fx)   
        theta_x = -torch.atan((uy - cy) / fy)  

        # R_adj (N, 3, 3)
        cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)
        cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
        ones = torch.ones_like(theta_y)
        zeros = torch.zeros_like(theta_y)

        Ry = torch.stack([
            torch.stack([cos_y,  zeros, sin_y], dim=-1),
            torch.stack([zeros,  ones,  zeros], dim=-1),
            torch.stack([-sin_y, zeros, cos_y], dim=-1)
        ], dim=-2)

        Rx = torch.stack([
            torch.stack([ones,  zeros,  zeros], dim=-1),
            torch.stack([zeros, cos_x, -sin_x], dim=-1),
            torch.stack([zeros, sin_x,  cos_x], dim=-1)
        ], dim=-2)

        R_adj = Ry @ Rx 

        # R_world = R_c2w @ R_adj @ R_hamer
        y_global_orient_world = R_c2w.unsqueeze(0) @ R_adj @ y_global_orient.squeeze(1)
        
        return y_global_orient_world.unsqueeze(1), R_c2w, R_adj


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
    dataset = GigaHands(split="train", num_frames=128, sampling="conseq", sampling_step=1, pose_rep="rot6d", translation=True, align_pose_frontview=True)
    print(len(dataset))
    for sample_idx in range(10):
        sample = dataset[sample_idx]
        import utils.rotation_conversions as geometry
        x_trans = sample['x_trans'].permute(1, 0)
        x_root_trans = sample['x_root_trans']
        x_trans += x_root_trans
        x_pose_rot6d = sample['x_pose'].permute(2, 0, 1)
        x_pose_rotmat = geometry.rotation_6d_to_matrix(x_pose_rot6d)

        # y_trans = sample['ref_trans'].permute(1, 0)   
        # y_root = sample['ref_motion_root']
        # y_trans += y_root
        y_trans = x_trans.clone()
        y_pose_rot6d = sample['y_pose'].permute(2, 0, 1)
        y_pose_rotmat = geometry.rotation_6d_to_matrix(y_pose_rot6d)


        
        #align_pose_frontview[
        x_all_root_pose_mat = x_pose_rotmat[:, 0,]
        x_all_root_pose_mat = torch.matmul(sample['x_ff_root_orient_rotmat'], x_all_root_pose_mat)
        x_pose_rotmat[:, 0,] = x_all_root_pose_mat
        x_pose_rotvec = geometry.matrix_to_axis_angle(x_pose_rotmat)

        y_all_root_pose_mat = y_pose_rotmat[:, 0,]
        y_all_root_pose_mat = torch.matmul(sample['y_ff_root_orient_rotmat'], y_all_root_pose_mat)
        y_pose_rotmat[:, 0,] = y_all_root_pose_mat
        y_pose_rotvec = geometry.matrix_to_axis_angle(y_pose_rotmat)


        inpaint_mask = sample['inpaint_mask']
        suffix_mask = sample['suffix_mask']
        x_beta = sample['x_beta']
        is_right = sample['is_right']
        cam = sample['cam']
        frame_indices = sample['frame_indices']

        # temporal mask
        valid_indices = torch.nonzero(suffix_mask).squeeze()
        if valid_indices.numel() > 0:
            start_idx = valid_indices[0].item() 
            end_idx = valid_indices[-1].item()

        # model
        from data_loaders_hand.hand.vis import load_model, Renderer
        render = Renderer(height=720, width=1280, faces=None, extra_mesh=[])
        model_path = '/home/zvc/Project/motion-diffusion-model/body_models'
        if is_right:
            hand_model = load_model(
            gender='neutral', model_type='manor', model_path=model_path,
            num_pca_comps=6, use_pose_blending=True, use_shape_blending=True,
            use_pca=False, use_flat_mean=False)
        else:
            raise NotImplementedError
            # hand_model = load_model(
            # gender='neutral', model_type='manol', model_path=model_path,
            # num_pca_comps=6, use_pose_blending=True, use_shape_blending=True,
            # use_pca=False, use_flat_mean=False)
            # x0_trans[:, 0] *= -1
            # y_trans[:, 0] *= -1
            # x0[:, :, 1] *= -1
            # x0[:, :, 2] *= -1
            # y[:, :, 1] *= -1
            # y[:, :, 2] *= -1

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
                "poses": torch.cat([torch.zeros_like(x_pose_rotvec[idx, 0, :]).unsqueeze(0), x_pose_rotvec[idx, 1:, :]], dim=0).unsqueeze(0).reshape(1, -1).to(device), 
                "Rh": x_pose_rotvec[idx, 0, :].unsqueeze(0).to(device),
                "Th": x_trans[idx].unsqueeze(0).to(device),
                "shapes": x_beta.unsqueeze(0).to(device),
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
            combined_image_gt = (render_rgb_gt * alpha_gt + frames_rgb[sample['frame_indices'][idx]] * (1 - alpha_gt)).astype(np.uint8)
            frames_gt.append(combined_image_gt.astype(np.uint8))

            # ref
            hand_param_ref = {
                "poses": torch.cat([torch.zeros_like(y_pose_rotvec[idx, 0, :]).unsqueeze(0), y_pose_rotvec[idx, 1:, :]], dim=0).unsqueeze(0).reshape(1, -1).to(device), 
                "Rh": y_pose_rotvec[idx, 0, :].unsqueeze(0).to(device),
                "Th": x_trans[idx].unsqueeze(0).to(device),
                "shapes": x_beta.unsqueeze(0).to(device),
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
            combined_image_ref = (render_rgb_ref * alpha_ref + frames_rgb[sample['frame_indices'][idx]] * (1 - alpha_ref)).astype(np.uint8)
            frames_ref.append(combined_image_ref.astype(np.uint8))

        # Save video
        import imageio
        save_path = '/home/zvc/Project/motion-diffusion-model/_vis/datasets/'
        os.makedirs(save_path, exist_ok=True)

        output_video_gt = os.path.join(save_path, f'x_{sample_idx}.mp4')
        imageio.mimsave(str(output_video_gt), frames_gt, fps=30)
        print(f"Saved output video to {output_video_gt}")

        output_video_ref = os.path.join(save_path, f'y_{sample_idx}.mp4')
        imageio.mimsave(str(output_video_ref), frames_ref, fps=30)
        print(f"Saved output video to {output_video_ref}")

        print(f"data name {sample['name']}")