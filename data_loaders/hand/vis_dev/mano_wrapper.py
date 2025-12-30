import sys
from pathlib import Path
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent.parent 
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
import os
import torch
import numpy as np
import pickle
from decord import VideoReader, cpu
from typing import Optional
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import MANOOutput, to_tensor
from smplx.vertex_ids import vertex_ids
import json
import utils.rotation_conversions as geometry
from data_loaders.hand.vis import Renderer as Renderer_giga
from .renderer import Renderer as Renderer_hamer
import imageio

class MANO(smplx.MANOLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official MANO implementation to support more joints.
        Args:
            Same as MANOLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(MANO, self).__init__(*args, **kwargs)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        #2, 3, 5, 4, 1
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('extra_joints_idxs', to_tensor(list(vertex_ids['mano'].values()), dtype=torch.long))
        self.register_buffer('joint_map', torch.tensor(mano_to_openpose, dtype=torch.long))
        self.pose2rot = kwargs.get('pose2rot')

    def forward(self, *args, **kwargs) -> MANOOutput:
        """
        Run forward pass. Same as MANO and also append an extra set of joints if joint_regressor_extra is specified.
        """
        mano_output = super(MANO, self).forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output


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

def get_intr(param, undistort=False):
    intr = np.eye(3)
    intr[0, 0] = param["fx_undist" if undistort else "fx"]
    intr[1, 1] = param["fy_undist" if undistort else "fy"]
    intr[0, 2] = param["cx_undist" if undistort else "cx"]
    intr[1, 2] = param["cy_undist" if undistort else "cy"]

    # TODO: Make work for arbitrary dist params in opencv
    dist = np.asarray([param["k1"], param["k2"], param["p1"], param["p2"]])

    return intr, dist


def get_rot_trans(param):
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
    r = qvec2rotmat(-qvec)
    return r, tvec

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def get_extr(param):
    r, tvec = get_rot_trans(param)
    extr = np.vstack([np.hstack([r, tvec[:, None]]), np.zeros((1, 4))])
    extr[3, 3] = 1
    extr = extr[:3]
    return extr

def get_projections(params, cam_names, n_Frames=1):
    """Returns camera intrinsics, extrinsics, projections, and distortion parameters for the named camera."""
    projs, intrs, dists, rot, trans = [], [], [], [], []
    for param in params:
        if param["cam_name"] == cam_names:
            extr = get_extr(param)
            intr, dist = get_intr(param)
            r, t = get_rot_trans(param)
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

def get_inv_ext(cameras, nv=0):
    R = cameras['R'][nv]
    T = cameras['T'][nv].reshape(3, 1)
    # E = np.eye(4)
    # E[:3, :3] = R
    # E[:3, 3:4] = T
    # E_inv = np.linalg.inv(E)
    E_inv_fast = np.eye(4)
    E_inv_fast[:3, :3] = R.T
    E_inv_fast[:3, 3:4] = -R.T @ T
    return E_inv_fast

def get_pyrender_pose(cameras, nv=0):
    E_inv = get_inv_ext(cameras, nv)
    R_flip = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return E_inv @ R_flip



if __name__ == "__main__":
    mano_cfg = {
        'data_dir': '/home/zvc/Project/VHand/_DATA/data/',
        'model_path': '/home/zvc/Project/VHand/_DATA/data/mano',
        'gender': 'neutral',
        'num_hand_joints': '15',
        'mean_params': 'data/mano_mean_params.npz',
        'create_body_pose': False,
    }
    mano = MANO(pose2rot=False, use_pca=True, **mano_cfg)

    data_path = '/home/zvc/Project/VHand/test_dataset/GigaHands/vhand/hamer_out/p001-folder_017_brics-odroid-002_cam0/results/track_500.0/track_2.pkl'
    mano_path = '/home/zvc/Data/GigaHands/hand_poses/p001-folder/params/017.json'
    cam_path = '/home/zvc/Data/GigaHands/hand_poses/p001-folder/optim_params.txt'
    video_path = '/home/zvc/Data/GigaHands/symlinks/p001-folder_017_brics-odroid-002_cam0/brics-odroid-002_cam0.mp4'
    cam = 'brics-odroid-002_cam0'
    model_path = '/home/zvc/Project/VHand/_DATA/data/mano/MANO_RIGHT.pkl'
    save_root = 'home/zvc/Project/motion-diffusion-model/_vis/mano'
    # frame_indices = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

    
    with open(data_path, 'rb') as f:
        y_data = pickle.load(f)
    with open(mano_path, 'r') as f:
        x_data = json.load(f)


    frame_indices = y_data['frame_indices']
    start_idx = frame_indices[0]
    end_idx = frame_indices[-1]
    N = len(frame_indices)


    # y
    y_betas = torch.from_numpy(np.asarray([mano['betas'] for mano in y_data['mano']])).mean(dim=0).reshape(-1, 10)
    y_hand_pose = torch.from_numpy(np.asarray([mano['hand_pose'] for mano in y_data['mano']]))
    y_global_orient = torch.from_numpy(np.asarray([mano['global_orient'] for mano in y_data['mano']]))
    y_pose_rotmat = torch.cat([y_global_orient, y_hand_pose], dim=1) # (N, 16, 3, 3)
    y_pose_rotvec = geometry.matrix_to_axis_angle(y_pose_rotmat) # (N, 16, 3)

    # x
    x_betas = torch.tensor(x_data["right"]["shapes"], dtype=torch.float32).reshape(-1, 10)
    x_poses = torch.tensor(x_data["right"]["poses"], dtype=torch.float32)[frame_indices].reshape(-1, 16, 3)
    x_Rh = torch.tensor(x_data["right"]["Rh"], dtype=torch.float32)[frame_indices].reshape(-1, 1, 3)
    x_pose_rotvec = torch.cat([x_Rh, x_poses[:, 1:]], dim=1) # (N, 16, 3, 3)
    x_pose_rotmat = geometry.axis_angle_to_matrix(x_pose_rotvec) # (N, 16, 3)


    # trans and cam (from x)
    transl = torch.tensor(x_data["right"]["Th"], dtype=torch.float32)[frame_indices].reshape(-1, 3)
    cam_params = read_params(cam_path)
    cam = get_projections(cam_params, cam, n_Frames=1)

    
    # y Canonical on first frame
    y_pose_rotvec_can = y_pose_rotvec.clone()
    first_frame_root_pose_matrix_y = geometry.axis_angle_to_matrix(y_pose_rotvec_can[0][0])
    all_root_poses_matrix_y = geometry.axis_angle_to_matrix(y_pose_rotvec_can[:, 0, :])
    aligned_root_poses_matrix_y = torch.matmul(torch.transpose(first_frame_root_pose_matrix_y, 0, 1), all_root_poses_matrix_y)
    y_pose_rotvec_can[:, 0, :] = geometry.matrix_to_axis_angle(aligned_root_poses_matrix_y)
    y_pose_rotmat_can = geometry.axis_angle_to_matrix(y_pose_rotvec_can)
    y_pose_rot6d_can = geometry.matrix_to_rotation_6d(y_pose_rotmat_can)
    


    # x Canonical on first frame
    x_pose_rotvec_can = x_pose_rotvec.clone()
    first_frame_root_pose_matrix_x = geometry.axis_angle_to_matrix(x_pose_rotvec_can[0][0])
    all_root_poses_matrix_x = geometry.axis_angle_to_matrix(x_pose_rotvec_can[:, 0, :])
    aligned_root_poses_matrix_x= torch.matmul(torch.transpose(first_frame_root_pose_matrix_x, 0, 1), all_root_poses_matrix_x)
    x_pose_rotvec_can[:, 0, :] = geometry.matrix_to_axis_angle(aligned_root_poses_matrix_x)
    x_pose_rotmat_can = geometry.axis_angle_to_matrix(x_pose_rotvec_can)
    x_pose_rot6d_can = geometry.matrix_to_rotation_6d(x_pose_rotmat_can)

    # y mano rec with mano wrapper
    y_input_mano_wrapper = {
        'global_orient': y_pose_rotmat_can[:, 0].unsqueeze(dim=1),
        'hand_pose': y_pose_rotmat_can[:, 1:],
        'betas': y_betas.repeat(N, 1),
        'transl': transl,
    }
    y_output_mano_wrapper = mano(**y_input_mano_wrapper, pose2rot=False)

    # x mano rec  with mano wrapper
    x_input_mano_wrapper = {
        'global_orient': x_pose_rotmat_can[:, 0].unsqueeze(dim=1),
        'hand_pose': x_pose_rotmat_can[:, 1:],
        'betas': x_betas.repeat(N, 1),
        'transl': transl,
    }
    x_output_mano_wrapper = mano(**x_input_mano_wrapper, pose2rot=False)


    # video
    vr = VideoReader(video_path, ctx=cpu(0))
    frames_rgb = [frame.asnumpy() for frame in vr]
    h, w = frames_rgb[0].shape[:2]

    # faces
    with open(model_path, 'rb') as mano_file:
        mano_model = pickle.load(mano_file, encoding='latin1')
    faces = mano_model['f']

    # renderer
    render_giga = Renderer_giga(height=720, width=1280, faces=None, extra_mesh=[])
    render_hamer = Renderer_hamer(faces=faces)

    # iter
    y_mano_wrapper_render_hamer = []
    x_mano_wrapper_render_hamer = []

    for i, idx in enumerate(range(frame_indices)):
        # img
        input_img = frames_rgb[idx].astype(np.float32)/ 255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
        
        # with render_hamer
        misc_args = dict(
            mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
            scene_bg_color=(1, 1, 1),
            focal_length=[cam['K'][0][0, 0], cam['K'][0][1, 1]],
            camera_center=[cam['K'][0][0, 2], cam['K'][0][1, 2]],
            camera_pose=get_pyrender_pose(cam),
            cam_t=[0, 0, 0], 
            render_res=[h, w], 
            is_right=1, 
        )

        # y render
        cam_view_y_hamer, _ = render_hamer.render_rgba_multiple(
            y_output_mano_wrapper.vertices[i], 
            **misc_args
        )
        output_img_y_hamer = input_img[:, :, :3] * (1 - cam_view_y_hamer[:, :, 3:]) + cam_view_y_hamer[:, :, :3] * cam_view_y_hamer[:, :, 3:]
        y_mano_wrapper_render_hamer.append(output_img_y_hamer)

         # x render
        cam_view_x_hamer, _ = render_hamer.render_rgba_multiple(
            x_output_mano_wrapper.vertices[i], 
            **misc_args
        )
        output_img_x_hamer = input_img[:, :, :3] * (1 - cam_view_x_hamer[:, :, 3:]) + cam_view_x_hamer[:, :, :3] * cam_view_x_hamer[:, :, 3:]
        x_mano_wrapper_render_hamer.append(output_img_x_hamer)

    os.makedirs(save_root, exist_ok=True)
    y_mano_wrapper_render_hamer_output_video = os.path.join(save_root, 'y_mano_wrapper_render_hamer.mp4')
    imageio.mimsave(str(y_mano_wrapper_render_hamer_output_video), y_mano_wrapper_render_hamer, fps=30)

    x_mano_wrapper_render_hamer_output_video = os.path.join(save_root, 'x_mano_wrapper_render_hamer.mp4')
    imageio.mimsave(str(x_mano_wrapper_render_hamer_output_video), x_mano_wrapper_render_hamer, fps=30)