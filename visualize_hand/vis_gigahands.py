from .renderer import Renderer
import pickle
import os
import torch
from decord import VideoReader, cpu
import numpy as np
import imageio
import cv2

skeleton_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),        
        (0, 5), (5, 6), (6, 7), (7, 8),       
        (0, 9), (9, 10), (10, 11), (11, 12),   
        (0, 13), (13, 14), (14, 15), (15, 16), 
        (0, 17), (17, 18), (18, 19), (19, 20)  
    ]
finger_colors = [
        (255, 50, 50)] * 4 + \
        [(255, 153, 51)] * 4 + \
        [(50, 255, 50)] * 4 + \
        [(51, 255, 255)] * 4 + \
        [(51, 51, 255)] * 4

def render_video(verts, xyzs, out_dir, rgb_video_paths, rgb_frame_indices, cams, suffix_masks):
    assert len(verts) ==  len(rgb_video_paths) and len(verts) == len(rgb_frame_indices) and len(verts) == len(cams['K']), f"Length of verts, rgb_video_paths, rgb_frame_indices and cams must be the same."

    model_path = '/home/zvc/Project/VHand/_DATA/data/mano/MANO_RIGHT.pkl'
    with open(model_path, 'rb') as mano_file:
        mano_model = pickle.load(mano_file, encoding='latin1')
    faces = mano_model['f']
    renderer = Renderer(height=720, width=1280, faces=None, extra_mesh=[])


    for i in range(len(verts)):
        vertices = verts[i].cpu().numpy() 
        joints = xyzs[i].cpu().numpy()
        video_path = rgb_video_paths[i]
        frame_indices = rgb_frame_indices[i]
        suffix_mask = suffix_masks[i]
        cam = {
                'K': cams['K'][i].numpy(),
                'R': cams['R'][i].numpy(),
                'T': cams['T'][i].numpy(),
                "dist": cams['dist'][i].numpy(),
                'P': cams['P'][i].numpy(),
        }
        video_name = os.path.basename(video_path).split('.')[0]
        os.makedirs(out_dir, exist_ok=True)
        out_video_path_mesh = os.path.join(out_dir, f'{str(i).zfill(3)}_{video_name}_rendered.mp4')
        out_video_path_joints = os.path.join(out_dir, f'{str(i).zfill(3)}_{video_name}_joints.mp4')
        
        valid_indices = torch.nonzero(~suffix_mask).squeeze()
        if valid_indices.numel() > 0:
            start_idx = valid_indices[0].item() 
            end_idx = valid_indices[-1].item()

        vr = VideoReader(video_path, ctx=cpu(0))
        frames_rgb = [frame.asnumpy() for frame in vr]
        frames_mesh = []
        frames_joints = []

        for idx in range(start_idx, end_idx+1):
            rgb_img = frames_rgb[frame_indices[idx].item()]
            # mesh video
            bg_img = np.zeros((720, 1280, 3))
            render_data = {
                0: {
                    'vertices': vertices[idx],
                    'faces': faces,
                    'joints': joints[:, :, idx],
                    'vid': 1,
                    'name': f'{i}'
                }
            }
            render_result = renderer.render(render_data, cam, [bg_img.copy()], add_back=False)
            res_img = render_result[0][:, :, [2, 1, 0, 3]]
            vis_img = res_img[:, :, :3]
            alpha = res_img[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]
            vis_img = (vis_img * alpha + rgb_img * (1 - alpha)).astype(np.uint8)
            frames_mesh.append(vis_img.astype(np.uint8))
            # joints video
            joints_3d = joints[:, :, idx]
            R_mat = cam['R']
            T_vec = cam['T'].reshape(1, 3)
            K_mat = cam['K']
            j_cam = np.dot(joints_3d, R_mat.T) + T_vec
            j_img = np.dot(j_cam, K_mat.T)
            z = j_img[:, 2:3] + 1e-6
            j_2d = j_img[:, :2] / z
            joint_img = rgb_img.copy()
            # draw
            for edge, color in zip(skeleton_edges, finger_colors):
                pt1 = (int(j_2d[edge[0], 0]), int(j_2d[edge[0], 1]))
                pt2 = (int(j_2d[edge[1], 0]), int(j_2d[edge[1], 1]))
                cv2.line(joint_img, pt1, pt2, color, thickness=3, lineType=cv2.LINE_AA)
            for pt in j_2d:
                cv2.circle(joint_img, (int(pt[0]), int(pt[1])), radius=4, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            frames_joints.append(joint_img.astype(np.uint8))


        imageio.mimsave(str(out_video_path_mesh), frames_mesh, fps=30)
        imageio.mimsave(str(out_video_path_joints), frames_joints, fps=30)