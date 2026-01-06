from .renderer import Renderer
import pickle
import os
import torch
from decord import VideoReader, cpu
import numpy as np
import imageio

def render_video(verts, out_dir, rgb_video_paths, rgb_frame_indices, cams, suffix_masks):
    assert len(verts) ==  len(rgb_video_paths) and len(verts) == len(rgb_frame_indices) and len(verts) == len(cams['K']), f"Length of verts, rgb_video_paths, rgb_frame_indices and cams must be the same."

    model_path = '/home/zvc/Project/VHand/_DATA/data/mano/MANO_RIGHT.pkl'
    with open(model_path, 'rb') as mano_file:
        mano_model = pickle.load(mano_file, encoding='latin1')
    faces = mano_model['f']
    renderer = Renderer(height=720, width=1280, faces=None, extra_mesh=[])

    for i in range(len(verts)):
        vertices = verts[i].cpu().numpy() 
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
        out_video_path = os.path.join(out_dir, f'{video_name}_rendered.mp4')
        
        valid_indices = torch.nonzero(~suffix_mask).squeeze()
        if valid_indices.numel() > 0:
            start_idx = valid_indices[0].item() 
            end_idx = valid_indices[-1].item()

        vr = VideoReader(video_path, ctx=cpu(0))
        frames_rgb = [frame.asnumpy() for frame in vr]
        frames_render = []

        for idx in range(start_idx, end_idx+1):
            rgb_img = frames_rgb[frame_indices[idx].item()]
            bg_img = np.zeros((720, 1280, 3))
            render_data = {
                0: {
                    'vertices': vertices[idx],
                    'faces': faces,
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
            frames_render.append(vis_img.astype(np.uint8))
        imageio.mimsave(str(out_video_path), frames_render, fps=30)