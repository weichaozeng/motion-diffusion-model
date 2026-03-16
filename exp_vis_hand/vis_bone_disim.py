import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model_hand.rotation2xyz import Rotation2xyz



BONE_CONNECTIONS = torch.tensor([
    [0, 1], [1, 2], [2, 3], [3, 4],        # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],        # Index
    [0, 9], [9, 10], [10, 11], [11, 12],   # Middle
    [0, 13], [13, 14], [14, 15], [15, 16], # Ring
    [0, 17], [17, 18], [18, 19], [19, 20]  # Pinky
], dtype=torch.int64)

parent_idx = BONE_CONNECTIONS[:, 0]
child_idx = BONE_CONNECTIONS[:, 1]


import pickle
import os

def get_mano_faces(model_path='/home/zvc/Project/VHand/_DATA/data/mano/MANO_RIGHT.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Can't Find in: {model_path}")   
    with open(model_path, 'rb') as mano_file:
        mano_model = pickle.load(mano_file, encoding='latin1')
    faces = mano_model['f']
    return faces


def generate_mano_poses(rot2xyz_model, num_poses=5, device='cpu'):
    """
    使用 MANO 模型生成从平展到握拳的序列
    """
    B = 1 # Batch size
    F = num_poses # Frames
    
    poses = torch.zeros((B, 16, 3, F), device=device)
    
    # 模拟握拳过程
    for i in range(F):
        # 最大弯曲约 1.5 弧度（接近 90 度）
        bend_angle = - i * (1.5 / (F - 1)) 
        
        # 【关键修改】：改变旋转轴
        # MANO 中，四指（食指、中指、无名指、小指）的屈伸通常在 Y 轴 (1) 或 Z 轴 (2)
        # 这里的符号 (+/-) 和轴 (1 或 2) 取决于具体的右手坐标系方向。
        # 如果发现手指是向手背翻折的，将 bend_angle 前面加个负号即可。
        poses[0, 1:13, 1, i] = bend_angle  # 四指沿 Y 轴弯曲
        
        # 拇指（索引 13~15）的朝向与四指不同，通常在 Z 轴上弯曲更容易向掌心内收
        poses[0, 13:16, 2, i] = -bend_angle 
        
    beta = torch.zeros((B, 10), device=device) 
    translation = torch.zeros((B, F, 3), device=device) 
    
    with torch.no_grad():
        joints, vertices = rot2xyz_model(
            pose=poses, 
            pose_rep="rotvec", 
            beta=beta, 
            translation=translation, 
            return_vertices=True
        )
        
    joints = joints.squeeze(0).permute(2, 0, 1).cpu() 
    vertices = vertices.squeeze(0).cpu()
    
    return joints, vertices


def calculate_dissim(kps):
    F_frames = kps.shape[0]
    box_scales = torch.ones((F_frames, 1, 1)) 
    
    rel_vecs = (kps[:, child_idx, :] - kps[:, parent_idx, :]) / box_scales.clamp(min=1e-6)
    bone_vecs_norm = F.normalize(rel_vecs, p=2, dim=2) 
    
    anchor_vecs_norm = bone_vecs_norm[0:1] # 获取第 0 帧作为基准
    
    cos_sim = torch.sum(anchor_vecs_norm * bone_vecs_norm, dim=2) 
    bone_dissim = 1.0 - (cos_sim + 1.0) / 2.0 
    pose_dissim = torch.mean(bone_dissim, dim=1) 
    
    return pose_dissim, bone_dissim


def plot_mano_poses(joints_tensor, vertices_tensor, faces, dissim_scores, save_path="mano_dissim_separated.png"):
    F_frames = joints_tensor.shape[0]
    
    # 1. 【画布优化】调整画布比例，适应左右两个子图
    fig = plt.figure(figsize=(10, 4.5 * F_frames)) 
    
    for i in range(F_frames):
        pose = joints_tensor[i].numpy()
        verts = vertices_tensor[i].numpy()
        
        # --- 创建左侧子图：专门用于绘制骨架 ---
        ax_skeleton = fig.add_subplot(F_frames, 2, 2 * i + 1, projection='3d')
        
        # 【特征高亮】在独立空间绘制骨架
        for bone_idx, (p_idx, c_idx) in enumerate(BONE_CONNECTIONS.numpy()):
            ax_skeleton.plot([pose[p_idx, 0], pose[c_idx, 0]],
                             [pose[p_idx, 1], pose[c_idx, 1]],
                             [pose[p_idx, 2], pose[c_idx, 2]],
                             c="#1C526A", linewidth=1.5, zorder=5) # 鲜艳的红色线条
            
        ax_skeleton.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c="#20DA80", s=15, zorder=10)
        
        # 设置骨架空间的 Bounding Box
        x_min_s, x_max_s = pose[:, 0].min(), pose[:, 0].max()
        y_min_s, y_max_s = pose[:, 1].min(), pose[:, 1].max()
        z_min_s, z_max_s = pose[:, 2].min(), pose[:, 2].max()
        
        max_range_s = np.array([x_max_s-x_min_s, y_max_s-y_min_s, z_max_s-z_min_s]).max() / 2.0
        mid_x_s = (x_max_s+x_min_s) * 0.5
        mid_y_s = (y_max_s+y_min_s) * 0.5
        mid_z_s = (z_max_s+z_min_s) * 0.5
        
        ax_skeleton.set_xlim(mid_x_s - max_range_s, mid_x_s + max_range_s)
        ax_skeleton.set_ylim(mid_y_s - max_range_s, mid_y_s + max_range_s)
        ax_skeleton.set_zlim(mid_z_s - max_range_s, mid_z_s + max_range_s)
        
        ax_skeleton.set_box_aspect([1, 1, 1]) 
        ax_skeleton.view_init(elev=-90, azim=-90) # 保持俯视
        ax_skeleton.dist = 7.5 
        ax_skeleton.set_axis_off() 
        
        # 左侧标题
        title_color_s = '#2E8B57' if i == 0 else '#B22222'
        ax_skeleton.set_title(f"Pose {i} | Skeleton", 
                              color=title_color_s, fontsize=12, pad=-15, fontweight='bold')
        
        # --- 创建右侧子图：专门用于绘制模型 ---
        ax_model = fig.add_subplot(F_frames, 2, 2 * i + 2, projection='3d')
        
        # 【材质优化】开启光影 (shade=True)，换用高级的莫兰迪蓝
        ax_model.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                              triangles=faces, 
                              color='#A0C4FF',  # 舒适的淡蓝色
                              alpha=0.6,       # 增加不透明度，使模型更清晰
                              edgecolor='none', # 去掉网格线
                              shade=True,       # 开启光照
                              zorder=1)
        
        # 设置模型空间的 Bounding Box
        x_min_m, x_max_m = verts[:, 0].min(), verts[:, 0].max()
        y_min_m, y_max_m = verts[:, 1].min(), verts[:, 1].max()
        z_min_m, z_max_m = verts[:, 2].min(), verts[:, 2].max()
        
        max_range_m = np.array([x_max_m-x_min_m, y_max_m-y_min_m, z_max_m-z_min_m]).max() / 2.0
        mid_x_m = (x_max_m+x_min_m) * 0.5
        mid_y_m = (y_max_m+y_min_m) * 0.5
        mid_z_m = (z_max_m+z_min_m) * 0.5
        
        ax_model.set_xlim(mid_x_m - max_range_m, mid_x_m + max_range_m)
        ax_model.set_ylim(mid_y_m - max_range_m, mid_y_m + max_range_m)
        ax_model.set_zlim(mid_z_m - max_range_m, mid_z_m + max_range_m)
        
        ax_model.set_box_aspect([1, 1, 1]) 
        ax_model.view_init(elev=-90, azim=-90) # 保持俯视
        ax_model.dist = 7.5 
        ax_model.set_axis_off() 
        
        # 右侧标题，包含 Dis_sim 信息
        title_color_m = '#2E8B57' if i == 0 else '#B22222'
        ax_model.set_title(f"Pose {i} | Model | Dis_sim: {dissim_scores[i]:.4f}", 
                           color=title_color_m, fontsize=12, pad=-15, fontweight='bold')

    # 【消除子图间距】让图片紧凑排列
    plt.subplots_adjust(hspace=0.0, wspace=0.1) 
    
    # 保存透明背景的高清图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    rot2xyz_model = Rotation2xyz(device=device)
    mano_pkl_path = '/home/zvc/Project/VHand/_DATA/data/mano/MANO_RIGHT.pkl'
    faces = get_mano_faces(mano_pkl_path)
   
    joints_kps, vertices_kps = generate_mano_poses(rot2xyz_model, num_poses=5, device=device)

    pose_dissim, _ = calculate_dissim(joints_kps)
    

    for i, score in enumerate(pose_dissim):
        print(f"Pose {i} vs Pose 0: {score.item():.4f}")
    plot_mano_poses(joints_kps, vertices_kps, faces, pose_dissim)