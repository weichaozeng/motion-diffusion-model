import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pickle
import math

# 如果这行报错，请确保你的环境路径正确
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

def get_mano_faces(model_path='/home/zvc/Project/VHand/_DATA/data/mano/MANO_RIGHT.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Can't Find in: {model_path}")   
    with open(model_path, 'rb') as mano_file:
        mano_model = pickle.load(mano_file, encoding='latin1')
    return mano_model['f']

def define_manual_poses():
    F = 5 
    poses = torch.zeros((F, 16, 3)) 
    # Pose 0
    poses[0, 1:13, 2] = -0.4   
    poses[0, 13:16, 2] = 0.0   
    # Pose 1
    poses[1, 1:13, 2] = -0.2   
    poses[1, 13:16, 2] = 0.0   
    # Pose 2
    poses[2, 1:13, 2] = -0.0   
    poses[2, 13:16, 2] = 0.0   
    # Pose 3
    poses[3, 1:13, 2] = 0.2   
    poses[3, 13:16, 2] = 0.0   
    # Pose 4
    poses[4, 1:13, 2] = 0.4   
    poses[4, 13:16, 2] = 0.0   
    return poses

def generate_mano_poses(rot2xyz_model, custom_poses, device='cpu'):
    B = 1 
    F = custom_poses.shape[0]
    poses = custom_poses.unsqueeze(0).permute(0, 2, 3, 1).to(device)
    beta = torch.zeros((B, 10), device=device) 
    translation = torch.zeros((B, F, 3), device=device) 
    
    with torch.no_grad():
        joints, vertices = rot2xyz_model(
            pose=poses, pose_rep="rotvec", beta=beta, 
            translation=translation, return_vertices=True
        )
    joints = joints.squeeze(0).permute(2, 0, 1).cpu() 
    vertices = vertices.squeeze(0).cpu()
    return joints, vertices

# =======================================================
# 新增：严格的 3D 到 2D 投影函数
# =======================================================
def project_points(points_3d, K, R, T):
    """
    将 3D 点投影到 2D 像素坐标系，并返回深度 Z 以供渲染排序
    points_3d: Shape (F, N, 3)
    """
    T = T.view(1, 1, 3) # 适配广播
    # 1. 转到相机坐标系
    pts_cam = torch.matmul(points_3d, R.T) + T
    # 2. 转到图像坐标系
    pts_img = torch.matmul(pts_cam, K.T)
    # 3. 透视除法
    z = pts_img[..., 2:3] + 1e-6
    pts_2d = pts_img[..., :2] / z
    return pts_2d, pts_cam[..., 2] # 返回 2D 坐标和相机空间深度

# =======================================================
# 修改：完全基于 2D 投影平面的区分度计算
# =======================================================
def calculate_dissim_2d(kps_2d):
    F_frames = kps_2d.shape[0]
    box_scales = torch.ones((F_frames, 1, 1)) 
    
    # 这里的 rel_vecs 现在是 2D 图像上的像素向量 (X, Y)
    rel_vecs = (kps_2d[:, child_idx, :] - kps_2d[:, parent_idx, :]) / box_scales.clamp(min=1e-6)
    bone_vecs_norm = F.normalize(rel_vecs, p=2, dim=2) 
    
    anchor_vecs_norm = bone_vecs_norm[0:1] 
    cos_sim = torch.sum(anchor_vecs_norm * bone_vecs_norm, dim=2) 
    bone_dissim = 1.0 - (cos_sim + 1.0) / 2.0 
    pose_dissim = torch.mean(bone_dissim, dim=1) 
    
    return pose_dissim, bone_dissim

# =======================================================
# 极客级 2D 渲染器 (Painter's Algorithm)
# =======================================================
def plot_mano_2d_projection(joints_2d, vertices_3d, faces, K, R, T, dissim_scores, save_path="mano_dissim_2d.png"):
    F_frames = joints_2d.shape[0]
    fig = plt.figure(figsize=(10, 4.5 * F_frames)) 
    
    # 图像分辨率 (根据 K 矩阵的中心点推算)
    img_w, img_h = int(K[0, 2].item() * 2), int(K[1, 2].item() * 2)

    for i in range(F_frames):
        j_2d = joints_2d[i].numpy()
        v_3d = vertices_3d[i:i+1] # 保持 batch 维度以复用 project_points
        
        # 将 Mesh 顶点投影到 2D
        v_2d, v_depth = project_points(v_3d, K, R, T)
        v_2d = v_2d[0].numpy()
        v_depth = v_depth[0].numpy()
        
        # --- 创建左侧子图：2D 投影骨架 ---
        ax_skeleton = fig.add_subplot(F_frames, 2, 2 * i + 1)
        for p_idx, c_idx in BONE_CONNECTIONS.numpy():
            ax_skeleton.plot([j_2d[p_idx, 0], j_2d[c_idx, 0]],
                             [j_2d[p_idx, 1], j_2d[c_idx, 1]],
                             c="#1C526A", linewidth=2.0)
        ax_skeleton.scatter(j_2d[:, 0], j_2d[:, 1], c="#20DA80", s=25, zorder=10)
        
        # 锁定相机视角域 (像素范围)
        ax_skeleton.set_xlim([0, img_w])
        ax_skeleton.set_ylim([img_h, 0]) # 像素坐标系 Y 轴朝下
        ax_skeleton.set_aspect('equal')
        ax_skeleton.set_axis_off() 
        ax_skeleton.set_title(f"Pose {i} | 2D Camera View Skeleton", color='#2E8B57' if i == 0 else '#B22222', fontweight='bold')
        
        # --- 创建右侧子图：2D 软光栅化 Mesh ---
        ax_model = fig.add_subplot(F_frames, 2, 2 * i + 2)
        
        # 提取面片的 2D 坐标和平均深度
        face_verts_2d = v_2d[faces] # (num_faces, 3, 2)
        face_z = v_depth[faces].mean(axis=1) # 面片的相机 Z 深度
        
        # 【深度排序】：将离相机远的面片先画，近的面片后画 (Painter's Algorithm)
        sort_idx = np.argsort(face_z)[::-1] 
        sorted_face_verts_2d = face_verts_2d[sort_idx]
        
        # 【基础光影计算】：使用相机坐标系的法向量
        v_cam = (torch.matmul(v_3d[0], R.T) + T).numpy()
        v0, v1, v2 = v_cam[faces[:, 0]], v_cam[faces[:, 1]], v_cam[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6)
        
        # 光源方向 (假设来自相机前方)
        light_dir = np.array([0, 0, -1.0])
        # 使用 abs 实现双面光照，避免内部面片变全黑
        intensity = np.abs(np.dot(normals, light_dir)) 
        
        # 上色：莫兰迪蓝底色 + 光照强度
        base_color = np.array([160/255, 196/255, 255/255])
        face_colors = base_color * (0.4 + 0.6 * intensity[:, None]) # Ambient + Diffuse
        sorted_face_colors = np.clip(face_colors[sort_idx], 0, 1)

        # 使用 PolyCollection 一次性渲染所有面片，速度极快
        poly = PolyCollection(sorted_face_verts_2d, facecolors=sorted_face_colors, edgecolors='none', antialiased=True)
        ax_model.add_collection(poly)
        
        ax_model.set_xlim([0, img_w])
        ax_model.set_ylim([img_h, 0])
        ax_model.set_aspect('equal')
        ax_model.set_axis_off() 
        ax_model.set_title(f"Pose {i} | 2D Projected Dis_sim: {dissim_scores[i]:.4f}", color='#2E8B57' if i == 0 else '#B22222', fontweight='bold')

    plt.subplots_adjust(hspace=0.1, wspace=0.1) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    rot2xyz_model = Rotation2xyz(device=device)
    mano_pkl_path = '/home/zvc/Project/VHand/_DATA/data/mano/MANO_RIGHT.pkl'
    faces = get_mano_faces(mano_pkl_path)
   
    custom_poses = define_manual_poses()
    joints_3d, vertices_3d = generate_mano_poses(rot2xyz_model, custom_poses, device=device)

    # =======================================================
    # 定义虚拟相机参数 (在这里调整你的相机位置和角度！)
    # =======================================================
    # 焦距 1000，中心点设在 (500, 500)，代表输出 1000x1000 的正方形画面
    K = torch.tensor([[1000.0, 0.0, 500.0],
                      [0.0, 1000.0, 500.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    
    # 【旋转相机】：绕 Y 轴旋转相机视角 (例如 np.radians(45) 就是倾斜 45 度看)
    # 如果全为 0，则是正对着手掌背部或掌心观察。
    angle_x = np.radians(0)  # 俯仰
    angle_y = np.radians(45)  # 左右转头 <--- 修改这个值看不同视角
    angle_z = np.radians(0)  # 偏航
    
    cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
    R = torch.tensor([
        [cos_y,  0, sin_y],
        [0,      1,     0],
        [-sin_y, 0, cos_y]
    ], dtype=torch.float32)
    
    # 【平移相机】：把手推离相机 0.5 米远 (如果 Z 太小手会爆出屏幕)
    T = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)

    # 1. 把 3D 关节点严格按照相机参数投影到 2D 像素
    joints_2d, _ = project_points(joints_3d, K, R, T)
    
    # 2. 仅仅使用 2D 投影点计算你的 Dissim (完美符合真实 Tracker 所见)
    pose_dissim_2d, _ = calculate_dissim_2d(joints_2d)
    
    for i, score in enumerate(pose_dissim_2d):
        print(f"Pose {i} 2D Dis_sim vs Pose 0: {score.item():.4f}")
        
    # 3. 画出相机视野里的二维手
    plot_mano_2d_projection(joints_2d, vertices_3d, faces, K, R, T, pose_dissim_2d)



