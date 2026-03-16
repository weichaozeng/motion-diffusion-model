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

def define_manual_poses():
    """
    在这里手动定义每一帧、每个关节的旋转角度（弧度制）。
    
    【MANO 关节索引参考】
    0: 全局手腕旋转 (控制整只手的朝向)
    1, 2, 3: 食指 (根部 MCP, 中部 PIP, 指尖 DIP)
    4, 5, 6: 中指 (根部 MCP, 中部 PIP, 指尖 DIP)
    7, 8, 9: 小指 (根部 MCP, 中部 PIP, 指尖 DIP)
    10, 11, 12: 无名指 (根部 MCP, 中部 PIP, 指尖 DIP)
    13, 14, 15: 拇指 (根部 CMC, 中部 MCP, 指尖 IP)
    
    【旋转轴参考】
    0: X 轴 (通常是扭转 / 麻花)
    1: Y 轴 (通常是四指的屈伸 Flexion/Extension)
    2: Z 轴 (通常是手指的侧偏外展，或者拇指的屈伸)
    """
    F = 5 # 总帧数
    poses = torch.zeros((F, 16, 3)) # 初始化全 0 (平展手)
    
    # --------------------------------------------------
    # Pose 0: 平展状态 (保持默认全 0 即可)
    # --------------------------------------------------
    poses[0, 1:13, 2] = -0.8   # 四指 Y 轴整体下弯
    poses[0, 13:16, 2] = 0.0   # 拇指 Z 轴内收
    # --------------------------------------------------
    # Pose 1: 微屈 (整体一起调整)
    # --------------------------------------------------
    poses[1, 1:13, 1] = -0.4   # 四指 Y 轴整体下弯
    poses[1, 13:16, 2] = 0.4   # 拇指 Z 轴内收
    
    # --------------------------------------------------
    # Pose 2: 不对称姿态 (比如食指单独伸直，其他弯曲)
    # --------------------------------------------------
    poses[2, 1:4, 1] = 0.0     # 食指 (1,2,3) 保持伸直(0弧度)
    poses[2, 4:13, 1] = -0.8   # 其他三指弯曲
    poses[2, 13:16, 2] = 0.6   # 拇指弯曲
    
    # --------------------------------------------------
    # Pose 3: 进一步测试特定关节 (例如只弯曲指尖)
    # --------------------------------------------------
    # 让中指(4,5,6) 只有指尖(6)弯曲，根部不弯
    poses[3, 4:6, 1] = 0.0     
    poses[3, 6, 1] = -1.2      
    poses[3, 1:4, 1] = -1.0    # 食指弯曲
    poses[3, 7:13, 1] = -1.0   # 其他指弯曲
    poses[3, 13:16, 2] = 0.9   # 拇指
    
    # --------------------------------------------------
    # Pose 4: 极度握拳 + 侧偏测试
    # --------------------------------------------------
    poses[4, 1:13, 1] = -1.5   # 四指极度弯曲 (接近90度)
    poses[4, 13:16, 2] = 1.2   # 拇指极度内收
    
    # 添加一点外展(Abduction)：让小指(7,8,9)和食指(1,2,3)往外撇一点 (调整 Z 轴)
    poses[4, 7, 2] = -0.3      # 小指根部向外
    poses[4, 1, 2] = 0.3       # 食指根部向外
    
    return poses


def generate_mano_poses(rot2xyz_model, custom_poses, device='cpu'):
    """
    使用 MANO 模型生成自定义的手部序列。
    :param custom_poses: 形状为 (F, 16, 3) 的 Tensor，F 为帧数
    """
    B = 1 
    F = custom_poses.shape[0]
    
    # 你的模型期望的输入是 (B, 16, 3, F)
    # 我们将 (F, 16, 3) 增加 batch 维度并调换轴向
    poses = custom_poses.unsqueeze(0).permute(0, 2, 3, 1).to(device)
    
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
    # plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    rot2xyz_model = Rotation2xyz(device=device)
    mano_pkl_path = '/home/zvc/Project/VHand/_DATA/data/mano/MANO_RIGHT.pkl'
    faces = get_mano_faces(mano_pkl_path)
   
    custom_poses = define_manual_poses()
    joints_kps, vertices_kps = generate_mano_poses(rot2xyz_model, custom_poses, device=device)

    pose_dissim, _ = calculate_dissim(joints_kps)
    

    for i, score in enumerate(pose_dissim):
        print(f"Pose {i} vs Pose 0: {score.item():.4f}")
    plot_mano_poses(joints_kps, vertices_kps, faces, pose_dissim)