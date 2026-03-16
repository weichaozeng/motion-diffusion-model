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



def generate_mano_poses(rot2xyz_model, num_poses=5, device='cpu'):
    """
    使用 MANO 模型生成从平展到握拳的序列
    """
    B = 1 # Batch size
    F = num_poses # Frames
    
    # 初始化全 0 的 pose (MANO 中通常全 0 代表平展手)
    # 按照 Rotation2xyz 的 permute 逻辑 (0,3,1,2)，输入 shape 应为 [B, njoints, feats, F]
    # MANO 有 1个全局旋转 + 15个手指关节 = 16 joints. 使用 rotvec (axis-angle) 则是 3 feats.
    poses = torch.zeros((B, 16, 3, F), device=device)
    
    # 模拟握拳过程：逐步增加指间关节的弯曲角度
    for i in range(F):
        # 假设弯曲主要沿着局部的某个轴 (例如 x 或 z 轴，视具体 MANO 坐标系而定)
        # 这里假设 index=0 是全局旋转，1-15 是手指。我们逐渐增大它们的弯曲角度 (最大约 1.5 弧度)
        bend_angle = i * (1.5 / (F - 1)) 
        
        # 将相同的弯曲角度应用到所有手指关节 (简化模拟)
        # 注意：实际 MANO 握拳可能需要针对不同关节设置不同的轴，这里假设索引 0 代表主要屈伸轴
        poses[0, 1:, 0, i] = bend_angle 
        
    beta = torch.zeros((B, 10), device=device) # 固定形状
    translation = torch.zeros((B, F, 3), device=device) # 根节点位移为 0
    
    # 调用你的模型进行前向推断
    with torch.no_grad():
        joints, vertices = rot2xyz_model(
            pose=poses, 
            pose_rep="rotvec", 
            beta=beta, 
            translation=translation, 
            return_vertices=True
        )
        
    # joints shape 经过代码处理后是 [B, 21, 3, F]，我们需要转成 [F, 21, 3] 方便计算
    joints = joints.squeeze(0).permute(2, 0, 1).cpu() 
    # vertices shape 是 [B, F, 778, 3]，我们需要转成 [F, 778, 3]
    vertices = vertices.squeeze(0).cpu()
    
    return joints, vertices

# ==========================================
# 3. 计算 Dissimilarity (与初始手做对比)
# ==========================================
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

# ==========================================
# 4. 高级 3D 可视化 (Vertices + 骨架)
# ==========================================
def plot_mano_poses(joints_tensor, vertices_tensor, dissim_scores, save_path="mano_dissim.png"):
    F_frames = joints_tensor.shape[0]
    fig = plt.figure(figsize=(5, 4 * F_frames))
    
    for i in range(F_frames):
        ax = fig.add_subplot(F_frames, 1, i + 1, projection='3d')
        pose = joints_tensor[i].numpy()
        verts = vertices_tensor[i].numpy()
        
        # 1. 渲染 MANO 真实的 3D Vertices (使用散点图模拟曲面)
        # 如果你的 hand_model 有 faces 属性，用 ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces) 会更平滑
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], 
                   s=0.5, c='lightblue', alpha=0.3, zorder=1) # 半透明网格
        
        # 2. 叠加骨架连接线 (加粗，使其在网格中清晰可见)
        for bone_idx, (p_idx, c_idx) in enumerate(BONE_CONNECTIONS.numpy()):
            ax.plot([pose[p_idx, 0], pose[c_idx, 0]],
                    [pose[p_idx, 1], pose[c_idx, 1]],
                    [pose[p_idx, 2], pose[c_idx, 2]],
                    c='red', linewidth=2.5, zorder=5)
            
        # 3. 绘制关节球
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='darkred', s=20, zorder=10)
        
        # 统一视角和坐标系
        # 注意：MANO 坐标系单位通常是米，坐标范围很小，这里设置一个合理的固定范围 (比如 0.2米)
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([-0.1, 0.1])
        ax.view_init(elev=-90, azim=-90) # 视具体 MANO 坐标系调整朝向，通常需要俯视或正视
        
        ax.set_axis_off() # 关闭坐标轴，让画面极其干净
        
        title_color = 'green' if i == 0 else 'red'
        ax.set_title(f"Pose {i} | Dis_sim vs Initial: {dissim_scores[i]:.4f}", color=title_color, fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 5. 主执行入口
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    rot2xyz_model = Rotation2xyz(device=device)
    
   
    joints_kps, vertices_kps = generate_mano_poses(rot2xyz_model, num_poses=5, device=device)

    pose_dissim, _ = calculate_dissim(joints_kps)
    

    for i, score in enumerate(pose_dissim):
        print(f"Pose {i} vs Pose 0: {score.item():.4f}")
    plot_mano_poses(joints_kps, vertices_kps, pose_dissim)