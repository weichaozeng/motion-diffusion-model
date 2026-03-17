from diffusion_hand.nn import mean_flat, sum_flat
import torch
import numpy as np
import utils_hand.rotation_conversions as geometry
import torch.nn.functional as F

def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l2(a, b):
    return (a - b) ** 2

def masked_l2(a, b, mask, loss_fn=diff_l2, epsilon=1e-8, entries_norm=True):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(1).unsqueeze(1)
    loss = loss_fn(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val


def masked_goal_l2(pred_goal, ref_goal, cond, all_goal_joint_names):
    all_goal_joint_names_w_traj = np.append(all_goal_joint_names, 'traj')
    target_joint_idx = [[np.where(all_goal_joint_names_w_traj == j)[0][0] for j in sample_joints] for sample_joints in cond['target_joint_names']]
    loc_mask = torch.zeros_like(pred_goal[:,:-1], dtype=torch.bool)
    for sample_idx in range(loc_mask.shape[0]):
        loc_mask[sample_idx, target_joint_idx[sample_idx]] = True
    loc_mask[:, -1, 1] = False  # vertical joint of 'traj' is always masked out
    loc_loss = masked_l2(pred_goal[:,:-1], ref_goal[:,:-1], loc_mask, entries_norm=False)
    
    heading_loss = masked_l2(pred_goal[:,-1:, :1], ref_goal[:,-1:, :1], cond['is_heading'].unsqueeze(1).unsqueeze(1), loss_fn=angle_l2, entries_norm=False)

    loss =  loc_loss + heading_loss
    return loss



def masked_geodesic_loss(pred_rot6d, target_rot6d, mask, epsilon=1e-6):
    """
    pred_rot6d, target_rot6d: [B, J, 6, T]
    mask: [B, 1, 1, T]
    """

    pred = pred_rot6d.permute(0, 1, 3, 2).contiguous()     # [B, J, T, 6]
    target = target_rot6d.permute(0, 1, 3, 2).contiguous() # [B, J, T, 6]
    
    # [B, J, T, 3, 3]
    pred_mat = geometry.rotation_6d_to_matrix(pred)       
    target_mat = geometry.rotation_6d_to_matrix(target)   
    
    # R_rel = R_target @ R_pred^T
    pred_mat_inv = pred_mat.transpose(-1, -2)
    r_rel = torch.matmul(target_mat, pred_mat_inv)
    
    # Trace
    tr = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
    
    # Geodesic Distance
    cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0 + epsilon, 1.0 - epsilon)
    theta = torch.acos(cos_theta) # [B, J, T]
    
    loss = theta ** 2 
    mask_squeeze = mask.squeeze(2) # -> [B, 1, T]
    loss = loss * mask_squeeze.float()
    
    # Normalize
    loss_sum = sum_flat(loss)
    n_entries = loss.shape[1]
    non_zero_elements = sum_flat(mask_squeeze.float()) * n_entries
    
    return loss_sum / (non_zero_elements + epsilon)


def masked_smooth_l1(a, b, m):
    loss = F.smooth_l1_loss(a, b, reduction='none', beta=0.1)
    loss = (loss * m).sum()
    non_zero_elements = m.sum() * (a.shape[1] * a.shape[2])
    return loss / (non_zero_elements + 1e-8)