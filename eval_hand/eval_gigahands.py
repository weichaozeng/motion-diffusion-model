import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils_hand import dist_util

class GigaHandsEvaluator(Dataset):
    def __init__(self, args, model, diffusion, dataloader, num_samples_limit=None, scale=1.):
        self.args = args
        self.dataloader = dataloader
        self.model = model
        
        use_ddim = False 
        self.sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        self.generated_data = []
        self.model.eval()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc="Generating Eval Samples"):
                
                if num_samples_limit is not None and len(self.generated_data) >= num_samples_limit:
                    break

                batch = {k: v.to(dist_util.dev()) if torch.is_tensor(v) else v 
                                for k, v in batch.items()}
                if scale != 1.:
                    batch['scale'] = torch.tensor(scale, dtype=torch.float32).to(dist_util.dev())

                sample = self.sample_fn(
                    self.model,
                    batch['x_pose'],
                    model_kwargs=batch,
                    progress=False
                )

                for bs_i in range(batch['x_pose'].shape[0]):
                    entry = {
                        'pred_pose': sample[bs_i],
                        'gt_pose': batch['x_pose'][bs_i],
                        'suffix_mask': batch['suffix_mask'][bs_i],
                        'cam': batch['cam'][bs_i],
                        'gt_beta': batch['x_beta'][bs_i],
                        'video_path': batch['video_path'][bs_i],
                        'frame_indices': batch['frame_indices'][bs_i],
                        'gt_trans': batch['x_trans'][bs_i],
                        'gt_ff_root_orient_rotmat': batch['x_ff_root_orient_rotmat'][bs_i],
                        'y_pose': batch['y_pose'][bs_i],
                        'y_ff_root_orient_rotmat': batch['y_ff_root_orient_rotmat'][bs_i],
                    }   
                    self.generated_data.append(entry)
        self.model.train()

    def __len__(self):
        return len(self.generated_data)

    def __getitem__(self, item):
        data = self.generated_data[item]
        
        pred = data['pred_pose']
        gt = data['gt_pose']
        
        return pred, gt, data['suffix_mask'], data['gt_beta'], data
    

def compute_batch_metrics(pred_xyz, gt_xyz, suffix_mask):
    """
    pred_xyz, gt_xyz: [bs, njoints, 3, nframes]
    suffix_mask: [bs, nframes]
    """
    # [bs, njoints, 3, nframes] -> [bs, nframes, njoints, 3]
    pred = pred_xyz.permute(0, 3, 1, 2)
    gt = gt_xyz.permute(0, 3, 1, 2)

    p = pred[~suffix_mask] 
    g = gt[~suffix_mask]

    if p.shape[0] == 0:
        return 0.0, 0.0
    
    mu_p = p.mean(dim=1, keepdim=True)
    mu_g = g.mean(dim=1, keepdim=True)
    p_c = p - mu_p
    g_c = g - mu_g

    M = torch.matmul(g_c.transpose(1, 2), p_c)
    U, S, Vh = torch.linalg.svd(M)
    det = torch.linalg.det(torch.matmul(U, Vh))
    U_copy = U.clone()
    U_copy[det < 0, :, 2] *= -1
    R = torch.matmul(U_copy, Vh)

    p_aligned = torch.matmul(p_c, R.transpose(1, 2)) + mu_g

    dists = torch.norm(p_aligned - g, dim=-1).flatten()
    
    pa_mpjpe = dists.mean().item()

    thresholds = torch.linspace(0, 50, 100, device=dists.device)
    pck_curves = (dists.unsqueeze(0) <= thresholds.unsqueeze(1)).float().mean(dim=1)
    auc = torch.trapz(pck_curves, thresholds).item() / 50.0

    return pa_mpjpe, auc