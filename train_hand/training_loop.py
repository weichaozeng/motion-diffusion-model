import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import re
from os.path import join as pjoin
from typing import Optional

import blobfile as bf
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from data_loaders_hand.get_data import get_dataset_loader

from diffusion_hand import logger
from diffusion_hand.fp16_util import MixedPrecisionTrainer
from diffusion_hand.resample import LossAwareSampler, UniformSampler
from diffusion_hand.resample import create_named_schedule_sampler


from utils_hand.model_util import load_model
from utils_hand.sampler_util import ClassifierFreeSampleModel
from utils_hand import dist_util

from eval_hand import eval_gigahands

from visualize_hand import vis_gigahands

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.model_avg = None
        if self.args.use_ema:
            self.model_avg = copy.deepcopy(self.model)
        self.model_for_eval = self.model_avg if self.args.use_ema else self.model
        if args.gen_guidance_param != 1:
            self.model_for_eval = ClassifierFreeSampleModel(self.model_for_eval)   # wrapping model with the classifier-free sampler
        self.diffusion = diffusion
        self.cond_mask_prob = args.cond_mask_prob
        
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        if self.args.use_ema:
            self.opt = AdamW(
                # with amp, we don't need to use the mp_trainer's master_params
                (self.model.parameters()
                 if self.use_fp16 else self.mp_trainer.master_params),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
            )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.use_ddp = False
        self.ddp_model = self.model

        if args.eval_during_training:
            print(f"Loading validation dataset for {args.dataset}...")
            self.val_data = get_dataset_loader(
                name=args.dataset, 
                batch_size=args.eval_batch_size, 
                num_frames=args.num_frames,
                split=args.eval_split, # 'val'
                device=self.device,
                translation=True
            )
        


    def _load_and_sync_parameters(self):
        resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            # we add 1 because self.resume_step has already been done and we don't want to run it again
            # in particular we don't want to run the evaluation and generation again
            self.step += 1  
            
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint) 
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model(self.model_avg, state_dict_avg)
            else:
                load_model(self.model, state_dict)
                if self.args.use_ema:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=False)


    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd
            self.opt.param_groups[0]['capturable'] = True

    def cond_modifiers(self, cond_mask_prob, batch):
        # All modifiers must be in-place
        batch_size = batch['y_pose'].shape[0]
        device = batch['y_pose'].device

        mask = torch.rand(batch_size, device=device) < cond_mask_prob
        if mask.any():
            batch['y_pose'][mask] = 0.
            batch['inpaint_mask'][mask] = 0.


    def run_loop(self):
        print('train steps:', self.num_steps)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for batch in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break
                # move to device
                batch = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in batch.items()}
                # cfg 
                # Modify in-place for efficiency
                self.cond_modifiers(self.cond_mask_prob, batch) 

                # self.run_step(batch['x_pose'], batch)
                self.run_step(batch['x_ret'], batch)
                if self.total_step() % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')

                if self.total_step() % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    if self.args.use_ema:
                        self.model_avg.eval()
                    self.evaluate()
                    self.model.train()
                    if self.args.use_ema:
                        self.model_avg.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.total_step() - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def forward_backward(self, x_ret, batch):
        self.mp_trainer.zero_grad()
        for i in range(0, x_ret.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = x_ret
            micro_cond = batch
            last_batch = (i + self.microbatch) >= x_ret.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"


    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        assert self.dataset == 'gigahands'

        eval_dataset = eval_gigahands.GigaHandsEvaluator(self.args, self.model_for_eval, self.diffusion, self.val_data, scale=self.args.gen_guidance_param)

        eval_loader = DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False)

        total_pa_mpjpe = []
        total_auc = []

        vis_sample = None 
        vis_batch_idx = np.random.randint(len(eval_loader))
        
        self.model.eval()
        with torch.no_grad():
            for i, (pred_pose, gt_pose, suffix_mask, gt_beta, batch_data) in tqdm(enumerate(eval_loader), desc="Calculating Metrics"):

                pred_xyz = self.model.rot2xyz(pose=pred_pose, pose_rep='rot6d', beta=gt_beta)
                gt_xyz = self.model.rot2xyz(pose=gt_pose, pose_rep='rot6d', beta=gt_beta)

                pa_mpjpe, auc = eval_gigahands.compute_batch_metrics(pred_xyz * 1000, gt_xyz * 1000, suffix_mask)
                
                total_pa_mpjpe.append(pa_mpjpe)
                total_auc.append(auc)

                if i == vis_batch_idx:
                    vis_sample = batch_data

        avg_pa_mpjpe = np.mean(total_pa_mpjpe)
        avg_auc = np.mean(total_auc)
        eval_time = time.time() - start_eval

        print(f"Eval Results: PA-MPJPE: {avg_pa_mpjpe:.2f}mm, AUC: {avg_auc:.4f} (Time: {(eval_time)/60:.2f}min)")

        if self.train_platform:
            self.train_platform.report_scalar(name='PA-MPJPE', value=avg_pa_mpjpe, iteration=self.step, group_name='Eval')
            self.train_platform.report_scalar(name='AUC', value=avg_auc, iteration=self.step, group_name='Eval')
            if vis_sample is not None:
                vis_out_dir = os.path.join(self.save_dir, 'eval_vis', f'step_{self.total_step()}')
                os.makedirs(vis_out_dir, exist_ok=True)

                rgb_video_paths = vis_sample['video_path']
                rgb_frame_indices = vis_sample['frame_indices']
                cams = vis_sample['cam']
                suffix_masks = vis_sample['suffix_mask']

                y_xyz, y_verts = self.model.rot2xyz(pose=vis_sample['y_pose'], pose_rep='rot6d', beta=vis_sample['gt_beta'], ff_rotmat=vis_sample['y_ff_root_orient_rotmat'], translation=vis_sample['gt_trans'], return_vertices=True)
                y_video_dir = os.path.join(vis_out_dir, 'ori_video')
                vis_gigahands.render_video(y_verts, y_video_dir, rgb_video_paths, rgb_frame_indices, cams, suffix_masks)

                pred_xyz, pred_verts = self.model.rot2xyz(pose=vis_sample['pred_pose'], pose_rep='rot6d', beta=vis_sample['gt_beta'], ff_rotmat=vis_sample['y_ff_root_orient_rotmat'], translation=vis_sample['pred_trans'], return_vertices=True)
                pred_video_dir = os.path.join(vis_out_dir, 'pred_video')
                vis_gigahands.render_video(pred_verts, pred_video_dir, rgb_video_paths, rgb_frame_indices, cams, suffix_masks)

                gt_xyz, gt_verts = self.model.rot2xyz(pose=vis_sample['gt_pose'], pose_rep='rot6d', beta=vis_sample['gt_beta'], ff_rotmat=vis_sample['y_ff_root_orient_rotmat'], translation=vis_sample['gt_trans'], return_vertices=True)
                gt_video_dir = os.path.join(vis_out_dir, 'gt_video')
                vis_gigahands.render_video(gt_verts, gt_video_dir, rgb_video_paths, rgb_frame_indices, cams, suffix_masks)

                self.train_platform.report_media(
                    title='Eval_Visualization', 
                    series='y', 
                    iteration=self.step, 
                    local_path=y_video_dir
                )

                self.train_platform.report_media(
                    title='Eval_Visualization', 
                    series='pred', 
                    iteration=self.step, 
                    local_path=pred_video_dir
                )

                self.train_platform.report_media(
                    title='Eval_Visualization', 
                    series='gt', 
                    iteration=self.step, 
                    local_path=gt_video_dir
                )


        self.model.train()


    
    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None
    
    def total_step(self):
        return self.step + self.resume_step
    
    def save(self):
        def save_checkpoint():
            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)

            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
            "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            if self.use_fp16:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt': opt_state,
                    'scaler': self.scaler.state_dict(),
                }

            torch.save(opt_state, f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
