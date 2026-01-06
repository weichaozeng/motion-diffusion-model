from argparse import ArgumentParser
import argparse
import os
import json



def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    # return apply_rules(parser.parse_args())
    return parser.parse_args()


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=42, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str, help="Choose platform to log results. NoPlatform means no logging.")
    parser.add_argument("--platform_name", required=True, type=str, help="Name of the experiment for WandB and saving.")
    group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='gigahands', choices=['gigahands'], type=str, help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="/home/zvc/Project/motion-diffusion-model/dataset/gigahands", type=str, help="If empty, will use defaults according to the specified dataset.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--layers", default=6, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    # group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_target_loc", default=0.0, type=float, help="For HumanML only, when . L2 with target location.")

    group.add_argument("--pos_embed_max_len", default=5000, type=int,
                       help="Pose embedding max length.")
    group.add_argument("--use_ema", action='store_true',
                    help="If True, will use EMA model averaging.")
    

def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str, help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int, help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument(
        "--strategy", 
        default='standard', 
        choices=['standard', 'residual'], 
        type=str, 
        help="Diffusion strategy: 'standard' for vanilla DDPM, 'residual' for Refinement."
    )
    group.add_argument(
        "--kappa", 
        default=0.1, 
        type=float, 
        help="The weight of noise/residual in RESIDUAL strategy. Lower values mean higher trust in y_pose."
    )


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='val', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", default=True,
                       help="If True, will run evaluation during training.")
   
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=30_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=120, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--gen_guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta (for EMA).")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    group.add_argument("--uncond_prob", default=0.1, type=float, help="The probability of dropping the condition during training.")