import torch
from model_hand.mdm_hand import MDM_Hand
from diffusion_hand import gaussian_diffusion as gd
from diffusion_hand.respace import SpacedDiffusion, space_timesteps

def create_model_and_diffusion(args):
    model = MDM_Hand(**get_model_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion





def get_model_args(args):

    # SMPL defaults
    if args.dataset == 'gigahands':
        data_rep = 'rot6d'
        njoints = 16 + 1
        nfeats = 6
        all_goal_joint_names = []
    else:
        raise NotImplementedError()
    return {
        'modeltype': '', 
        'njoints': njoints, 
        'nfeats': nfeats, 
        'data_rep': data_rep,

        'translation': False, 
        'pose_rep': 'rot6d', 
        'glob': True, 
        'glob_rot': True,

        'latent_dim': args.latent_dim, 
        
        'ff_size': 1024, 
        'num_layers': args.layers, 
        'num_heads': 4,
        'dropout': 0.1, 

        'activation': "gelu", 

        'cond_mask_prob': args.cond_mask_prob,
        'suffix_mask': True,  # mask_frames,
        'arch': args.arch,

        'dataset': args.dataset,
        'pos_embed_max_len': args.pos_embed_max_len, 

        'all_goal_joint_names': all_goal_joint_names,
    }



def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    if hasattr(args, 'lambda_target_loc'):
        lambda_target_loc = args.lambda_target_loc
    else:
        lambda_target_loc = 0.

    STR_TO_STRATEGY = {
        'standard': gd.DiffusionStrategy.STANDARD,
        'residual': gd.DiffusionStrategy.RESIDUAL
    }
    strategy_str = getattr(args, 'strategy', 'standard')
    strategy_enum = STR_TO_STRATEGY.get(strategy_str.lower(), gd.DiffusionStrategy.STANDARD)

    kappa = getattr(args, 'kappa', 1.0)

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        strategy=strategy_enum,
        kappa=kappa,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_target_loc=lambda_target_loc,
    )


def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or 'sequence_pos_encoder' in k for k in missing_keys])