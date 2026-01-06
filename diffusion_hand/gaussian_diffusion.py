import math
import numpy as np
import enum
import torch
import torch as th
from copy import deepcopy

from diffusion_hand.nn import mean_flat, sum_flat
from diffusion_hand.losses import normal_kl, discretized_gaussian_log_likelihood
from utils_hand.loss_util import masked_l2

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL
    
class DiffusionStrategy(enum.Enum):
    STANDARD = enum.auto()          
    RESIDUAL = enum.auto()       


class GaussianDiffusion:
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        strategy=DiffusionStrategy.STANDARD,
        kappa=1.0,
        rescale_timesteps=False,
        # lambda for loss
        lambda_rcxyz=0.,
        lambda_vel=0.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_target_loc=0.,
        **kargs,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_target_loc = lambda_target_loc
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_target_loc > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # strategy
        self.strategy = strategy
        self.kappa = kappa

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        

        if strategy == DiffusionStrategy.RESIDUAL:
            self.eta = 1.0 - self.alphas_cumprod
            self.eta_prev = np.append(0.0, self.eta[:-1])
            self.eta_next = np.append(self.eta[1:], 1.0)

        elif strategy == DiffusionStrategy.STANDARD:
            self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
            self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
            assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
            self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

            # calculations for posterior q(x_{t-1} | x_t, x_0)
            self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            self.posterior_log_variance_clipped = np.log(
                np.append(self.posterior_variance[1], self.posterior_variance[1:])
            )
            self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
            )

        # self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.
        self.masked_l2 = masked_l2

    def q_mean_variance(self, x_start, t, model_kwargs=None):
        """
        Get the distribution q(x_t | x_0) or q(x_t | x_0, y).
        """
        if self.strategy == DiffusionStrategy.STANDARD:
            alphas_cumprod = _extract_into_tensor(self.alphas_cumprod, t, x_start.shape)
            sqrt_alpha_bar = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            mean = sqrt_alpha_bar * x_start
            variance = 1.0 - alphas_cumprod
            log_variance = th.log(variance)
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            y = model_kwargs['y_pose']
            eta = _extract_into_tensor(self.eta, t, x_start.shape)
            mean = x_start + eta * (y - x_start)
            variance = (self.kappa ** 2) * eta
            log_variance = th.log(variance.clamp(min=1e-20))
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")

        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise=None, model_kwargs=None):
        """
        Diffuse the dataset for a given number of diffusion steps. In other words, sample from q(x_t | x_0) or q(x_t | x_0, y).
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        if self.strategy == DiffusionStrategy.STANDARD:
            sqrt_alpha_bar = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            return sqrt_alpha_bar * x_start + sqrt_one_minus_alphas_cumprod * noise

        elif self.strategy == DiffusionStrategy.RESIDUAL:
            y = model_kwargs['y_pose']
            eta = _extract_into_tensor(self.eta, t, x_start.shape)
            return x_start + eta * (y - x_start) + self.kappa * th.sqrt(eta) * noise
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
        
    def q_posterior_mean_variance(self, x_start, x_t, t, model_kwargs=None):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0) or q(x_{t-1} | x_t, x_0, y).
        """
        assert x_start.shape == x_t.shape

        if self.strategy == DiffusionStrategy.STANDARD:
            posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
            posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x_t.shape
            )
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            y = model_kwargs['y_pose']
            eta_prev = _extract_into_tensor(self.eta_prev, t, x_start.shape)
            posterior_mean = x_start + eta_prev * (y - x_start)
            posterior_variance = _extract_into_tensor(self.betas, t, x_t.shape)
            posterior_log_variance_clipped = th.log(posterior_variance.clamp(min=1e-20))
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t).
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), model_kwargs)


        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)

        else:
            if self.strategy == DiffusionStrategy.STANDARD:
                model_variance, model_log_variance = {
                    ModelVarType.FIXED_LARGE: (np.append(self.posterior_variance[1], self.betas[1:]), np.log(np.append(self.posterior_variance[1], self.betas[1:]))),
                    ModelVarType.FIXED_SMALL: (self.posterior_variance, self.posterior_log_variance_clipped),
                }[self.model_var_type]
            else:
                model_variance = self.betas
                model_log_variance = np.log(np.append(self.betas[1], self.betas[1:]))

        def process_xstart(x_in):
            if denoised_fn is not None:
                x_in = denoised_fn(x_in)
            if clip_denoised:
                return x_in.clamp(-1, 1)
            return x_in
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output, model_kwargs=model_kwargs)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t, model_kwargs=model_kwargs
            )
        else:
            raise NotImplementedError(self.model_mean_type)
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _predict_xstart_from_eps(self, x_t, t, eps, model_kwargs=None):
        assert x_t.shape == eps.shape
        if self.strategy == DiffusionStrategy.STANDARD:
            return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
            )
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            y = model_kwargs['y_pose']
            eta = _extract_into_tensor(self.eta, t, x_t.shape)
            return (x_t - eta * y - self.kappa * th.sqrt(eta) * eps) / (1 - eta).clamp(min=1e-5)
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
        
    def _predict_xstart_from_xprev(self, x_t, t, xprev, model_kwargs=None):
        assert x_t.shape == xprev.shape
        if self.strategy == DiffusionStrategy.STANDARD:
            return (
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
                    self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
                ) * x_t
            )
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            y = model_kwargs['y_pose']
            eta_prev = _extract_into_tensor(self.eta_prev, t, x_t.shape)
            return (xprev - eta_prev * y) / (1 - eta_prev).clamp(min=1e-5)
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
        
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart, model_kwargs=None):
        if self.strategy == DiffusionStrategy.STANDARD:
            return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
            ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            y = model_kwargs['y_pose']
            eta = _extract_into_tensor(self.eta, t, x_t.shape)
            denom = (self.kappa * th.sqrt(eta)).clamp(min=1e-5)
            return (x_t - pred_xstart - eta * (y - pred_xstart)) / denom
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
        
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        if self.strategy == DiffusionStrategy.STANDARD:
            noise_coef = (1 - _extract_into_tensor(self.alphas_cumprod, t, x.shape)).sqrt()
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            eta = _extract_into_tensor(self.eta, t, x.shape)
            noise_coef = self.kappa * th.sqrt(eta)
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
        
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"], model_kwargs=model_kwargs)
        eps = eps - noise_coef * cond_fn(x, self._scale_timesteps(t), model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps, model_kwargs=model_kwargs)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t, model_kwargs=model_kwargs)

        return out
    
    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        if self.strategy == DiffusionStrategy.STANDARD:
            noise_coef = (1 - _extract_into_tensor(self.alphas_cumprod, t, x.shape)).sqrt()
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            eta = _extract_into_tensor(self.eta, t, x.shape)
            noise_coef = self.kappa * th.sqrt(eta)
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"], model_kwargs=model_kwargs)
        eps = eps - noise_coef * cond_fn(x, t, p_mean_var, model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps, model_kwargs=model_kwargs)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t, model_kwargs=model_kwargs)
        return out
    
    def p_sample(self, model, x,t , clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, const_noise=False):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)

        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        if cond_fn is not None:
            if self.strategy == DiffusionStrategy.RESIDUAL:
                out = self.condition_mean(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
            elif self.strategy == DiffusionStrategy.STANDARD:
                out["mean"] = self.condition_mean(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
            else:
                raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        
        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"],
        }
    
    def p_sample_with_grad(self, model, x,t , clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, const_noise=False):
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
            
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            ) 
            
            if cond_fn is not None:
                if self.strategy == DiffusionStrategy.RESIDUAL:
                    out = self.condition_mean_with_grad(
                        cond_fn, out, x, t, model_kwargs=model_kwargs
                    )
                elif self.strategy == DiffusionStrategy.STANDARD:
                    out["mean"] = self.condition_mean_with_grad(
                        cond_fn, out, x, t, model_kwargs=model_kwargs
                    )
                else:
                    raise NotImplementedError(f"Strategy {self.strategy} not implemented.")
                    
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}
    
    def p_sample_loop(self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
        )):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            final = sample
        
        assert final is not None, "Sampling loop failed to produce any samples."
        
        if dump_steps is not None:
            return dump
            
        return final["sample"]
    
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            if self.strategy == DiffusionStrategy.STANDARD:
                # STANDARD：pure noise
                img = th.randn(*shape, device=device)
            elif self.strategy == DiffusionStrategy.RESIDUAL:
                # RESIDUAL： y_pose + kappa * noise 
                y = model_kwargs['y_pose']
                img = y + self.kappa * th.randn_like(y)
        
        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img, model_kwargs=model_kwargs)

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample

            out = sample_fn(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                const_noise=const_noise,
            )
            yield out
            img = out["sample"]

    
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"], model_kwargs=model_kwargs)

        if self.strategy == DiffusionStrategy.STANDARD:
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
            sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            y = model_kwargs['y_pose']
            eta_n = _extract_into_tensor(self.eta, t, x.shape)
            eta_prev = _extract_into_tensor(self.eta_prev, t, x.shape)
            sigma = (
                eta
                * self.kappa
                * th.sqrt(eta_prev * (1 - eta_prev / eta_n.clamp(min=1e-5)))
            )
            mean_pred = (
                out["pred_xstart"] 
                + eta_prev * (y - out["pred_xstart"]) 
                + self.kappa * th.sqrt(eta_prev) * eps
            )
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {
            "sample": sample, 
            "pred_xstart": out_orig["pred_xstart"],
        }


    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(
                    cond_fn, out_orig, x, t, model_kwargs=model_kwargs
                )
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"], model_kwargs=model_kwargs)

        if self.strategy == DiffusionStrategy.STANDARD:
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
            
            sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            
            mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
        elif self.strategy == DiffusionStrategy.RESIDUAL:
            y = model_kwargs['y_pose']
            eta_n = _extract_into_tensor(self.eta, t, x.shape)
            eta_prev = _extract_into_tensor(self.eta_prev, t, x.shape)
            
            sigma = (
                eta
                * self.kappa
                * th.sqrt(eta_prev * (1 - eta_prev / eta_n.clamp(min=1e-5)))
            )
            mean_pred = (
                out["pred_xstart"] 
                + eta_prev * (y - out["pred_xstart"]) 
                + self.kappa * th.sqrt(eta_prev) * eps
            )
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        
        return {
            "sample": sample, 
            "pred_xstart": out_orig["pred_xstart"].detach()
        }
    
    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"], model_kwargs=model_kwargs)

        if self.strategy == DiffusionStrategy.STANDARD:
            alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
            
            mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
            )
        else:
            y = model_kwargs['y_pose']
            eta_next = _extract_into_tensor(self.eta_next, t, x.shape)
            # x_{t+1} = x0 + eta_next * (y - x0) + kappa * sqrt(eta_next) * eps
            mean_pred = (
                out["pred_xstart"] 
                + eta_next * (y - out["pred_xstart"])
                + self.kappa * th.sqrt(eta_next) * eps
            )

        return {
            "sample": mean_pred, 
            "pred_xstart": out["pred_xstart"]
        }
    
    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]
    
    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            if self.strategy == DiffusionStrategy.STANDARD:
                img = th.randn(*shape, device=device)
            else:
                # y_pose + kappa * noise
                y = model_kwargs['y_pose']
                img = y + self.kappa * th.randn_like(y)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img, model_kwargs=model_kwargs)

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
            
            out = sample_fn(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            yield out
            img = out["sample"]

    # TODO: plms_sample

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t, model_kwargs=model_kwargs
        )

        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)


        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):

        if model_kwargs is None:
            model_kwargs = {}

        enc = model.model
        mask = ~model_kwargs['suffix_mask']
        mano_beta = model_kwargs['x_beta']

        get_xyz = lambda sample: enc.rot2xyz(pose=sample, pose_rep=model.data_rep, beta=mano_beta)

        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise, model_kwargs=model_kwargs)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            print(f'x_t:{x_.shape}')
            print(f't: {self._scale_timesteps(t).shape}')
            print(f'batch: {model_kwargs.keys()}')
            model_output = model(x_t, self._scale_timesteps(t), model_kwargs)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t, model_kwargs=model_kwargs
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape

            terms["rot_mse"] = self.masked_l2(target, model_output, mask)

            target_xyz, model_output_xyz = None, None

            if self.lambda_rcxyz  > 0.:
                target_xyz = get_xyz(target)
                model_output_xyz = get_xyz(model_output)
                terms["rcxyz_mse"] = self.masked_l2(target_xyz, model_output_xyz, mask)
            
            if self.lambda_vel_rcxyz > 0.:
                
                target_xyz = get_xyz(target) if target_xyz is None else target_xyz
                model_output_xyz = get_xyz(model_output) if model_output_xyz is None else model_output_xyz
                target_xyz_vel = (target_xyz[:, :, :, 1:] - target_xyz[:, :, :, :-1])
                model_output_xyz_vel = (model_output_xyz[:, :, :, 1:] - model_output_xyz[:, :, :, :-1])
                terms["vel_xyz_mse"] = self.masked_l2(target_xyz_vel, model_output_xyz_vel, mask[:, :, :, 1:])
            
            if self.lambda_vel > 0.:
                target_vel = (target[..., 1:] - target[..., :-1])
                model_output_vel = (model_output[..., 1:] - model_output[..., :-1])
                terms["vel_mse"] = self.masked_l2(target_vel[:, :-1, :, :], model_output_vel[:, :-1, :, :], mask[:, :, :, 1:])

            terms["loss"] = terms["rot_mse"] + terms.get('vb', 0.) +\
                            (self.lambda_vel * terms.get('vel_mse', 0.)) +\
                            (self.lambda_rcxyz * terms.get('rcxyz_mse', 0.)) + \
                            (self.lambda_vel_rcxyz * terms.get('vel_xyz_mse', 0.))
        else:
            raise NotImplementedError(self.loss_type)

        return terms