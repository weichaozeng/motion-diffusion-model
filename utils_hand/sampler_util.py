import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy



class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep


    def forward(self, x, timesteps, batch=None):
        bs = x.shape[0]

        # scale
        scale = batch.get('scale', 1.0)
        if isinstance(scale, torch.Tensor):
            scale = scale.view(-1, 1, 1, 1)

        # input of both cond and uncond
        x_combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([timesteps, timesteps], dim=0)

        # fist half is cond, second half is uncond    
        combined_batch = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                combined_batch[k] = torch.cat([v, v], dim=0)
            else:
                combined_batch[k] = v + v

        uncond_mask = torch.zeros(2 * bs, dtype=torch.bool, device=x.device)
        uncond_mask[bs:] = True
        combined_batch['uncond'] = uncond_mask

        output_combined = self.model(x_combined, t_combined, combined_batch)
        out_cond = output_combined[:bs]
        out_uncond = output_combined[bs:]
        
        return out_uncond + scale * (out_cond - out_uncond)

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)


def wrapped_getattr(self, name, default=None, wrapped_member_name='model'):
    ''' should be called from wrappers of model classes such as ClassifierFreeSampleModel'''

    if isinstance(self, torch.nn.Module):
        # for descendants of nn.Module, name may be in self.__dict__[_parameters/_buffers/_modules] 
        # so we activate nn.Module.__getattr__ first.
        # Otherwise, we might encounter an infinite loop
        try:
            attr = torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            wrapped_member = torch.nn.Module.__getattr__(self, wrapped_member_name)
            attr = getattr(wrapped_member, name, default)
    else:
        # the easy case, where self is not derived from nn.Module
        wrapped_member = getattr(self, wrapped_member_name)
        attr = getattr(wrapped_member, name, default)
    return attr        


