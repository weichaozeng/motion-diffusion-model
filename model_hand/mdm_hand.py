import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_hand.rotation2xyz import Rotation2xyz
from utils_hand.misc import WeightedSum


class MDM_Hand(nn.Module):
    def __init__(self, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", data_rep='rot6d', arch='trans_enc', **kargs):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.input_feats = self.njoints * self.nfeats

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.suffix_mask = kargs.get('suffix_mask', False)
        self.arch = arch
        self.input_process = InputProcess(self.data_rep, self.input_feats * 2 + 1, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu')


        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')


    def parameters(self):
        return [p for name, p in self.named_parameters()]


    def forward(self, x, timesteps, batch):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        device = x.device

        # x: [bs, njoints, nfeats, nframes] -> [nframes, bs, njoints*nfeats]
        x = x.permute(3, 0, 1, 2).contiguous().view(nframes, bs, njoints * nfeats) 
       # y: [bs, njoints, nfeats, nframes] -> [nframes, bs, njoints*nfeats]
        y_ret = batch['y_ret']                
        y_ret = y_ret.permute(3, 0, 1, 2).contiguous().view(nframes, bs, njoints * nfeats)
        # inpaint_mask: [bs, nframes] -> [nframes, bs, 1]
        inpaint_mask = batch['inpaint_mask']      
        inpaint_mask = inpaint_mask.transpose(1, 0).unsqueeze(-1)

        # for CFG
        uncond = batch.get('uncond', False)
        if isinstance(uncond, bool):
            if uncond:
                y_ret = torch.zeros_like(y_ret)
                inpaint_mask = torch.zeros_like(inpaint_mask)
        else:
            # if uncond is a tensor of shape [bs]
            keep_mask = (~uncond).float().to(x.device).view(1, bs, 1)
            y_ret = y_ret * keep_mask
            inpaint_mask = inpaint_mask * keep_mask
        
        # concat: [nframes, bs, njoints*nfeats*2 + 1]
        x_full = torch.cat([x, y_ret, inpaint_mask], dim=2)
        
        # x_full: [nframes, bs, latent_dim]; time_emb: [1, bs, latent_dim]
        x_full = self.input_process(x_full)
        time_emb = self.embed_timestep(timesteps)

        if self.suffix_mask:
            # True will be masked
            step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
            frames_mask = torch.cat([step_mask, batch['suffix_mask']], dim=1)
        else:
            raise NotImplementedError("Suffix mask must be True for hand model")

        assert self.arch == 'trans_enc'
        # adding the timestep embed; [nframes+1, bs, d]
        xseq = torch.cat((time_emb, x_full), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        # output: [nframes+1, bs, d]
        output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]
        
        # output: [nframes+1, bs, d] -> [bs, njoints, nfeats, nframes]
        output = self.output_process(output)
        return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.hand_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.hand_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x_full):
        # x_full: [seqlen, bs, input_feats]; input_feats = njoints*nfeats*2 + 1
        if self.data_rep in ['rot6d', 'xyz']:
            x = self.poseEmbedding(x_full)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x_full[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [nframes, bs, njoints*nfeats]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, njoints*nfeats]
            vel = output[1:]  # [nframes-1, bs, d]
            vel = self.velFinal(vel)  # [nframes-1, bs, njoints*nfeats]
            output = torch.cat((first_pose, vel), axis=0)  # [nframes, bs, njoints*nfeats]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output
