import torch
import torch.nn as nn
import numpy as np
import pickle
from .lbs import batch_rodrigues
from .lbs import lbs, dqs

def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)



NUM_POSES = {'mano': 9}
NUM_SHAPES = 10
NUM_EXPR = 10
MODEL_PATH = '/home/zvc/Project/motion-diffusion-model/body_models/smplh/MANO_RIGHT.pkl'
REGRESSOR_PATH = '/home/zvc/Project/motion-diffusion-model/body_models/J_regressor_mano_RIGHT.txt'


class MANO(nn.Module):
    def __init__(self, device=None, use_pose_blending=True, use_shape_blending=True, use_joints=True, with_color=False, use_lbs=True, **kwargs) -> None:
        super().__init__()


        dtype = torch.float32
        self.dtype = dtype
        self.use_pose_blending = use_pose_blending
        self.use_shape_blending = use_shape_blending
        self.use_joints = use_joints
        
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        # create the SMPL model
        if use_lbs:
            self.lbs = lbs
        else:
            self.lbs = dqs
         
        with open(MODEL_PATH, 'rb') as smpl_file:
            data = pickle.load(smpl_file, encoding='latin1')

        if with_color:
            self.color = data['vertex_colors']
        else:
            self.color = None
        self.faces = data['f']
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))
        for key in ['J_regressor', 'v_template', 'weights']:
            val = to_tensor(to_np(data[key]), dtype=dtype)
            self.register_buffer(key, val)
        # add poseblending
        if use_pose_blending:
            # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
            num_pose_basis = data['posedirs'].shape[-1]
            # 207 x 20670
            posedirs = data['posedirs']
            data['posedirs'] = np.reshape(data['posedirs'], [-1, num_pose_basis]).T
            val = to_tensor(to_np(data['posedirs']), dtype=dtype)
            self.register_buffer('posedirs', val)
        else:
            self.posedirs = None
        # add shape blending
        if use_shape_blending:
            val = to_tensor(to_np(data['shapedirs']), dtype=dtype)
            self.register_buffer('shapedirs', val)
        else:
            self.shapedirs = None
        if use_shape_blending:
            self.J_shaped = None
        else:
            val = to_tensor(to_np(data['J']), dtype=dtype)
            self.register_buffer('J_shaped', val)

        self.nVertices = self.v_template.shape[0]
        # indices of parents for each joints
        parents = to_tensor(to_np(data['kintree_table'][0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        

        # joints regressor
        if use_joints:
            regressor_data = np.loadtxt(REGRESSOR_PATH)
            with open(REGRESSOR_PATH, 'r') as f:
                shape = f.readline().split()[1:]
            reg = np.zeros((int(shape[0]), int(shape[1])))
            for i, j, v in regressor_data:
                reg[int(i), int(j)] = v
            X_regressor = to_tensor(reg)
            X_regressor = torch.cat((self.J_regressor, X_regressor), dim=0)
            j_J_regressor = torch.zeros(self.J_regressor.shape[0], X_regressor.shape[0], device=device)
            for i in range(self.J_regressor.shape[0]):
                j_J_regressor[i, i] = 1
            j_v_template = X_regressor @ self.v_template
            # 
            # (25, 24)
            j_weights = X_regressor @ self.weights
            if self.use_pose_blending:
                j_posedirs = torch.einsum('ab, bde->ade', [X_regressor, torch.Tensor(posedirs)]).numpy()
                j_posedirs = np.reshape(j_posedirs, [-1, num_pose_basis]).T
                j_posedirs = to_tensor(j_posedirs)
                self.register_buffer('j_posedirs', j_posedirs)
            else:
                self.j_posedirs = None
            if self.use_shape_blending:
                j_shapedirs = torch.einsum('vij,kv->kij', [self.shapedirs, X_regressor])
                self.register_buffer('j_shapedirs', j_shapedirs)
            else:
                self.j_shapedirs = None
            self.register_buffer('j_weights', j_weights)
            self.register_buffer('j_v_template', j_v_template)
            self.register_buffer('j_J_regressor', j_J_regressor)
       
        self.num_pca_comps = kwargs['num_pca_comps']
        self.use_pca = kwargs['use_pca']
        self.use_flat_mean = kwargs['use_flat_mean']
        if self.use_pca:
            self.NUM_POSES = self.num_pca_comps + 3
        else:
            self.NUM_POSES = 45 + 3

        val = to_tensor(to_np(data['hands_mean'].reshape(1, -1)), dtype=dtype)
        self.register_buffer('mHandsMean', val)
        val = to_tensor(to_np(data['hands_components'][:self.num_pca_comps, :]), dtype=dtype)
        self.register_buffer('mHandsComponents', val)
        
        self.to(self.device)
    
    @staticmethod
    def extend_hand(poses, use_pca, use_flat_mean, coeffs, mean):
        if use_pca:
            poses = poses @ coeffs
        if not use_flat_mean:
            poses = poses + mean
        return poses

    def extend_pose(self, poses):
        if self.model_type == 'mano' and poses.shape[-1] == 48 and self.use_flat_mean:
            return poses
        # skip mano
        if self.model_type == 'mano':
            poses_hand = self.extend_hand(poses[..., 3:], self.use_pca, self.use_flat_mean,
                self.mHandsComponents, self.mHandsMean)
            poses = torch.cat([poses[..., :3], poses_hand], dim=-1)
            return poses

    def forward(self, poses, shapes, Rh=None, Th=None, expression=None, 
        v_template=None,
        return_verts=True, return_tensor=True, return_smpl_joints=True, 
        only_shape=False, pose2rot=True, **kwargs):
        """ Forward pass for SMPL model

        Args:
            poses (n, 72)
            shapes (n, 10)
            Rh (n, 3): global orientation
            Th (n, 3): global translation
            return_verts (bool, optional): if True return (6890, 3). Defaults to False.
        """
        if 'torch' not in str(type(poses)):
            dtype, device = self.dtype, self.device
            poses = to_tensor(poses, dtype, device)
            shapes = to_tensor(shapes, dtype, device)
            if Rh is not None:
                Rh = to_tensor(Rh, dtype, device)
            if Th is not None:
                Th = to_tensor(Th, dtype, device)
            if expression is not None:
                expression = to_tensor(expression, dtype, device)

        bn = poses.shape[0]
        # process Rh, Th
        if Rh is None:
            Rh = torch.zeros(bn, 3, device=poses.device)
        if Th is None:
            Th = torch.zeros(bn, 3, device=poses.device)
        
        if len(Rh.shape) == 2: # angle-axis
            rot = batch_rodrigues(Rh)
        else:
            rot = Rh
        transl = Th.unsqueeze(dim=1)
        # process shapes
        if shapes.shape[0] < bn:
            shapes = shapes.expand(bn, -1)
        if expression is not None and self.model_type == 'smplx':
            shapes = torch.cat([shapes, expression], dim=1)
        # process poses
        if pose2rot: # if given rotation matrix, no need for this
            poses = self.extend_pose(poses)
        if return_verts or not self.use_joints:
            if v_template is None:
                v_template = self.v_template
            vertices, joints = self.lbs(shapes, poses, v_template,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.weights, pose2rot=pose2rot, dtype=self.dtype,
                                use_pose_blending=self.use_pose_blending, use_shape_blending=self.use_shape_blending, J_shaped=self.J_shaped)
        else:
            NotImplementedError()
            # vertices, joints = self.lbs(shapes, poses, self.j_v_template,
            #                     self.j_shapedirs, self.j_posedirs,
            #                     self.j_J_regressor, self.parents,
            #                     self.j_weights, pose2rot=pose2rot, dtype=self.dtype, only_shape=only_shape,
            #                     use_pose_blending=self.use_pose_blending, use_shape_blending=self.use_shape_blending, J_shaped=self.J_shaped)
            # if return_smpl_joints:
            #     vertices = vertices[:, :self.J_regressor.shape[0], :]
            # else:
            #     vertices = vertices[:, self.J_regressor.shape[0]:, :]
        vertices = torch.matmul(vertices, rot.transpose(1, 2)) + transl
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()
        if return_smpl_joints:
            return vertices, joints
        else:
            return vertices
    
    def init_params(self, nFrames=1, nShapes=1, ret_tensor=False):
        params = {
            'poses': np.zeros((nFrames, self.NUM_POSES)),
            'shapes': np.zeros((nShapes, NUM_SHAPES)),
            'Rh': np.zeros((nFrames, 3)),
            'Th': np.zeros((nFrames, 3)),
        }
    
        if ret_tensor:
            for key in params.keys():
                params[key] = to_tensor(params[key], self.dtype, self.device)
        return params

    def check_params(self, body_params):
        model_type = self.model_type
        nFrames = body_params['poses'].shape[0]
        if body_params['poses'].shape[1] != self.NUM_POSES:
            body_params['poses'] = np.hstack((body_params['poses'], np.zeros((nFrames, self.NUM_POSES - body_params['poses'].shape[1]))))
        if model_type == 'smplx' and 'expression' not in body_params.keys():
            body_params['expression'] = np.zeros((nFrames, NUM_EXPR))
        return body_params

    @staticmethod    
    def merge_params(param_list, share_shape=True):
        output = {}
        for key in ['poses', 'shapes', 'Rh', 'Th', 'expression']:
            if key in param_list[0].keys():
                output[key] = np.vstack([v[key] for v in param_list])
        if share_shape:
            output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
        # add other keys
        for key in param_list[0].keys():
            if key in output.keys():
                continue
            output[key] = np.stack([v[key] for v in param_list])
        return output
    
    @staticmethod
    def select_nf(params_all, nf):
        output = {}
        for key in ['poses', 'Rh', 'Th']:
            output[key] = params_all[key][nf:nf+1, :]
        if 'expression' in params_all.keys():
            output['expression'] = params_all['expression'][nf:nf+1, :]
        if params_all['shapes'].shape[0] == 1:
            output['shapes'] = params_all['shapes']
        else:
            output['shapes'] = params_all['shapes'][nf:nf+1, :]
        return output