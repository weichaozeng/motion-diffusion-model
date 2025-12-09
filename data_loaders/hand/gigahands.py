from .dataset import Dataset
import torch
import os
import json

class GigaHands(Dataset):
    dataname = "gigahands"

    def __init__(self, datapath="dataset/gigahands", **kwargs):
        super().__init__(**kwargs)
        self.datapath = datapath

        self.scenes = sorted(os.listdir(self.datapath))
        self.seqs_kp3d = []
        self.seqs_mano = []
        self._num_frames_in_video = []
        for scene in self.scenes:
            dir_kp3d = os.path.join(self.datapath, scene, "keypoints3d_mano_align")
            dir_mano = os.path.join(self.datapath, scene, "params")
            for seq in sorted(os.listdir(dir_kp3d)):
                with open(os.path.join(dir_mano, seq), 'r') as f:
                    _data = json.load(f)
                # left hand
                self.seqs_kp3d.append(os.path.join(dir_kp3d, seq))
                self.seqs_mano.append(os.path.join(dir_mano, seq))
                self._num_frames_in_video.append(len(_data["left"]["poses"]))
                # right hand
                self.seqs_kp3d.append(os.path.join(dir_kp3d, seq))
                self.seqs_mano.append(os.path.join(dir_mano, seq))
                self._num_frames_in_video.append(len(_data["right"]["poses"]))
        
        self._train = list(range(100, len(self.seqs_kp3d)))
        self._test = list(range(0, 100))


    def _load_rotvec(self, ind, frame_ix, flip_left=True):
        mano_params_path = self.seqs_mano[ind]
        with open(mano_params_path, 'r') as f:
            mano_data = json.load(f)
        is_left_hand = (ind % 2 == 0)
        if is_left_hand:
            full_poses = torch.tensor(mano_data["left"]["poses"], dtype=torch.float32) 
        else:
            full_poses = torch.tensor(mano_data["right"]["poses"], dtype=torch.float32)
        poses = full_poses[frame_ix].reshape(-1, 16, 3)  # (num_frames, 16, 3)
        if is_left_hand and flip_left:
            poses[:, :, 1] *= -1
            poses[:, :, 2] *= -1
        return poses        
    
    def _load_joints3D(self, ind, frame_ix, flip_left=True):
        joints_path = self.seqs_kp3d[ind]
        with open(joints_path, 'r') as f:
            joints_data = json.load(f)
        full_joints = torch.tensor(joints_data, dtype=torch.float32)
        batch_joints = full_joints[frame_ix]  # (num_frames, 42, 3)
        is_left_hand = (ind % 2 == 0) 
        if is_left_hand:
            joints3D = batch_joints[:, :21]  # (num_frames, 21, 3)
        else:
            joints3D = batch_joints[:, 21:]  # (num_frames, 21, 3)
        if is_left_hand and flip_left:
            joints3D[:, 0] *= -1
        return joints3D


if __name__ == "__main__":
    dataset = GigaHands(split="train", num_frames=128, sampling="conseq", pose_rep="rot6d")
    print(len(dataset))
    sample = dataset[0]
    print("debug")