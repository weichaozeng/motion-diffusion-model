from dataset import Dataset
from pathlib import Path
import torch
import os
import json
import random
import numpy as np
import pickle
from model_hand.rotation2xyz import Rotation2xyz

beta_dir = {
    '01': '/home/zvc/Data/DexYCB/calibration/mano_20200709_140042_subject-01_right',
    '02': '/home/zvc/Data/DexYCB/calibration/mano_20200813_143449_subject-02_right',
    '03': '/home/zvc/Data/DexYCB/calibration/mano_20200820_133405_subject-03_right',
    '04': '/home/zvc/Data/DexYCB/calibration/mano_20200903_101911_subject-04_right',
    '05': '/home/zvc/Data/DexYCB/calibration/mano_20200908_140650_subject-05_right',
    '06': '/home/zvc/Data/DexYCB/calibration/mano_20200918_110920_subject-06_right',
    '07': '/home/zvc/Data/DexYCB/calibration/mano_20200807_132210_subject-07_right',
    '08': '/home/zvc/Data/DexYCB/calibration/mano_20201002_103251_subject-08_right',
    '09': '/home/zvc/Data/DexYCB/calibration/mano_20200514_142106_subject-09_right',
    '10': '/home/zvc/Data/DexYCB/calibration/mano_20201022_105224_subject-10_right',
}

def read_anno_from_dir(anno_dir):
    dir_path = Path(anno_dir)
    npz_files = sorted(dir_path.glob("*.npz"))
    all_pose_m = []
    all_kp_2d = []
    all_kp_3d = []
    for file_path in npz_files:
        with np.load(file_path) as data:
            pose_m = data['pose_m']  
            kp_2d = data['joint_2d']   
            kp_3d = data['joint_3d']   
            all_pose_m.append(pose_m)
            all_kp_2d.append(kp_2d)
            all_kp_3d.append(kp_3d)
    return np.array(all_pose_m), np.array(all_kp_2d), np.array(all_kp_3d)



class DexYCB(Dataset):
    dataname = "dexycb"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seqs_y = []
        self.seqs_kp3d = []
        self.seqs_mano = []
        self.seqs_cam = []
        self.seqs_video = []
        
        data_path = Path("/home/zvc/Project/VHand/test_dataset/DexYCB/vhand/hamer_out")
        anno_root = Path("/home/zvc/Data/DexYCB/s0_train") 
        rgb_root = Path("/home/zvc/Data/DexYCB/s0_train_rgb")   
        cam_root = Path("/home/zvc/Data/DexYCB/calibration") 

        outs = sorted(os.listdir(anno_root))

        for out in outs:
            beta_name = out.split('_')[0].split('-')[-1]
            cam = out.split('_')[-1]
            # video
            video_path = rgb_root / out
            # anno
            all_pose_m, all_pose_y, all_kp_2d, all_kp_3d = read_anno_from_dir(anno_root / out)
            # beta
            beta = Path(beta_dir[beta_name]) / 'mano.yml'
            # cam
            cam_path = cam_root / 'intrinsics' / f'{cam}_640x480.yml'

            for track in os.listdir(data_path / out / 'results' / 'track'):
                self.seqs_y.append(data_path / out / 'results' / 'track' / track)
                self.seqs_kp3d.append(all_kp_3d)
                self.seqs_mano.append(all)
                self.seqs_cam.append(cam)
                self.seqs_video.append(video_path)
                
    
        
        _all_indices = list(range(len(self.seqs_y)))
        random.seed(42)
        random.shuffle(_all_indices)

        val_size = 128
        self._val = _all_indices[:val_size]
        self._train = _all_indices[val_size:]

        self.rot2xyz = Rotation2xyz(device='cpu')




if __name__ == "__main__":
    dataset = DexYCB()
