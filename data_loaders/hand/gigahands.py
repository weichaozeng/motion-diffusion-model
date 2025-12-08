from .dataset import Dataset
import torch


class GigaHands(Dataset):
    dataname = "gigahands"

    def __init__(self, datapath="dataset/gigahands", **kwargs):
        super().__init__(**kwargs)
        self.datapath = datapath

        