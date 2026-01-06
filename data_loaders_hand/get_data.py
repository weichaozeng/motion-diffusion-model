from torch.utils.data import DataLoader
import torch
import numpy as np

def get_dataset_loader(
    name, 
    batch_size, 
    num_frames, 
    split='train',
    pose_rep="rot6d", 
    device=None,
    translation=False,
    glob=True
):
    dataset = get_dataset(name, num_frames, split=split, pose_rep=pose_rep, device=device, translation=translation, glob=glob)
    collate = get_collate_fn(name, batch_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader


def get_dataset(
    name, 
    num_frames, 
    split='train',
    pose_rep="rot6d", 
    device=None,
    translation=False,
    glob=True
): 
    DATA = get_dataset_class(name)
    dataset = DATA(split=split, num_frames=num_frames, pose_rep=pose_rep, translation=translation, glob=glob)
    return dataset


def get_dataset_class(name):
    if name == "gigahands":
        from .hand.gigahands import GigaHands
        return GigaHands
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')
    

def get_collate_fn(name,batch_size=1):
    return collate

def collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}
    def process_item(item_list):
        first_sample = item_list[0]
        # tensor: pose, rotmat, etc.
        if torch.is_tensor(first_sample):
            if first_sample.is_floating_point():
                return torch.stack([it.float() for it in item_list])
            return torch.stack(item_list)
        # np.array
        elif isinstance(first_sample, (np.ndarray, list)) and isinstance(first_sample[0], (int, float)):
            return torch.as_tensor(item_list)
        # dict: cam, etc.
        elif isinstance(first_sample, dict):
            return {key: process_item([it[key] for it in item_list]) for key in first_sample}
        # others: str, etc.
        else:
            return item_list
    batched_dict = {key: process_item([b[key] for b in batch]) for key in batch[0]}

    return batched_dict
