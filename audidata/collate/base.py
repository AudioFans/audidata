from torch.utils.data._utils.collate import collate, collate_tensor_fn
import torch


def collate_list_fn(batch, *, collate_fn_map=None):
    return batch


collate_fn = default_collate_fn_map.copy()
collate_fn.update({list: collate_list_fn})


def collate_fn(batch): 
    collate(batch, collate_fn_map=default_collate_fn_map)