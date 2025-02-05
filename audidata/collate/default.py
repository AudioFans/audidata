from torch.utils.data._utils.collate import collate, default_collate_fn_map
import torch


def collate_fn(batch: list):
    r"""The same as the default collate_fn in PyTorch. Except when input is a 
    list, then return a list."""

    def _collate_list_fn(batch, *, collate_fn_map=None):
        return batch

    collate_fn_map = default_collate_fn_map.copy()
    collate_fn_map.update({list: _collate_list_fn})

    return collate(batch, collate_fn_map=collate_fn_map)