from torch.utils.data._utils.collate import collate, collate_tensor_fn
import torch


def collate_list_fn(batch, *, collate_fn_map=None):
    return batch


collate_fn_map = {
    torch.Tensor: collate_tensor_fn,
    list: collate_list_fn,
}


class CollateToken:
    r"""Collate music transcription tokens.
    """
    def __init__(self):
        pass
        # default_collate_fn_map.update({list: collate_list_fn})

    def __call__(self, batch: list) -> dict:
        
        batch = collate(batch=batch, collate_fn_map=collate_fn_map)

        # Shorten sequence length
        max_tokens = batch["tokens_num"].max().item()
        batch["token"] = batch["token"][:, 0 : max_tokens]
        batch["mask"] = batch["mask"][:, 0 : max_tokens]
        
        return batch
    
class CollateDictToken:
    r"""Collate music transcription tokens in DictTokenizer format.
    """
    def __init__(self):
        pass
        # default_collate_fn_map.update({list: collate_list_fn})
    
    def __call__(self, batch: list) -> dict:
            
        batch = collate(batch=batch, collate_fn_map=collate_fn_map)
        max_tokens = batch["tokens_num"].max().item()
        batch["mask"] = batch["mask"][:, 0 : max_tokens]
        
        all_keys = set()
        all_keys.update(batch["token"][0][0].keys())
        
        for i in range(len(batch["token"])):
            token_dict = {key: [] for key in all_keys}
            batch["token"][i] = batch["token"][i][:max_tokens]
            for element in batch["token"][i]:
                for key in all_keys:
                    token_dict[key].append(element.get(key, 0))
            for key in token_dict:
                token_dict[key] = torch.LongTensor(token_dict[key])
            batch["token"][i] = token_dict
        
        token_dict = {key: torch.stack([element[key] for element in batch["token"]], dim=0) for key in all_keys}
        batch["token"] = token_dict
        
        return batch