from torch.utils.data._utils.collate import default_collate_fn_map, collate


def collate_list_fn(batch, *, collate_fn_map=None):
    return batch


class CollateToken:
    r"""Collate music transcription tokens.
    """
    def __init__(self):

        default_collate_fn_map.update({list: collate_list_fn})

    def __call__(self, batch: list) -> dict:
        
        batch = collate(batch=batch, collate_fn_map=default_collate_fn_map)

        # Shorten sequence length
        max_tokens = batch["tokens_num"].max().item()
        batch["token"] = batch["token"][:, 0 : max_tokens]
        batch["mask"] = batch["mask"][:, 0 : max_tokens]
        
        return batch