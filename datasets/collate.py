import numpy as np
import torch
import librosa


def collate_fn(list_data_dict):

    max_tokens = 784
    data_dict = {}

    for key in list_data_dict[0].keys():
        
        if key in ["tokens"]:

            max_len = max([len(dd[key]) for dd in list_data_dict])
            max_len = min(max_len, max_tokens)
            
            for dd in list_data_dict:

                dd[key] = librosa.util.fix_length(
                    data=np.array(dd[key]),
                    size=max_len, 
                    axis=0, 
                    constant_values=0
                )

            data_dict[key] = torch.LongTensor(np.stack([dd[key] for dd in list_data_dict], axis=0))

        elif key in ["audio_path", "segment_start_time"]:

            data_dict[key] = [dd[key] for dd in list_data_dict]

        else:
            data_dict[key] = torch.Tensor(np.stack([dd[key] for dd in list_data_dict], axis=0))

    return data_dict