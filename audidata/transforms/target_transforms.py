from typing import Dict, List

import librosa
import numpy as np

from data.tokenizers import BaseTokenizer


class Note2Token:
    r"""Target transform. Transform midi notes to tokens. Users may define their
    own target transforms.
    """

    def __init__(self, 
        clip_duration: int, 
        tokenizer: BaseTokenizer, 
        max_tokens: int
    ):
        
        self.clip_duration = 10.
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __call__(self, target_data: Dict) -> List[int]:
        
        notes = target_data["note"]

        # Notes to words
        words = ["<sos>"]

        for note in notes:

            onset_time = note.start
            offset_time = note.end
            pitch = note.pitch
            velocity = note.velocity

            if 0 <= onset_time <= self.clip_duration:

                words.append("name=note_on")
                words.append("time={}".format(onset_time))
                words.append("pitch={}".format(pitch))
                words.append("velocity={}".format(velocity))
                
            if 0 <= offset_time <= self.clip_duration:

                words.append("name=note_off")
                words.append("time={}".format(offset_time))
                words.append("pitch={}".format(pitch))

        words.append("<sos>")

        # Words to tokens
        tokens = np.array([self.tokenizer.stoi(w) for w in words])
        tokens_num = len(tokens)

        # Masks
        masks = np.ones_like(tokens)

        tokens = librosa.util.fix_length(data=tokens, size=self.max_tokens)
        masks = librosa.util.fix_length(data=masks, size=self.max_tokens)

        target_data["word"] = words
        target_data["token"] = tokens
        target_data["mask"] = masks
        target_data["tokens_num"] = tokens_num

        return target_data