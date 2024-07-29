import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from pathlib import Path
from torch.utils.data import DataLoader

from audidata.datasets import Slakh2100 
from audidata.tokenizers import (ConcatTokenizer, SpecialTokenizer, NameTokenizer, 
    TimeTokenizer, PitchTokenizer, VelocityTokenizer, ProgramTokenizer, DrumTokenizer)
from audidata.utils import Compose
from audidata.io.crops import RandomCrop
from audidata.transforms.midi import MultiTrackNote2Token
from audidata.collate import CollateToken


if __name__ == '__main__':
    root = "/root/slakh2100_flac_redux"
    sr = 16000
    max_tokens = 8192

    tokenizer = ConcatTokenizer([
        SpecialTokenizer(),
        NameTokenizer(),
        TimeTokenizer(),
        PitchTokenizer(),
        DrumTokenizer(),
        VelocityTokenizer(),
        ProgramTokenizer()
    ])

    target_transforms = Compose(callables=[
        MultiTrackNote2Token(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
    ])

    dataset = Slakh2100(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=10., end_pad=9.9),
        target_transform=target_transforms,
    )

    collate_fn = CollateToken()

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=1, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    for data in dataloader:
        n = 0
        audio = data["audio"][n].cpu().numpy()
        tokens = data["token"][n].cpu().numpy()
        masks = data["mask"][n].cpu().numpy()
        tokens_num = data["tokens_num"][n].cpu().numpy()
        break

    # ------ Visualize ------
    print("audio:", audio.shape)
    print("tokens:", tokens.shape)
    print("masks:", masks.shape)
    print("tokens_num:", tokens_num)

    # Write audio
    Path("results").mkdir(parents=True, exist_ok=True)
    out_path = "results/out.wav"
    soundfile.write(file=out_path, data=audio.T, samplerate=sr)
    print("Write out audio to {}".format(out_path))