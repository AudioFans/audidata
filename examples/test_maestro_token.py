import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from pathlib import Path
from torch.utils.data import DataLoader

from audidata.datasets import MAESTRO
from audidata.tokenizers import (ConcatTokenizer, SpecialTokenizer, NameTokenizer, 
    TimeTokenizer, PitchTokenizer, VelocityTokenizer, DictTokenizer)
from audidata.utils import Compose
from audidata.io.crops import RandomCrop
from audidata.transforms.midi import PianoRoll, Note2Token, Note2DictToken
from audidata.collate import CollateToken, CollateDictToken


if __name__ == '__main__':
    r"""Example of loading audio and converting MIDI events to tokens.

    The MAESTRO V3.0.0 dataset: https://magenta.tensorflow.org/datasets/maestro

    The dataset looks like:

        dataset_root (131 GB)
        ├── 2004 (132 songs, wav + flac + midi + tsv)
        ├── 2006 (115 songs, wav + flac + midi + tsv)
        ├── 2008 (147 songs, wav + flac + midi + tsv)
        ├── 2009 (125 songs, wav + flac + midi + tsv)
        ├── 2011 (163 songs, wav + flac + midi + tsv)
        ├── 2013 (127 songs, wav + flac + midi + tsv)
        ├── 2014 (105 songs, wav + flac + midi + tsv)
        ├── 2015 (129 songs, wav + flac + midi + tsv)
        ├── 2017 (140 songs, wav + flac + midi + tsv)
        ├── 2018 (93 songs, wav + flac + midi + tsv)
        ├── LICENSE
        ├── maestro-v3.0.0.csv
        ├── maestro-v3.0.0.json
        └── README
    """

    root = "/root/maestro-v3.0.0"

    sr = 16000

    # Tokenizer. Users may define their own tokenizer
    # tokenizer = ConcatTokenizer([
    #     SpecialTokenizer(),
    #     NameTokenizer(),
    #     TimeTokenizer(),
    #     PitchTokenizer(),
    #     VelocityTokenizer()
    # ])
    tokenizer = DictTokenizer({
        "special": SpecialTokenizer(),
        "onset": ConcatTokenizer([
            TimeTokenizer(max_duration=10.),
            NameTokenizer(),
        ]),
        "pitch": PitchTokenizer(),
        "velocity": VelocityTokenizer(),
        "offset": ConcatTokenizer([
            TimeTokenizer(max_duration=10.),
            NameTokenizer(),
        ]),
    })

    target_transforms = Compose(callables=[
        PianoRoll(fps=100, pitches_num=128),
        Note2DictToken(tokenizer=tokenizer, max_tokens=4096)
    ])

    # Dataset
    dataset = MAESTRO(
        root=root,
        split="train",
        sr=sr,
        crop=RandomCrop(clip_duration=10., end_pad=9.9),
        target_transform=target_transforms,
    )

    # Collate. Users may define their own collate
    collate_fn = CollateDictToken()

    # Dataloader
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=16, 
        num_workers=0, 
        collate_fn=collate_fn
    )

    for data in dataloader:
        n = 0
        audio = data["audio"].cpu().numpy()
        frame_roll = data["frame_roll"].cpu().numpy()
        onset_roll = data["onset_roll"].cpu().numpy()
        offset_roll = data["offset_roll"].cpu().numpy()
        velocity_roll = data["velocity_roll"].cpu().numpy()
        tokens = data["token"]
        masks = data["mask"].cpu().numpy()
        tokens_num = data["tokens_num"].cpu().numpy()
        break

    # ------ Visualize ------
    print("audio:", audio.shape)
    print("frame_roll:", frame_roll.shape)
    print("onset_roll:", frame_roll.shape)
    print("offset_roll:", frame_roll.shape)
    print("velocity_roll:", frame_roll.shape)
    print("tokens:", tokens)
    for key in tokens:
        print(key, tokens[key].shape)
    print("masks:", masks)
    print("tokens_num:", tokens_num)
    print("vocab sizes: ", tokenizer.get_vocab_sizes())

    # # Write audio
    # Path("results").mkdir(parents=True, exist_ok=True)
    # out_path = "results/out.wav"
    # soundfile.write(file=out_path, data=audio.T, samplerate=sr)
    # print("Write out audio to {}".format(out_path))

    # # Mel spectrogram
    # mel = librosa.feature.melspectrogram(y=audio[0], sr=sr, n_fft=2048, 
    #     hop_length=160, n_mels=229, fmin=0, fmax=8000)

    # # Plot
    # fig, axs = plt.subplots(5, 1, sharex=True, figsize=(20, 15))
    # axs[0].matshow(np.log(mel), origin='lower', aspect='auto', cmap='jet')
    # axs[1].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[1].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[2].matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[3].matshow(offset_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[4].matshow(velocity_roll.T, origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # fig_path = "results/out.pdf"
    # plt.savefig(fig_path)
    # print("Write out fig to {}".format(fig_path))