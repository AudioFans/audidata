from torch.utils.data import DataLoader

from audidata.datasets import LibriSpeech
from audidata.io import StartCrop
from audidata.transforms import Mono


if __name__ == '__main__':
    r"""Example. 

    LibriSpeech [1] is a speech dataset containing 292,367 English speech 
    files. The total duration is 1,000 hours. All audio files are mono and 
    sampled at 16,000 Hz. The durations of the audio files range from 0.83 s to 
    35.16 s, with 99.9% of the files having durations of less than 20 s. After 
    decompression, the dataset size is 60 GB.

    [1] Vassil Panayotov, et al., LibriSpeech: an ASR corpus based on public 
    domain audio books", ICASSP 2015

    The dataset looks like:

        librispeech (1.3 GB)
        ├── dev-clean (40 folders)
        │   ├── 1272 (3 folders)
        │   │   ├── 128104
        │   │   │   ├── 1272-128104-0000.flac
        │   │   │   ├── ...
        │   │   │   ├── 1272-128104-0014.flac
        │   │   │   └── 1272-128104.trans.txt
        │   │    ...
        │    ...
        ├── dev-other (33 folders)
        ├── test-clean (40 folders)
        ├── test-other (33 folders)
        ├── train-clean-100 (251 folders)
        ├── train-clean-360 (921 folders)
        ├── train-other-500 (1166 folders)
        ├── BOOKS.TXT
        ├── CHAPTERS.TXT
        ├── LICENSE.TXT
        ├── README.TXT
        └── SPEAKERS.TXT
    """

    root = "/datasets/librispeech"

    dataset = LibriSpeech(
        root=root,
        splits=["train-clean-100", "train-clean-360", "train-other-500"],
        sr=16000,
        crop=StartCrop(clip_duration=30.),
        transform=Mono(),
    )

    # Example of fetch a data
    print(dataset[0])

    # Example of dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:
        
        n = 0
        audio_path = data["audio_path"][n]
        start_time = data["start_time"][n].cpu().numpy()
        audio = data["audio"][n].cpu().numpy()
        caption = data["caption"][n]
        target = data["target"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("start_time:", start_time)
    print("audio:", audio.shape)
    print("caption:", caption)
    print("target:", target)