from torch.utils.data import DataLoader

from audidata.datasets import AudioCaps
from audidata.io.crops import StartCrop


if __name__ == '__main__':
    r"""Example. 

    AudioCaps dataset: https://github.com/cdjkim/audiocaps

    The dataset looks like:

        dataset_root (131 GB)
        ├── train (49274 files)
        ├── val (494 files)
        ├── test (957 files)
        ├── train.csv
        ├── val.csv
        ├── test.csv
        ├── LICENSE.txt
        └── README.md
    """

    root = "/datasets/audiocaps"

    sr = 16000
    clip_duration = 10.

    dataset = AudioCaps(
        root=root,
        split="train",
        sr=sr,
        crop=StartCrop(clip_duration=clip_duration)
    )

    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:
        
        n = 0
        audio_path = data["audio_path"][n]
        start_time = data["start_time"][n].cpu().numpy()
        audio = data["audio"][n].cpu().numpy()
        caption = data["caption"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("start_time:", start_time)
    print("audio:", audio.shape)
    print("caption:", caption)