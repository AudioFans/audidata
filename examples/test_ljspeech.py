from torch.utils.data import DataLoader

from audidata.datasets import LJSpeech
from audidata.io import StartCrop
from audidata.transforms import Mono, OneHot


if __name__ == '__main__':
    r"""Example. 

    GTZAN dataset: https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

    The dataset looks like:

        LJSpeech-1.1 (3.6 GB)
        ├── wavs (13,100 .wavs)
        │   ├── LJ001-0001.wav
        │   └── ...
        ├── metadata.csv
        ├── README
        ├── train.txt (https://huggingface.co/datasets/flexthink/ljspeech)
        ├── valid.txt (https://huggingface.co/datasets/flexthink/ljspeech)
        └── test.txt (https://huggingface.co/datasets/flexthink/ljspeech)
    """

    root = "/datasets/LJSpeech-1.1"

    dataset = LJSpeech(
        root=root,
        split="train",
        sr=22050,
        crop=StartCrop(clip_duration=10.),
        transform=Mono(),
    )

    # Example of fetch a data
    print(dataset[0])

    # Example of dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:
        
        n = 0
        audio_path = data["audio_path"][n]
        audio = data["audio"][n].cpu().numpy()
        caption = data["caption"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("audio:", audio.shape)
    print("caption:", caption)