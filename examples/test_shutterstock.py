from torch.utils.data import DataLoader

from audidata.datasets import Shutterstock
from audidata.io import RandomCrop
from audidata.transforms import Mono, TextNormalization


if __name__ == '__main__':
    r"""Example. 

    Shutterstock is a private dataset for academic purpose only.

    The dataset looks like:

        dataset_root (409 GB)
        ├── flac (28,558 files)
        └── 28kdescriptions.csv
    """

    root = "/datasets/shutterstock/shutterstock_mp3"

    sr = 48000
    
    dataset = Shutterstock(
        root=root,
        sr=sr,
        crop=RandomCrop(clip_duration=10.),
        transform=Mono(),
        target_transform=TextNormalization()
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
        target = data["target"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("audio:", audio.shape)
    print("caption:", caption)
    print("target:", target)