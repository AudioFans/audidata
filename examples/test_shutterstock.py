from torch.utils.data import DataLoader

from audidata.datasets import Shutterstock
from audidata.io.crops import RandomCrop


if __name__ == '__main__':
    r"""Example. 

    Private Shutterstock dataset for academic purpose only.

    The dataset looks like:

        dataset_root (409 GB)
        ├── flac (28,558 files)
        └── 28kdescriptions.csv
    """

    root = "/datasets/shutterstock/shutterstock_mp3"

    sr = 16000
    
    dataset = Shutterstock(
        root=root,
        sr=sr,
        crop=RandomCrop(clip_duration=10.),
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