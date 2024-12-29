from torch.utils.data import DataLoader

from audidata.datasets import GTZAN
from audidata.io import StartCrop
from audidata.transforms import Mono, OneHot


if __name__ == '__main__':
    r"""Example. 

    GTZAN dataset: http://marsyas.info/index.html

    The dataset looks like:

        dataset_root (1.3 GB)
        └── genres
            ├── blues (100 files)
            ├── classical (100 files)
            ├── country (100 files)
            ├── disco (100 files)
            ├── hiphop (100 files)
            ├── jazz (100 files)
            ├── metal (100 files)
            ├── pop (100 files)
            ├── reggae (100 files)
            └── rock (100 files)
    """

    root = "/datasets/gtzan"

    dataset = GTZAN(
        root=root,
        split="train",
        test_fold=0,
        sr=24000,
        crop=StartCrop(clip_duration=30.),
        transform=Mono(),
        target_transform=OneHot(classes_num=GTZAN.CLASSES_NUM)
    )

    # Example of fetch a data
    print(dataset[0])

    # Example of dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:
        
        n = 0
        audio_path = data["audio_path"][n]
        audio = data["audio"][n].cpu().numpy()
        target = data["target"][n].cpu().numpy()
        label = data["label"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("audio:", audio.shape)
    print("target:", target.shape)
    print("label:", label)