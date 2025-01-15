from torch.utils.data import DataLoader

from audidata.datasets import FreeSpokenDigit
from audidata.io import StartCrop
from audidata.transforms import Mono


if __name__ == '__main__':
    r"""Example. 

    Free Spoken Digit dataset: https://github.com/Jakobovski/free-spoken-digit-dataset

    The dataset looks like:

        free-spoken-digit-dataset (22 MB)
        └── recordings (3,000 wavs)
            ├── 0_george_0.wav
            ├── 1_george_0.wav
            └── ...
    """

    root = "/datasets/free-spoken-digit-dataset"

    dataset = FreeSpokenDigit(
        root=root,
        split="train",
        sr=8000,
        crop=StartCrop(clip_duration=3.),
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
        label = data["label"][n]
        speaker = data["speaker"][n]
        caption = data["caption"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("audio:", audio.shape)
    print("label:", label)
    print("speaker:", speaker)
    print("caption:", caption)