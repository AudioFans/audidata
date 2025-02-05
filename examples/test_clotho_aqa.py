from torch.utils.data import DataLoader

from audidata.datasets import ClothoAQA
from audidata.io.crops import StartCrop, RandomCrop
from audidata.utils import RandomChoice
from audidata.transforms import Mono, TextNormalization


if __name__ == '__main__':
    r"""Example. 

    Clotho-AQA dataset: https://zenodo.org/records/6473207

    The dataset looks like:

        clotho-aqa (3.7 GB)
        ├── audio_files (2894 wavs)
        ├── clotho_aqa_metadata.csv
        ├── clotho_aqa_test.csv
        ├── clotho_aqa_train.csv
        ├── clotho_aqa_val.csv
        └── LICENSE.txt
    """

    root = "/datasets/clotho-aqa"

    clip_duration = 10.
    crop1 = StartCrop(clip_duration=clip_duration)
    crop2 = RandomCrop(clip_duration=clip_duration, end_pad=0.)
    crop = RandomChoice(callables=[crop1, crop2], weights=[0.5, 0.5])

    dataset = ClothoAQA(
        root=root,
        split="train",
        sr=44100,
        crop=crop,
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
        start_time = data["start_time"][n].cpu().numpy()
        audio = data["audio"][n].cpu().numpy()
        question = data["question"][n]
        target = data["target"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("start_time:", start_time)
    print("audio:", audio.shape)
    print("question:", question)
    print("target:", target)