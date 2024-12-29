from torch.utils.data import DataLoader

from audidata.datasets import WavCaps
from audidata.io.crops import StartCrop


if __name__ == '__main__':
    r"""Example. 

    AudioCaps dataset: https://github.com/cdjkim/audiocaps

    The dataset looks like:

        dataset_root (131 GB)
        ├── Zip_files
        │   ├── AudioSet_SL (108,137 flac)
        │   ├── BBC_Sound_Effects (31,201 flac)
        │   ├── FreeSound (262,300 flac)
        │   └── SoundBible (1,320 flac)
        ├── Zip_files
        │   ├── AudioSet_SL
        │   │   └── as_final.json
        │   ├── BBC_Sound_Effects
        │   │   └── bbc_final.json
        │   ├── FreeSound
        │   │   └── fsd_final.json
        │   ├── SoundBible
        │   │   └── json_files_SoundBible_sb_final.json
        │   └── blacklist
        │       ├── blacklist_exclude_all_ac.json
        │       ├── blacklist_exclude_test_ac.json
        │       └── blacklist_exclude_ub8k_esc50_vggsound.json
        └── README.md
    """

    root = "/datasets/wavcaps"

    dataset = WavCaps(
        root=root,
        sr=32000,
        crop=StartCrop(clip_duration=10.)
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