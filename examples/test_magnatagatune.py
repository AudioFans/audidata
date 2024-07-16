from torch.utils.data import DataLoader

from audidata.datasets import MagnaTagATune

if __name__ == "__main__":
    r"""Example.
    """

    from torch.utils.data import DataLoader

    root = "/datasets/magnatagatune"

    sr = 16000

    dataset = MagnaTagATune(
        root=root,
        split="train",
        sr=sr,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:
        n = 0
        audio_path = data["audio_path"][n]
        audio = data["audio"][n].cpu().numpy()
        target = data["target"][n].cpu().numpy()
        label = data["labels"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("audio:", audio.shape)
    print("target:", target.shape)
    print("label:", label)