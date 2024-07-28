from torch.utils.data import DataLoader
from audidata.datasets import ESC50

if __name__ == "__main__":
    r"""Example.
    """

    root = "/datasets/esc50"

    sr = 44100

    dataset = ESC50(
        root=root,
        sr=sr,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=4)

    for data in dataloader:
        n = 0
        audio_path = data["audio_path"][n]
        start_time = data["start_time"][n].cpu().numpy()
        audio = data["audio"][n].cpu().numpy()
        target = data["target"][n].cpu().numpy()
        label = data["label"][n]
        break

    # ------ Visualize ------
    print("audio_path:", audio_path)
    print("start_time:", start_time)
    print("audio:", audio.shape)
    print("target:", target.shape)
    print("label:", label)

    # Evaluate the model on the dataset running the K(5)-fold cross-validation.
    K = 5
    for fold in range(K):
        print(f"Dataset for the {fold}-th fold of 5 folds")
        for fold_train in range(K):
            if fold_train != fold:
                train_set = ESC50(root, fold=fold, split="train")
            else:
                test_sst = ESC50(root, fold=fold, split="test")
        print(f"length of train_set = {len(train_set)}")
        print(f"length of train_set = {len(test_sst)}")
