import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Normalize


class DolphinsDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        df: pd.DataFrame,
        split_name: str,
        transform=None,
    ):
        self.metadata_df = df.query(f"split_name == '{split_name}'")
        self.npy_path = npy_path
        default_transform = Compose(
            [
                Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
                Normalize(
                    mean=[0.30949154, 0.30949154, 0.30949154],
                    std=[0.12463536, 0.12463536, 0.12463536],
                ),
            ]
        )
        self.transform = transform if transform else default_transform
        self.data = np.load(npy_path, mmap_mode="r")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.transform(self.data[idx]).clone().detach().float() / 255.0
        label = 0 if self.metadata_df.loc[:, "label"].iloc[idx] == "no_call" else 1
        label = torch.tensor(label, dtype=torch.long)
        return data, label
