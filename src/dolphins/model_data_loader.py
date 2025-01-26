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
    ):
        self.metadata_df = df.query(f"split_name == '{split_name}'").sort_values(
            ["split_name", "split_index"]
        )
        self.transform = Compose(
            [
                Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
                Normalize(
                    mean=[0.30949154, 0.30949154, 0.30949154],
                    std=[0.12463536, 0.12463536, 0.12463536],
                ),
            ]
        )
        self.data = np.load(npy_path, mmap_mode="r")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        selected_metadata = self.metadata_df.iloc[idx]
        split_index = selected_metadata["split_index"]
        label = 0 if selected_metadata["label"] == "no_call" else 1
        data = self.transform(self.data[split_index]).clone().detach().float() / 255.0
        label = torch.tensor(label, dtype=torch.long)
        return data, label
