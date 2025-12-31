import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class SpiderDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = 1 if row["label"] == 1 else -1
        return {
            "question": row["question"],
            "schema_item": row["schema_item"],
            "label": label,
        }


class SpiderDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size, num_workers):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):  # noqa: ARG002
        self.train_df = pd.read_parquet(self.train_path)
        self.val_df = pd.read_parquet(self.val_path)

    def train_dataloader(self):
        dataset = SpiderDataset(self.train_df)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        dataset = SpiderDataset(self.val_df)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
