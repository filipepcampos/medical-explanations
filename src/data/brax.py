import numpy as np
import os
from torch.utils.data import Dataset
from torch import utils
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import lightning as L

# Obtained from
# https://github.com/kamenbliznashki/chexpert/blob/master/dataset.py


class Brax(Dataset):
    attr_names = ["Cardiomegaly"]
    prefix = "files/brax/1.1.0/"

    def __init__(self, root, mode="train", transform=None, data_filter=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        assert mode in ["train", "valid", "test"]
        self.mode = mode

        df = self._maybe_process(data_filter)
        # Split data into train, validation and test
        train_df = df.sample(frac=0.8, random_state=0)
        val_df = df.drop(train_df.index)
        test_df = val_df.sample(frac=0.8, random_state=0)
        val_df = val_df.drop(test_df.index)
        self.data = (
            val_df if mode == "valid" else train_df if mode == "train" else test_df
        )

        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 2]  # 'Path' column is 2
        img = Image.open(os.path.join(self.root, self.prefix, img_path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(np.float32)
        attr = torch.from_numpy(attr).item()

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = self.data.index[
            idx
        ]  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
        # self.data.index(idx) pulls the index in the original dataframe and not the subset

        return img, attr

    def __len__(self):
        return len(self.data)

    def _maybe_process(self, data_filter):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        #    3. apply attr filters as a dictionary {data_attribute: value_to_keep} e.g. {'Frontal/Lateral': 'Frontal'}
        return self._load_and_preprocess_training_data(
            os.path.join(self.root, "master_spreadsheet_update.csv"),
            data_filter,
        )

    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)

        # 2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1, 1)

        if data_filter is not None:
            # 3. apply attr filters
            # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
            for k, v in data_filter.items():
                train_df = train_df[train_df[k] == v]

        return train_df


class BraxDataModule(L.LightningDataModule):
    def __init__(self, root: str, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        return utils.data.DataLoader(
            self._get_dataset("train"),
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return utils.data.DataLoader(
            self._get_dataset("valid"),
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return utils.data.DataLoader(
            self._get_dataset("test"),
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def _get_dataset(self, split):
        data_filter = {
            "ViewPosition": "PA",
        }

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=0.5,
                    std=0.5,
                ),
            ],
        )

        return Brax(
            root=self.hparams.root,
            mode=split,
            transform=transform,
            data_filter=data_filter,
        )
