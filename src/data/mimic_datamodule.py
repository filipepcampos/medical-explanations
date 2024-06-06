import lightning as L
from torch import utils
from torchvision import transforms
from torchvision.datasets import ImageFolder
from mimic_cxr_jpg_loader.modifiers import (
    Split,
    FilterByViewPosition,
    FilterBySplit,
    BinarizePathology,
    Pathology,
    ViewPosition,
)
from data.mimic_cxr_jpg import MIMIC_CXR_JPG


class MIMICCXRDataModule(L.LightningDataModule):
    def __init__(self, root: str, split_path: str, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        return utils.data.DataLoader(
            self._get_dataset(Split.TRAIN),
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return utils.data.DataLoader(
            self._get_dataset(Split.VAL),
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return utils.data.DataLoader(
            self._get_dataset(Split.TEST),
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def _get_dataset(self, split):
        return MIMIC_CXR_JPG(
            root=self.hparams.root,
            split_path=self.hparams.split_path,
            modifiers=[
                FilterByViewPosition(ViewPosition.PA),
                FilterBySplit(split),
                BinarizePathology(Pathology.CARDIOMEGALY),
            ],
        )


class SyntheticMIMICCXRDataModule(L.LightningDataModule):
    def __init__(self, root: str, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()

        dataset = ImageFolder(
            self.hparams.root,
            transform=transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ],
            ),
        )

        # Wrap dataset, return label as float
        class SyntheticDataset(utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                x, y = self.dataset[idx]
                return x, float(y)

        dataset = SyntheticDataset(dataset)

        # Split dataset # TODO: Undo splitting
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = utils.data.random_split(
            dataset,
            [train_size, test_size],
        )
        self.val_dataset, self.train_dataset = utils.data.random_split(
            dataset,
            [int(0.1 * len(dataset)), int(0.9 * len(dataset))],
        )

    def train_dataloader(self):

        return utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )
