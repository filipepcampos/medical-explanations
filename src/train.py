import os

import lightning as L
import torchmetrics
import torchvision
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.tuner import Tuner
from mimic_cxr_jpg_loader.modifiers import *
from torch import nn, optim, utils

from data.mimic_cxr_jpg import MIMIC_CXR_JPG
from models.densenet import DenseNet121

L.pytorch.seed_everything(42, workers=True)


class ExplainableClassifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.train_f1 = torchmetrics.classification.F1Score(task="binary")
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_f1 = torchmetrics.classification.F1Score(task="binary")

    def training_step(self, batch, batch_idx):
        x, y = batch

        prediction, _ = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

        loss = nn.functional.binary_cross_entropy(prediction, y)
        self.log("train_loss", loss)
        self.train_acc(prediction, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        self.train_f1(prediction, y)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prediction, features = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

        loss = nn.functional.binary_cross_entropy(prediction, y)
        self.log("val_loss", loss)
        self.val_acc(prediction, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_f1(prediction, y)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
classifier_module = ExplainableClassifier(densenet)


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


datamodule = MIMICCXRDataModule(
    root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
    split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
    batch_size=32,
)

trainer = L.Trainer(
    max_epochs=10,
    logger=pl_loggers.WandbLogger(project="explanations", name="train"),  # TODO: adjust
    callbacks=[
        pl_callbacks.ModelSummary(
            max_depth=3,
        ),
        pl_callbacks.ModelCheckpoint(
            monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
        ),
        pl_callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
    ],
    deterministic=True,
)

tuner = Tuner(trainer)
tuner.scale_batch_size(classifier_module, datamodule=datamodule)
trainer.fit(classifier_module, datamodule)  # TODO: Moreeee args
