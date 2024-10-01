import argparse

import torchvision
import lightning as L
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.tuner import Tuner

from data.mimic_cxr_jpg import MIMICCXRDataModule
from data.chexpert import ChexpertDataModule
from data.brax import BraxDataModule

from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)


L.pytorch.seed_everything(42, workers=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dataset", type=str, default="mimic-cxr")


def get_datamodule(dataset: str, batch_size: int):
    if dataset == "mimic-cxr":
        return MIMICCXRDataModule(
            root=config["mimic_path"],
            split_path=config["mimic_splits_path"],
            batch_size=batch_size,
        )
    if dataset == "chexpert":
        return ChexpertDataModule(
            root=config["chexpert_path"],
            batch_size=batch_size,
        )
    if dataset == "brax":
        return BraxDataModule(
            root=config["brax_path"],
            batch_size=batch_size,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def main():
    args = parser.parse_args()

    densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    classifier_module = ExplainableClassifier(densenet)

    datamodule = get_datamodule(args.dataset, args.batch_size)

    trainer = L.Trainer(
        max_epochs=100,
        logger=pl_loggers.WandbLogger(
            project="fl_cbe",
            name=f"centralized_{args.dataset}",
        ),
        callbacks=[
            pl_callbacks.ModelSummary(
                max_depth=3,
            ),
            pl_callbacks.ModelCheckpoint(
                monitor="val_f1",
                filename="best_model",
                save_top_k=1,
                mode="max",
            ),
            pl_callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        deterministic=True,
    )

    tuner = Tuner(trainer)
    tuner.scale_batch_size(classifier_module, datamodule=datamodule)
    trainer.fit(classifier_module, datamodule)
    trainer.test(classifier_module, datamodule)


if __name__ == "__main__":
    main()
