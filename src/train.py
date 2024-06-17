import lightning as L
import torchvision
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.tuner import Tuner

from data.mimic_datamodule import MIMICCXRDataModule
from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)


L.pytorch.seed_everything(42, workers=True)


densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
classifier_module = ExplainableClassifier(densenet)

datamodule = MIMICCXRDataModule(
    root=config["mimic_path"],
    split_path=config["mimic_splits_path"],
    batch_size=32,
)

trainer = L.Trainer(
    max_epochs=100,
    logger=pl_loggers.WandbLogger(project="explanations", name="train"),  # TODO: adjust
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
        pl_callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ],
    deterministic=True,
)

tuner = Tuner(trainer)
tuner.scale_batch_size(classifier_module, datamodule=datamodule)
trainer.fit(classifier_module, datamodule)
