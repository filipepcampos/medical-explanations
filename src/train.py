import lightning as L
import torchvision
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.tuner import Tuner
from mimic_cxr_jpg_loader.modifiers import *

from data.mimic_datamodule import MIMICCXRDataModule
from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier

L.pytorch.seed_everything(42, workers=True)


densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
classifier_module = ExplainableClassifier(densenet)

datamodule = MIMICCXRDataModule(
    root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
    split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
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
            monitor="val_f1", filename="best_model", save_top_k=1, mode="max"
        ),
        pl_callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ],
    deterministic=True,
)

tuner = Tuner(trainer)
tuner.scale_batch_size(classifier_module, datamodule=datamodule)
trainer.fit(classifier_module, datamodule)  # TODO: Moreeee args
