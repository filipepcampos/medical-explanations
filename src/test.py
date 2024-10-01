import lightning as L
import torchvision
from lightning.pytorch import loggers as pl_loggers

from data.mimic_datamodule import MIMICCXRDataModule
from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

L.pytorch.seed_everything(42, workers=True)

densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
classifier_module = ExplainableClassifier.load_from_checkpoint(
    config["explainable_classifier"],
    model=densenet,
)

datamodule = MIMICCXRDataModule(
    root=config["mimic_path"],
    split_path=config["mimic_splits_path"],
    batch_size=32,
)

trainer = L.Trainer(
    logger=pl_loggers.WandbLogger(project="explanations", name="train"),
    deterministic=True,
)

trainer.test(
    classifier_module,
    dataloaders=datamodule.test_dataloader(),
)
