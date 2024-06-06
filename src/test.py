import lightning as L
import torchvision
from lightning.pytorch import loggers as pl_loggers

from data.mimic_datamodule import MIMICCXRDataModule
from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier

L.pytorch.seed_everything(42, workers=True)


densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
classifier_module = ExplainableClassifier.load_from_checkpoint(
    "/nas-ctm01/homes/fpcampos/dev/explanations/explanations/tz9u4b0a/checkpoints/best_model.ckpt",
    model=densenet,
)  # TODO: This should a an arg

datamodule = MIMICCXRDataModule(
    root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
    split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
    batch_size=32,
)

trainer = L.Trainer(
    logger=pl_loggers.WandbLogger(project="explanations", name="train"),  # TODO: adjust
    deterministic=True,
)

# trainer.test(classifier_module, datamodule=datamodule)  # TODO: Moreeee args
trainer.test(
    classifier_module,
    dataloaders=datamodule.test_dataloader(),
)  # TODO: Moreeee args
