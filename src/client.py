from collections import OrderedDict

import pytorch_lightning as pl
import torchvision
import torch
import flwr as fl

from data.mimic_datamodule import MIMICCXRDataModule
from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return _get_parameters(self.model.model)

    def set_parameters(self, parameters):
        _set_parameters(self.model.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return self.get_parameters(config={}), 55000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, 10000, {"loss": loss}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    # Model and data
    densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    model = ExplainableClassifier(densenet)

    datamodule = MIMICCXRDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
        split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
        batch_size=32,
    )
    train_loader, val_loader, test_loader = datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()