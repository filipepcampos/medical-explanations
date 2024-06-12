from collections import OrderedDict

import lightning as L
import torchvision
import torch
import flwr as fl

from data.mimic_cxr_jpg import MIMICCXRDataModule
from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: L.LightningModule, datamodule: L.LightningDataModule):
        self.model = model
        self.datamodule = datamodule

    def get_parameters(self, config):
        return _get_parameters(self.model.model)

    def set_parameters(self, parameters):
        _set_parameters(self.model.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = L.Trainer(max_epochs=1)
        trainer.fit(self.model, self.datamodule)

        return (
            self.get_parameters(config={}),
            len(self.datamodule.train_dataloader()),
            {},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = L.Trainer()
        results = trainer.test(self.model, self.datamodule)
        loss = results[0]["test_loss"]
        acc = results[0]["test_acc"]
        f1 = results[0]["test_f1"]

        return (
            loss,
            len(self.datamodule.test_dataloader()),
            {"loss": loss, "acc": acc, "f1": f1},
        )


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    from data.chexpert import ChexpertDataModule
    from data.brax import BraxDataModule
    import argparse

    densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    model = ExplainableClassifier(densenet)
    parser = argparse.ArgumentParser()
    parser.add_argument("cid", type=str)

    def create_client(datamodule) -> FlowerClient:
        return FlowerClient(model, datamodule).to_client()

    def get_mimic_client() -> FlowerClient:
        datamodule = MIMICCXRDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
            split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
            batch_size=8,
        )
        return create_client(datamodule)

    def get_chexpert_client() -> FlowerClient:
        datamodule = ChexpertDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/CheXpert-small",
            batch_size=8,
        )
        return create_client(datamodule)

    def get_brax_client() -> FlowerClient:
        datamodule = BraxDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/BRAX/physionet.org",
            batch_size=8,
        )
        return create_client(datamodule)

    def client_fn(cid: str):
        if cid == "0":
            return get_mimic_client()
        elif cid == "1":
            return get_chexpert_client()
        elif cid == "2":
            return get_brax_client()
        raise Exception(f"Unknown client: {cid}")

    client = client_fn(parser.parse_args().cid)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
