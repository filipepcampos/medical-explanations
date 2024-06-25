from collections import OrderedDict

import lightning as L
import torch
import flwr as fl


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: L.LightningModule, datamodule: L.LightningDataModule):
        self.model = model
        self.datamodule = datamodule
        self.privacy_engine_state_dict = None

    def get_parameters(self, config):
        return _get_parameters(self.model.model)

    def set_parameters(self, parameters):
        _set_parameters(self.model.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = L.Trainer(max_epochs=1)
        if self.privacy_engine_state_dict:
            self.model.privacy_engine.load_state_dict(self.privacy_engine_state_dict)
        trainer.fit(self.model, self.datamodule)

        self.privacy_engine_state_dict = self.model.privacy_engine.state_dict()
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
