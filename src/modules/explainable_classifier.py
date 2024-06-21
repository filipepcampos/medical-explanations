import lightning as L
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader


class ExplainableClassifier(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        enable_dp: bool = False,
        delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()

        self.model = model

        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.train_f1 = torchmetrics.classification.F1Score(task="binary")
        self.train_auc = torchmetrics.classification.AUROC(task="binary")
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_f1 = torchmetrics.classification.F1Score(task="binary")
        self.val_auc = torchmetrics.classification.AUROC(task="binary")
        self.test_acc = torchmetrics.classification.Accuracy(task="binary")
        self.test_f1 = torchmetrics.classification.F1Score(task="binary")
        self.test_auc = torchmetrics.classification.AUROC(task="binary")

        self.enable_dp = enable_dp
        if self.enable_dp:
            self.privacy_engine = PrivacyEngine()
            self.delta = delta
            self.noise_multiplier = noise_multiplier
            self.max_grad_norm = max_grad_norm

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch

        prediction = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

        if (
            len(y) == 1
        ):  # Opacus changes the model so that the prediction has size [] instead of [1]
            prediction = prediction.unsqueeze(0)

        loss = nn.functional.binary_cross_entropy(prediction.float(), y.float())
        self.log("train_loss", loss)
        self.train_acc(prediction, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        self.train_f1(prediction, y)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)
        self.train_auc(prediction, y)
        self.log("train_auc", self.train_auc, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch

        prediction = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

        prediction = nn.functional.sigmoid(prediction).squeeze()

        if (
            len(y) == 1
        ):  # Opacus changes the model so that the prediction has size [] instead of [1]
            prediction = prediction.unsqueeze(0)

        loss = nn.functional.binary_cross_entropy(prediction.float(), y.float())
        self.log("val_loss", loss)
        self.val_acc(prediction, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_f1(prediction, y)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        self.val_auc(prediction, y)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch

        prediction = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

        loss = nn.functional.binary_cross_entropy(prediction.float(), y.float())
        self.log("test_loss", loss)
        self.test_acc(prediction, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_f1(prediction, y)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.test_auc(prediction, y)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True)

        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, features = self.model(x)
        return prediction

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        prediction, features = self.model(x)
        return prediction, features

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        if (
            self.enable_dp
        ):  # https://github.com/pytorch/opacus/blob/main/examples/mnist_lightning.py
            self.trainer.fit_loop.setup_data()
            data_loader = self.trainer.train_dataloader

            if hasattr(self, "dp"):
                self.dp["model"].remove_hooks()
                dp_model, optimizer, data_loader = self.privacy_engine.make_private(
                    module=self,
                    optimizer=optimizer,
                    data_loader=data_loader,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                    poisson_sampling=isinstance(data_loader, DPDataLoader),
                )
                self.dp = {"model": dp_model}

        return optimizer

    def on_train_epoch_end(self) -> None:
        if self.enable_dp:
            epsilon = self.privacy_engine.get_epsilon(self.delta)
            print("EPSILON", epsilon)
            self.log("epsilon", epsilon, on_epoch=True)
