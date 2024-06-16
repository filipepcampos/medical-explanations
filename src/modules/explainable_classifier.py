import lightning as L
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim


class ExplainableClassifier(L.LightningModule):
    def __init__(self, model: nn.Module):
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch

        prediction = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

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
        return optimizer
