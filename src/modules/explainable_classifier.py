from typing import Any
import lightning as L
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class ExplainableClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.train_f1 = torchmetrics.classification.F1Score(task="binary")
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_f1 = torchmetrics.classification.F1Score(task="binary")
        self.test_acc = torchmetrics.classification.Accuracy(task="binary")
        self.test_f1 = torchmetrics.classification.F1Score(task="binary")

    def training_step(self, batch, batch_idx):
        x, y = batch

        prediction = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

        loss = nn.functional.binary_cross_entropy(prediction.float(), y.float()) # TODO: .float were added only for synth training
        self.log("train_loss", loss)
        self.train_acc(prediction, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        self.train_f1(prediction, y)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prediction = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

        loss = nn.functional.binary_cross_entropy(prediction.float(), y.float())
        self.log("val_loss", loss)
        self.val_acc(prediction, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_f1(prediction, y)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        print(self.model(x))
        prediction = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()

        loss = nn.functional.binary_cross_entropy(prediction.float(), y.float())
        self.log("test_loss", loss)
        self.val_acc(prediction, y)
        self.log("test_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_f1(prediction, y)
        self.log("test_f1", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def forward(self, x):
        print(self.model(x))
        prediction, features = self.model(x)
        return prediction
    
    def forward_features(self, x):
        prediction, features = self.model(x)
        return prediction, features

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer