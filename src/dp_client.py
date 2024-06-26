import argparse
import warnings
from collections import OrderedDict

from flwr_datasets import FederatedDataset
from flwr.client import NumPyClient, ClientApp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import torchvision
from opacus.validators import ModuleValidator
from models.densenet import DenseNet121
from opacus import PrivacyEngine

from data.brax import BraxDataModule
from data.mimic_cxr_jpg import MIMICCXRDataModule
from data.chexpert import ChexpertDataModule

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, train_loader, privacy_engine, optimizer, target_delta, epochs=1):
    criterion = torch.nn.BCELoss()
    for _ in range(epochs):
        for (x, y) in tqdm(train_loader, "Training"):
            optimizer.zero_grad()
            criterion(net(x.to(DEVICE)), y.to(DEVICE)).backward() # TODO: Check sigmoid here
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return epsilon


def test(net, test_loader):
    criterion = torch.nn.BCELoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for (x, y) in tqdm(test_loader, "Testing"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = net(images) # TODO: Check sigmoid here
            loss += criterion(x, y).item()
            correct += (torch.max(outputs.data, 1)[1] == y).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return loss, accuracy

def load_data(partition_id: int, batch_size: int, split: str):
    if partition_id == 0: 
        dataloader = MIMICCXRDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
            split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
            batch_size=batch_size,
        )
    elif partition_id == 1: 
        dataloader = ChexpertDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/CheXpert-small",
            batch_size=batch_size,
        )
    elif partition_id == 2: 
        dataloader = BraxDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/BRAX/physionet.org",
            batch_size=batch_size,
        )
    else: 
        raise ValueError(f"Invalid partition_id {partition_id}")
    
    return dataloader.train_dataloader(), dataloader.val_dataloader()


class FlowerClient(NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        target_delta,
        noise_multiplier,
        max_grad_norm,
    ) -> None:
        super().__init__()
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        self.privacy_engine = PrivacyEngine(secure_mode=False)
        self.target_delta = target_delta
        (
            self.model,
            self.optimizer,
            self.train_loader,
        ) = self.privacy_engine.make_private(
            module=model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epsilon = train(
            self.model,
            self.train_loader,
            self.privacy_engine,
            self.optimizer,
            self.target_delta,
        )

        if epsilon is not None:
            print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
        else:
            print("Epsilon value not available.")
        return (self.get_parameters(config={}), len(self.train_loader), {})

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def client_fn_parameterized(
    partition_id, target_delta=1e-5, noise_multiplier=1.3, max_grad_norm=1.0
):
    def client_fn(partition_id: str):
        net = DenseNet121(weights=None).to(DEVICE)
        net = ModuleValidator.fix(net)
        train_loader, test_loader = load_data(partition_id=partition_id)
        return FlowerClient(
            net,
            train_loader,
            test_loader,
            target_delta,
            noise_multiplier,
            max_grad_norm,
        ).to_client()

    return client_fn


appA = ClientApp(
    client_fn=client_fn_parameterized(partition_id=0, noise_multiplier=1.3),
)

appB = ClientApp(
    client_fn=client_fn_parameterized(partition_id=1, noise_multiplier=1.3),
)

appC = ClientApp(
    client_fn=client_fn_parameterized(partition_id=2, noise_multiplier=1.3)
)
