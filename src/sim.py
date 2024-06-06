import flwr as fl
from client import FlowerClient
from data.mimic_datamodule import MIMICCXRDataModule
from data.chexpert import ChexpertDataModule
from data.brax import BraxDataModule
from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
import torchvision

import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar
from collections import OrderedDict
import torch

densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
model = ExplainableClassifier(densenet)


def create_client(datamodule) -> FlowerClient:
    train_loader, val_loader, test_loader = (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    )
    return FlowerClient(model, train_loader, val_loader, test_loader).to_client()


def get_mimic_client() -> FlowerClient:
    datamodule = MIMICCXRDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
        split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
        batch_size=32,
    )
    return create_client(datamodule)


def get_chexpert_client() -> FlowerClient:
    datamodule = ChexpertDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/CheXpert-small",
        batch_size=32,
    )
    return create_client(datamodule)


def get_brax_client() -> FlowerClient:
    datamodule = BraxDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/BRAX/physionet.org",
        batch_size=32,
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


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters,
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(model.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=0.5,
)


dp_strategy = fl.server.strategy.DifferentialPrivacyServerSideAdaptiveClipping(
    strategy=strategy,
    noise_multiplier=0.1,
    initial_clipping_norm=1.0,
    num_sampled_clients=3,
)

N_CLIENTS = 3

client_resources = {"num_cpus": 1, "num_gpus": 1 / N_CLIENTS}


client = fl.client.ClientApp(
    client_fn=client_fn,
)

server = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=dp_strategy,
)


def main():

    # Launch the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # A function to run a _virtual_ client when required
        num_clients=N_CLIENTS,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=3),  # Specify number of FL rounds
        strategy=dp_strategy,  # A Flower strategy
    )

    print(history)


if __name__ == "__main__":
    main()
