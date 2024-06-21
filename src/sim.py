import flwr as fl
from typing import List, Tuple, Union, Optional, Dict
from collections import OrderedDict
import numpy as np
from federated.client import FlowerClient
from helper.lightning import DPLightningDataModule
from data.mimic_cxr_jpg import MIMICCXRDataModule
from data.chexpert import ChexpertDataModule
from data.brax import BraxDataModule
from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
import torchvision

import torch

N_CLIENTS = 3
N_ROUNDS = 10

densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
model = ExplainableClassifier(densenet)


def create_client(datamodule, enable_dp: bool = False) -> FlowerClient:
    x, y = next(iter(datamodule.train_dataloader()))
    assert torch.min(x) >= -1 and torch.max(x) <= 1
    assert torch.min(y) >= 0 and torch.max(y) <= 1

    if enable_dp:
        datamodule = DPLightningDataModule(datamodule)

    return FlowerClient(model, datamodule).to_client()


def get_mimic_client(enable_dp: bool = False) -> FlowerClient:
    datamodule = MIMICCXRDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
        split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
        batch_size=8,
    )
    return create_client(datamodule, enable_dp=enable_dp)


def get_chexpert_client(enable_dp: bool = False) -> FlowerClient:
    datamodule = ChexpertDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/CheXpert-small",
        batch_size=8,
    )
    return create_client(datamodule, enable_dp=enable_dp)


def get_brax_client(enable_dp: bool = False) -> FlowerClient:
    datamodule = BraxDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/BRAX/physionet.org",
        batch_size=8,
    )
    return create_client(datamodule, enable_dp=enable_dp)


def client_fn(cid: str):
    enable_dp = True  # TODO: Remove debug
    if cid == "0":
        return get_mimic_client(enable_dp)
    elif cid == "1":
        return get_brax_client(enable_dp)
    elif cid == "2":
        return get_chexpert_client(enable_dp)
    raise Exception(f"Unknown client: {cid}")


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[
            Union[
                Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes],
                BaseException,
            ]
        ],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
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
            torch.save(model.state_dict(), "model_fl_latest.pth")

        return aggregated_parameters, aggregated_metrics


def metrics_aggregation_fn(metrics):
    result = {}
    for count, metric_dict in metrics:
        for metric_name, metric_val in metric_dict.items():
            current_count, current_val = result.get(metric_name, (0, 0))
            result[metric_name] = (
                current_count + count,
                current_val + metric_val * count,
            )

    aggregated_metrics = {}
    for metric_name, (count, value) in result.items():
        aggregated_metrics[metric_name] = value / count
    return aggregated_metrics


strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    fit_metrics_aggregation_fn=metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
)


def main():
    client_resources = {"num_cpus": 1, "num_gpus": 0}

    # Launch the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # A function to run a _virtual_ client when required
        num_clients=N_CLIENTS,
        client_resources=client_resources,
        config=fl.server.ServerConfig(
            num_rounds=N_ROUNDS,
        ),  # Specify number of FL rounds
        strategy=strategy,  # A Flower strategy
    )

    print(history)


if __name__ == "__main__":
    main()
