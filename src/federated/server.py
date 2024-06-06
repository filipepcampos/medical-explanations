import flwr as fl
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


class CustomStrategy(fl.server.strategy.FedAvg):
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

        # metrics = {
        #     "acc": 0,
        #     "f1": 0,
        # }

        # for metric in metrics:
        #     vals = [r.metrics.get(metric, 0) for _, r in results]
        #     examples = [r.num_examples for _, r in results]
        #     metrics[metric] = sum(vals) / sum(examples)

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
            torch.save(model.state_dict(), "lastest_fl_model.pth")

        return aggregated_parameters, aggregated_metrics


strategy = fl.server.strategy.DifferentialPrivacyServerSideAdaptiveClipping(
    strategy=CustomStrategy(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
    ),
    noise_multiplier=0.1,
    initial_clipping_norm=1.0,
    num_sampled_clients=3,
)

N_CLIENTS = 3


def main():
    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
