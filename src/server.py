import flwr as fl


def main() -> None:
    # Define strategy
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

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=dp_strategy,
    )


if __name__ == "__main__":
    main()