import argparse
import flwr as fl
import multiprocessing as mp
from dp_helpers import train, test

"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
# import os
# if os.environ.get("https_proxy"):
#     del os.environ["https_proxy"]
# if os.environ.get("http_proxy"):
#     del os.environ["http_proxy"]

def main():
    """Get all args necessary for dp"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-vb", type=int, default=256, help="Virtual batch size")
    parser.add_argument("-b", type=int, default=256, help="Batch size")
    parser.add_argument(
        "-lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "-nm", type=float, default=1.2, help="Noise multiplier for Private Engine."
    )
    parser.add_argument(
        "-mgn", type=float, default=1.0, help="Max grad norm for Private Engine."
    )
    parser.add_argument(
        "-eps",
        type=float,
        default=1.0,
        help="Target epsilon for the privacy budget.",
    )
    parser.add_argument(
        "-cid", type=int, default=0, help="Client id (0, 1 or 2)."
    )
    
    args = parser.parse_args()
    cid = int(args.cid)
    vbatch_size = int(args.vb)
    batch_size = int(args.b)
    lr = float(args.lr)
    nm = float(args.nm)
    mgn = float(args.mgn)
    eps = float(args.eps)
    

    # Flower client
    class DPClient(fl.client.NumPyClient):
        def __init__(
            self, 
            cid: int,
            vbatch_size: int,
            batch_size: int,
            lr: float,
            eps: float,
            nm: float,
            mgn: float,
            ):
            """Differentially private implementation of a Cifar client.

            Parameters
            ----------
            cid: int
                Client id
            vbatch_size : int
                Virtual batch size.
            batch_size : int
                Batch size.
            lr : float
                Learning rate.
            eps : float
                Target epsilon.
            nm : float
                Noise multiplier.
            mgn : float
                Maximum gradient norm.
            """
            self.cid = cid
            self.vbatch_size = vbatch_size
            self.batch_size = batch_size
            self.lr = lr
            self.eps = eps
            self.nm = nm
            self.mgn = mgn
            self.parameters = None
            self.state_dict = None

        def get_parameters(self):
            return self.parameters

        def set_parameters(self, parameters):
            self.parameters = parameters

        def fit(self, parameters, config):
            self.set_parameters(parameters)

            manager = mp.Manager()
            # We receive the results through a shared dictionary
            return_dict = manager.dict()
            # Create the process
            p = mp.Process(
                target=train,
                args=(
                    parameters,
                    return_dict,
                    config,
                    self.cid,
                    self.vbatch_size,
                    self.batch_size,
                    self.lr,
                    self.nm,
                    self.mgn,
                    self.state_dict,
                    )
                )
            # Start the process
            p.start()
            # Wait for it to end
            p.join()
            # Close it
            try:
                p.close()
            except ValueError as e:
                print(f"Couldn't close the training process: {e}")
            # Get the return values
            new_parameters = return_dict["parameters"]
            data_size = return_dict["data_size"]
            # Store updated state dict
            self.state_dict = return_dict["state_dict"]

            # Check if target epsilon value is respected
            accept = True
            # Leave +0.3 margin to accomodate with opacus imprecisions
            if return_dict["eps"] > self.eps + 0.3:
                # refuse the client new parameters
                accept = False
                print(
                    f"Epsilon over target value ({self.eps}), disconnecting client."
                )
                # Override new parameters with previous ones
                new_parameters = parameters
                print()
            # Init metrics dict
            metrics = {
                "epsilon": return_dict["eps"],
                "alpha": return_dict["alpha"],
                "accept": accept,
            }
            # Del everything related to multiprocessing
            del (manager, return_dict, p)
            return new_parameters, data_size, metrics

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            # Prepare multiprocess
            manager = mp.Manager()
            # We receive the results through a shared dictionary
            return_dict = manager.dict()
            # Create the process
            p = mp.Process(target=test, args=(
                parameters,
                return_dict,
                self.cid,
                self.batch_size
                ))
            # Start the process
            p.start()
            # Wait for it to end
            p.join()
            # Close it
            try:
                p.close()
            except ValueError as e:
                print(f"Coudln't close the evaluating process: {e}")
            # Get the return values
            loss = return_dict["loss"]
            accuracy = return_dict["accuracy"]
            data_size = return_dict["data_size"]
            # Del everything related to multiprocessing
            del (manager, return_dict, p)
            return float(loss), data_size, {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client(server_address="[::]:9231", client=DPClient(
        cid,
        vbatch_size,
        batch_size,
        lr,
        eps,
        nm,
        mgn,
        ))


if __name__ == "__main__":
    main()
