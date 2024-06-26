import flwr as fl
from collections import OrderedDict
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar, FitRes
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from models.densenet import DenseNet121
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
from data.brax import BraxDataModule
from data.mimic_cxr_jpg import MIMICCXRDataModule
from data.chexpert import ChexpertDataModule

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_dataloader(cid: int, batch_size: int, split: str):
    if cid == 0: 
        dataloader = MIMICCXRDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
            split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
            batch_size=batch_size,
        )
    elif cid == 1: 
        dataloader = ChexpertDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/CheXpert-small",
            batch_size=batch_size,
        )
    elif cid == 2: 
        dataloader = BraxDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/BRAX/physionet.org",
            batch_size=batch_size,
        )
    else: 
        raise ValueError(f"Invalid cid {cid}")

    if split == "train":
        return dataloader.train_dataloader()
    elif split == "test":
        return dataloader.test_dataloader()
    elif split == "val":
        return dataloader.val_dataloader()
    raise ValueError(f"Invalid split {split}")

def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_target_delta(data_size: int) -> float:
    """Generate target delta given the size of a dataset. Delta should be
    less than the inverse of the datasize.

    Parameters
    ----------
    data_size : int
        The size of the dataset.

    Returns
    -------
    float
        The target delta value.
    """
    den = 1
    while data_size // den >= 1:
        den *= 10
    return 1 / den

def train(
    parameters,
    return_dict,
    config,
    cid,
    vbatch_size,
    batch_size,
    lr,
    nm,
    mgn,
    state_dict,
    ):
    """Train the network on the training set."""
    train_loss = 0.0
    train_acc = 0.0
    # Define the number of cumulative steps
    assert(vbatch_size%batch_size==0)
    n_acc_steps = int(vbatch_size / batch_size)

    train_loader = get_dataloader(cid, batch_size, "train")
    len_dataset = len(train_loader.dataset)
    
    net = DenseNet121(weights=None)
    net = ModuleValidator.fix(net)
    net.to(DEVICE)

    if parameters is not None:
        set_weights(net, parameters)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    net.train()

    # Get orders for RDP
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    # Get delta
    delta = get_target_delta(len_dataset)
    # Set the sample rate
    sample_rate = batch_size / len_dataset

    privacy_engine = PrivacyEngine(
        net,
        sample_rate=sample_rate*n_acc_steps,
        alphas=alphas,
        noise_multiplier=nm,
        max_grad_norm=mgn,
        target_delta=delta,
    )
    # Load the state_dict if not None
    if state_dict is not None:
        privacy_engine.load_state_dict(state_dict)
    privacy_engine.to(DEVICE)
    # Attach PrivacyEngine after moving it to the same device as the model
    privacy_engine.attach(optimizer)
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = net(images)
        out = nn.functional.sigmoid(out).squeeze()
        loss = criterion(out, labels)
        
        # Get preds
        pred_ids = out

        # Compute accuracy
        acc = (pred_ids == labels).sum().item() / batch_size
        loss.backward()
        # Take a real optimizer step after n_virtual_steps
        if ((i + 1) % n_acc_steps == 0) or ((i + 1) == len(train_loader)):
            optimizer.step()  # real step
            optimizer.zero_grad()
        else:
            optimizer.virtual_step()  # take a virtual step
        # Detach loss to compute total loss
        train_loss += (loss.detach().item() - train_loss) / (i + 1)
        train_acc += (acc - train_acc) / (i + 1)
    else:
        print(
            f"Round Results:",
            f"Train Loss: {train_loss}",
            f"Train Accuracy: {train_acc}",
        )
        # print best alpha and epsilon
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print(f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
        # Prepare return values
        return_dict["eps"] = epsilon
        return_dict["alpha"] = best_alpha
        return_dict["parameters"] = get_weights(net)
        return_dict["data_size"] = len(train_loader)
        return_dict["state_dict"] = privacy_engine.state_dict()


def test(parameters, return_dict, cid, batch_size):
    """Validate the network on the entire test set."""
    test_loss = 0.0
    test_acc = 0.0

    test_loader = get_dataloader(cid, batch_size, "test")
    len_dataset = len(test_loader.dataset)
    
    net = DenseNet121(weights=None)
    net = ModuleValidator.fix(net)
    net.to(DEVICE)

    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        net.eval()
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = net(images)
            out = nn.functional.sigmoid(out).squeeze()
            loss = criterion(out, labels)
            pred_ids = out
            acc = (pred_ids == labels).sum().item() / batch_size
            test_loss += (loss.detach().item() - test_loss) / (i + 1)
            test_acc += (acc - test_acc) / (i + 1)
    print(
        f"Test Loss: {test_loss}",
        f"Test Accuracy: {test_acc}",
    )
    # Prepare return values
    return_dict["loss"] = test_loss
    return_dict["accuracy"] = test_acc
    return_dict["data_size"] = len(test_loader)



class FedAvgDp(FedAvg):
    """This class implements the FedAvg strategy for Differential Privacy context."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        FedAvg.__init__(
            self,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        # Keep track of the maximum possible privacy budget
        self.max_epsilon = 0.0

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        """Get the privacy budget"""
        if not results:
            return None
        # Get the privacy budget of each client
        accepted_results = []
        disconnect_clients = []
        epsilons = []
        for c, r in results:
            # Check if client can be accepted or not
            if r.metrics["accept"]:
                accepted_results.append([c, r])
                epsilons.append(r.metrics["epsilon"])
            else:
                disconnect_clients.append(c)
        results = accepted_results
        if epsilons:
            self.max_epsilon = max(self.max_epsilon, max(epsilons))
        print(f"Privacy budget ε at round {rnd}: {self.max_epsilon}")
        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_fit(rnd, results, failures)

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Configure the next round of evaluation. Returns None since evaluation is made server side.
        You could comment this method if you want to keep the same behaviour as FedAvg."""
        if client_manager.num_available() < self.min_fit_clients:
            print(
                f"{client_manager.num_available()} client(s) available(s), waiting for {self.min_available_clients} availables to continue."
            )
        # rnd -1 is a special round for last evaluation when all rounds are over
        return None
