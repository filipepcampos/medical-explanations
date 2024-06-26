import argparse

import torchvision
import lightning as L
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers

from data.mimic_cxr_jpg import MIMICCXRDataModule
from data.chexpert import ChexpertDataModule
from data.brax import BraxDataModule

from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier

L.pytorch.seed_everything(42, workers=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dataset", type=str, default="mimic-cxr")


def get_datamodule(dataset: str, batch_size: int):
    if dataset == "mimic-cxr":
        return MIMICCXRDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
            split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
            batch_size=batch_size,
        )
    if dataset == "chexpert":
        return ChexpertDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/CheXpert-small",
            batch_size=batch_size,
        )
    if dataset == "brax":
        return BraxDataModule(
            root="/nas-ctm01/datasets/public/MEDICAL/BRAX/physionet.org",
            batch_size=batch_size,
        )
    raise ValueError(f"Unknown dataset: {dataset}")

def train(model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        prediction = self.model(x)
        prediction = nn.functional.sigmoid(prediction).squeeze()
        loss = nn.functional.binary_cross_entropy(prediction.float(), y.float())
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            output = nn.functional.sigmoid(prediction).squeeze()
            test_loss += nn.functional.binary_cross_entropy(output.float(), y.float()).item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)

def main():
    args = parser.parse_args()

    model = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    datamodule = get_datamodule(args.dataset, args.batch_size)
    privacy_engine = PrivacyEngine()
   
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=datamodule.train_dataloader(),
        noise_multiplier=1.3,
        max_grad_norm=1.0
    )

    for epoch in range(1, 11):
        train(model,  device, train_loader, optimizer, privacy_engine, epoch)
    test_results = test(model, device, test_loader)
    print(test_results)


if __name__ == "__main__":
    main()
