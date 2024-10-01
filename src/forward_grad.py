import lightning as L
import torchvision
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
from data.mimic_datamodule import MIMICCXRDataModule
import yaml

L.pytorch.seed_everything(42, workers=True)
torch.multiprocessing.set_sharing_strategy("file_system")

with open("config.yaml") as f:
    config = yaml.safe_load(f)


def prepare_image(x):
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )

    x = inv_normalize(x).numpy()

    x = x.transpose(1, 2, 0)
    return x


def normalize_image(x):
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )

    x = inv_normalize(x).numpy()

    # Normalize to [0, 1]
    x = (x - x.min()) / (x.max() - x.min())
    x = x.transpose(1, 2, 0)

    return x


densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
module = ExplainableClassifier.load_from_checkpoint(
    config["explainable_classifier"],
    model=densenet,
)
module.eval()

datamodule = MIMICCXRDataModule(
    root=config["mimic_path"],
    split_path=config["mimic_splits_path"],
    batch_size=8,
)

x, y = next(iter(datamodule.test_dataloader()))
x = x.to(module.device)

target_layers = [densenet.densenet[-1].norm5]
cam = AblationCAM(model=densenet, target_layers=target_layers)
grayscale_cam = cam(input_tensor=x, targets=None, aug_smooth=True, eigen_smooth=True)

for i in range(8):
    grayscale_cam_im = grayscale_cam[i, :]
    visualization = show_cam_on_image(
        normalize_image(x[i].cpu()),
        grayscale_cam_im,
        use_rgb=True,
    )
    plt.imshow(visualization)
    plt.savefig(f"ablationcam_{i}.png")
