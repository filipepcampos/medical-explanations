import lightning as L
import torchvision
import torch
import matplotlib.pyplot as plt

from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
from data.mimic_cxr_jpg import MIMICCXRDataModule, SyntheticMIMICCXRDataModule
from data.brax import BraxDataModule
from data.chexpert import ChexpertDataModule

L.pytorch.seed_everything(42, workers=True)

torch.multiprocessing.set_sharing_strategy("file_system")
densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)

BATCH_SIZE = 8

datamodule = MIMICCXRDataModule(
    root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
    split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
    batch_size=BATCH_SIZE,
)

retrieval_datamodules = [
    SyntheticMIMICCXRDataModule(
        root="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/dataset/synthetic_250",
        batch_size=BATCH_SIZE,
    ),
    MIMICCXRDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
        split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
        batch_size=BATCH_SIZE,
    ),
    ChexpertDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/CheXpert-small",
        batch_size=BATCH_SIZE,
    ),
    BraxDataModule(
        root="/nas-ctm01/datasets/public/MEDICAL/BRAX/physionet.org",
        batch_size=BATCH_SIZE,
    ),
]

module = ExplainableClassifier(model=densenet)
module.load_state_dict(torch.load("fl_normal_latest.pth"))
module.eval()

test_x, _ = next(iter(datamodule.test_dataloader()))
test_x = test_x.to(module.device)
# masker_blur = shap.maskers.Image("blur(128,128)", test_x[0].shape)


def f(x):
    x = torch.tensor(x, device=module.device)
    y, feat = module.forward_features(x)
    return y


# explainer = shap.Explainer(f, masker_blur)

storage = []
image_storage = []
label_storage = []

i = 0
for dm in retrieval_datamodules:
    for x, y in dm.train_dataloader():
        x = x.to(module.device)
        pred, feat = module.forward_features(x)
        storage.append(feat.detach().cpu())

        image_storage.append(torch.arange(BATCH_SIZE, i + BATCH_SIZE))
        i += BATCH_SIZE
        label_storage.append(y.detach().cpu())

storage = torch.cat(storage, dim=0)
image_storage = torch.cat(image_storage, dim=0)
label_storage = torch.cat(label_storage, dim=0)


x, y = next(iter(datamodule.test_dataloader()))
x = x.to(module.device)
pred, feat = module.forward_features(x)

# Lookup closest feature vector
distances = torch.cdist(feat.cpu(), storage)
_, indices = torch.topk(distances, k=5, largest=False)


def prepare_image(x):
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )

    x = inv_normalize(x).numpy()

    x = x.transpose(1, 2, 0)
    return x


print(indices)
print("Done?")

# Show closest images
for i in range(x.shape[0]):
    # Show original image and closest images side by side
    fig, axs = plt.subplots(1, 6, figsize=(20, 20))
    axs[0].imshow(prepare_image(x[i].detach().cpu()))
    axs[0].axis("off")
    axs[0].set_title("Original")
    for j, idx in enumerate(indices[i]):
        axs[j + 1].imshow(prepare_image(image_storage[idx]))
        axs[j + 1].axis("off")
        axs[j + 1].set_title(f"Closest {j+1}")
        print(f"{i}-{j} {label_storage[idx]}")

    plt.show()
    plt.savefig(f"closest_{i}.png")
