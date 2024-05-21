import lightning as L
import torchvision
from mimic_cxr_jpg_loader.modifiers import *
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import skimage.metrics
from skimage import io
from skimage.transform import resize

from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
from data.mimic_datamodule import MIMICCXRDataModule, SyntheticMIMICCXRDataModule

L.pytorch.seed_everything(42, workers=True)
torch.multiprocessing.set_sharing_strategy("file_system")
densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)

datamodule = MIMICCXRDataModule(
    root="/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR",
    split_path="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv",
    batch_size=8,
)

synthetic_datamodule = SyntheticMIMICCXRDataModule(
    root="/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/dataset/synthetic_250",
    batch_size=8,
)

module = ExplainableClassifier.load_from_checkpoint(
    "explanations/tz9u4b0a/checkpoints/best_model.ckpt",
    model=densenet,
)
module.eval()

test_x, _ = next(iter(datamodule.test_dataloader()))
test_x = test_x.to(module.device)
masker_blur = shap.maskers.Image("blur(128,128)", test_x[0].shape)



def f(x):
    x = torch.tensor(x, device=module.device)
    y, feat = module.model(x)
    return y
explainer = shap.Explainer(f, masker_blur)

# shap_values_fine = explainer(
#     test_x[1:3], max_evals=300, batch_size=16
# )

# def show_samples(shap_values):
#     inv_normalize = torchvision.transforms.Normalize(
#         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#         std=[1/0.229, 1/0.224, 1/0.255]
#     )
    
#     v = np.array([np.swapaxes(np.swapaxes(s, 1, -1), 1, 2).transpose(1,2,0) for s in shap_values.values])
#     print(v[0].shape, len(v))
#     p = shap_values.data.cpu().numpy()

#     # Undo image normalization by reverting mean and std
#     for i in range(len(p)):
#         p[i] = inv_normalize(torch.tensor(p[i])).numpy()

#     p = p.transpose(0, 2, 3, 1)
    
#     shap.image_plot(
#         shap_values=v,
#         pixel_values=p,
#         labels=shap_values_fine.output_names,
#     )

#     plt.savefig("shap.png")

# show_samples(shap_values_fine)

storage = []
image_storage = []
label_storage = []

for x, y in synthetic_datamodule.train_dataloader():
    x = x.to(module.device)
    pred, feat = module.model(x)
    storage.append(feat.detach().cpu())
    image_storage.append(x.detach().cpu())
    label_storage.append(y.detach().cpu())

storage = torch.cat(storage, dim=0)
image_storage = torch.cat(image_storage, dim=0)
label_storage = torch.cat(label_storage, dim=0)


x, y = next(iter(datamodule.test_dataloader()))
x = x.to(module.device)
pred, feat = module.model(x)

# Lookup closest feature vector
distances = torch.cdist(feat.cpu(), storage)
_, indices = torch.topk(distances, k=5, largest=False)

def prepare_image(x):
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    
    x = inv_normalize(x).numpy()

    x = x.transpose(1, 2, 0)
    return x

# Show closest images
for i in range(x.shape[0]):
    # Show original image and closest images side by side
    fig, axs = plt.subplots(1, 6, figsize=(20, 20))
    axs[0].imshow(prepare_image(x[i].detach().cpu()))
    axs[0].axis("off")
    axs[0].set_title("Original")
    print(f"{i}-real={y[i]}, pred={pred[i]}")
    for j, idx in enumerate(indices[i]):
        axs[j+1].imshow(prepare_image(image_storage[idx]))
        axs[j+1].axis("off")
        axs[j+1].set_title(f"Closest {j+1}")
        print(f"{i}-{j} {label_storage[idx]}")
    
    plt.show()
    plt.savefig(f"closest_{i}.png")

# Show closest images based on SSIM
# def ssim(x, y):
#     print("Ay", x.shape, y.shape)
#     x = x.permute(1, 2, 0).cpu().numpy()
#     y = y.permute(1, 2, 0).cpu().numpy()
#     x = resize(x, (256, 256))
#     y = resize(y, (256, 256))
#     print(x.max(), x.min(), y.max(), y.min())
#     return skimage.metrics.structural_similarity(x, y, multichannel=True, channel_axis=2, data_range=4.757904)

# for i in range(x.shape[0]):
#     # Show original image and closest images side by side
#     fig, axs = plt.subplots(1, 6, figsize=(20, 20))
#     axs[0].imshow(prepare_image(x[i].detach().cpu()))
#     axs[0].axis("off")
#     axs[0].set_title("Original")

#     closest_images = []
#     for j, img in enumerate(image_storage):
#         prepared_xi = prepare_image(x[i].detach().cpu())
#         prepared_img = prepare_image(img)
#         closest_images.append(ssim(prepared_xi, prepared_img))
#     closest_images = np.array(closest_images)
#     closest_images = np.argsort(closest_images)

#     for j, idx in enumerate(closest_images[:5]):
#         axs[j+1].imshow(prepare_image(image_storage[idx]))
#         axs[j+1].axis("off")
#         axs[j+1].set_title(f"Closest {j+1}")
#     plt.show()
#     plt.savefig(f"closest_ssim_{i}.png")
