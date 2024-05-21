import os
import lightning as L
import torchvision
from mimic_cxr_jpg_loader.modifiers import *
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim

from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier
from data.mimic_datamodule import MIMICCXRDataModule, SyntheticMIMICCXRDataModule

L.pytorch.seed_everything(42, workers=True)
torch.multiprocessing.set_sharing_strategy("file_system")
densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)

module = ExplainableClassifier.load_from_checkpoint(
    "explanations/tz9u4b0a/checkpoints/best_model.ckpt",
    model=densenet,
)
module.eval()

transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

catalogues_dir = "/nas-ctm01/homes/fpcampos/catalogues/output"

for catalogue_dir in os.listdir(catalogues_dir):
    catalogue_path = os.path.join(catalogues_dir, catalogue_dir)
    print(catalogue_path)
    
    # Get the test image
    image_paths = [os.path.join(catalogue_path, f) for f in os.listdir(catalogue_path) if f.endswith(".jpg") and not f.startswith("annotated")]
    test_image = transform(Image.open(image_paths[0]).convert("RGB"))

    # Get the catalogue images
    image_paths = [os.path.join(catalogue_path + "/catalogue", f) for f in os.listdir(catalogue_path + "/catalogue")]
    image_paths = sorted(image_paths, key=lambda x: int(x.split("/")[-1].split("-")[0]))

    catalogue_images = [transform(Image.open(f).convert("RGB")) for f in image_paths]

    # Get the feature vectors
    storage = []
    for x in catalogue_images:
        x = x.unsqueeze(0).to(module.device)
        pred, feat = module.model(x)
        storage.append(feat.detach().cpu())
    storage = torch.stack(storage, dim=0)

    # Get the feature vector of the test image
    x = test_image.unsqueeze(0).to(module.device)
    pred, feat = module.model(x)

    # Lookup closest feature vector
    distances = torch.cdist(feat.cpu().unsqueeze(0), storage)
    _, indices = torch.topk(distances, k=10, largest=False)

    #print("Closest images:", indices)

    # Find closest images based on ssim
    ssim_scores = []

    ma = test_image[0].cpu().numpy().max()
    mi = test_image[0].cpu().numpy().min()

    for i in range(len(catalogue_images)):
        ssim_scores.append(ssim(test_image[0].cpu().numpy(), catalogue_images[i][0].cpu().numpy(), data_range=ma-mi))
    ssim_scores = np.array(ssim_scores)
    ssim_indices = np.argsort(ssim_scores)[::-1]

    print("Closest images based on SSIM:", ssim_indices)
    





# test_x, _ = next(iter(datamodule.test_dataloader()))
# test_x = test_x.to(module.device)
# masker_blur = shap.maskers.Image("blur(128,128)", test_x[0].shape)



# def f(x):
#     x = torch.tensor(x, device=module.device)
#     y, feat = module.model(x)
#     return y
# explainer = shap.Explainer(f, masker_blur)


# storage = []
# image_storage = []
# label_storage = []

# for x, y in synthetic_datamodule.train_dataloader():
#     x = x.to(module.device)
#     pred, feat = module.model(x)
#     storage.append(feat.detach().cpu())
#     image_storage.append(x.detach().cpu())
#     label_storage.append(y.detach().cpu())

# storage = torch.cat(storage, dim=0)
# image_storage = torch.cat(image_storage, dim=0)
# label_storage = torch.cat(label_storage, dim=0)


# x, y = next(iter(datamodule.test_dataloader()))
# x = x.to(module.device)
# pred, feat = module.model(x)

# # Lookup closest feature vector
# distances = torch.cdist(feat.cpu(), storage)
# _, indices = torch.topk(distances, k=5, largest=False)

# def prepare_image(x):
#     inv_normalize = torchvision.transforms.Normalize(
#         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#         std=[1/0.229, 1/0.224, 1/0.255]
#     )
    
#     x = inv_normalize(x).numpy()

#     x = x.transpose(1, 2, 0)
#     return x

# # Show closest images
# for i in range(x.shape[0]):
#     # Show original image and closest images side by side
#     fig, axs = plt.subplots(1, 6, figsize=(20, 20))
#     axs[0].imshow(prepare_image(x[i].detach().cpu()))
#     axs[0].axis("off")
#     axs[0].set_title("Original")
#     print(f"{i}-real={y[i]}, pred={pred[i]}")
#     for j, idx in enumerate(indices[i]):
#         axs[j+1].imshow(prepare_image(image_storage[idx]))
#         axs[j+1].axis("off")
#         axs[j+1].set_title(f"Closest {j+1}")
#         print(f"{i}-{j} {label_storage[idx]}")
    
#     plt.show()
#     plt.savefig(f"closest_{i}.png")
