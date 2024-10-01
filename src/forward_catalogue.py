import os
import lightning as L
import torchvision
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import yaml

from models.densenet import DenseNet121
from modules.explainable_classifier import ExplainableClassifier

L.pytorch.seed_everything(42, workers=True)
torch.multiprocessing.set_sharing_strategy("file_system")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

densenet = DenseNet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)

module = ExplainableClassifier.load_from_checkpoint(
    config["explainable_classifier"],
    model=densenet,
)
module.eval()

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ],
)

catalogues_dir = config["catalogues_dir"]

for catalogue_dir in os.listdir(catalogues_dir):
    catalogue_path = os.path.join(catalogues_dir, catalogue_dir)
    print(catalogue_path)

    # Get the test image
    image_paths = [
        os.path.join(catalogue_path, f)
        for f in os.listdir(catalogue_path)
        if f.endswith(".jpg") and not f.startswith("annotated")
    ]
    test_image = transform(Image.open(image_paths[0]).convert("RGB"))

    # Get the catalogue images
    image_paths = [
        os.path.join(catalogue_path + "/catalogue", f)
        for f in os.listdir(catalogue_path + "/catalogue")
    ]
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

    # Find closest images based on ssim
    ssim_scores = []

    ma = test_image[0].cpu().numpy().max()
    mi = test_image[0].cpu().numpy().min()

    for i in range(len(catalogue_images)):
        ssim_scores.append(
            ssim(
                test_image[0].cpu().numpy(),
                catalogue_images[i][0].cpu().numpy(),
                data_range=ma - mi,
            ),
        )

    ssim_indices = np.argsort(np.array(ssim_scores))[::-1]
    print("Closest images based on SSIM:", ssim_indices)
