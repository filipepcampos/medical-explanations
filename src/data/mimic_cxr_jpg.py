import numpy as np
import torchvision
from mimic_cxr_jpg_loader.dataset import MIMICDataset
from torch.utils.data import Dataset


class MIMIC_CXR_JPG(Dataset):
    """
    Wrapper class for the MIMIC-CXR-JPG dataset.
    """

    def __init__(
        self, root: str, split_path: str, modifiers=None, transform=None
    ):  # TODO: Add annotation
        self.root = root

        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        self.dataset = MIMICDataset(
            root=root, split_path=split_path, modifiers=modifiers
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, datum = self.dataset[idx]
        return self.transform(img), datum["Cardiomegaly"].astype(
            np.float32
        )  # TODO: Move Cardiomegaly to config
