import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.densenet = torchvision.models.densenet121(weights=weights)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Linear(1024, 1)

    def forward(self, x):
        z = self.densenet(x)
        z = self.global_pool(z).squeeze()
        z = self.fully_connected(z)
        return z

    def forward_with_features(self, x):
        z = self.densenet(x)
        features = self.global_pool(z).squeeze()
        z = self.fully_connected(features)
        return z, features
