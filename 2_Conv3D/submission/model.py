import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.layer1 = nn.Conv3d(1, 4, stride=(1, 2, 2), kernel_size=5)
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3))
        #self.layer2 = nn.Conv3d(4, 4, stride=(1, 2, 2), kernel_size=5)
        #self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.flatten = nn.Flatten()
        self.layer3 = nn.Linear(3200, 24 * 64 * 64)

    def forward(self, x):
        x = x.unsqueeze(dim=1) / 1023.0
        x = torch.relu(self.down_pool(x))
        x = self.layer1(x)
        x = torch.relu(self.pool(x))
        x = self.flatten(x)
        x = torch.relu(self.layer3(x))

        return x.view(-1, 24, 64, 64) * 1023.0