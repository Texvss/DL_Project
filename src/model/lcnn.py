# src/model/lcnn.py
import torch
from torch import nn

class MFM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        a, b = torch.split(x, self.out_channels, dim=1)
        return torch.maximum(a, b)

class LCNN(nn.Module):
    TARGET_T = 600

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.layer1 = nn.Sequential(
            MFM(1, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            MFM(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            MFM(32, 48, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48)
        )
        self.layer3 = nn.Sequential(
            MFM(48, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            MFM(48, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            MFM(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            MFM(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            MFM(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            MFM(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # fully connected
        self.fc1 = nn.Linear(32, 160)
        self.bn_fc1 = nn.BatchNorm1d(160)
        self.bn_fc2 = nn.BatchNorm1d(80)
        self.dropout = nn.Dropout(0.75)
        self.fc2 = nn.Linear(80, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        if T != self.TARGET_T:
            if T < self.TARGET_T:
                x = nn.functional.pad(x, (0, self.TARGET_T - T))
            else:
                x = x[:, :, :, :self.TARGET_T]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        a, b = x.chunk(2, dim=1)
        x = torch.maximum(a, b)
        x = self.bn_fc2(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
