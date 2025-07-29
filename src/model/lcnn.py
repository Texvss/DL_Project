import torch
from torch import nn

class LCNNModel4(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels_list: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        pool_kernel: int,
        pool_stride: int,
        dropout: float,
        fc_size: int,
        n_classes: int
    ):
        super().__init__()
        
        assert len(channels_list) == len(kernel_sizes) == len(strides), \
            "channels_list, kernel_sizes и strides должны быть одной длины"
        
        blocks = []
        in_ch = input_channels
        for out_ch, k, s in zip(channels_list, kernel_sizes, strides):
            pad = k // 2
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad))
            blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
            blocks.append(nn.Dropout2d(p=dropout))
            in_ch = out_ch
        
        self.feature_extractor = nn.Sequential(*blocks)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten     = nn.Flatten()
        
        self.fc1 = nn.Linear(in_ch, fc_size)
        self.bn1 = nn.BatchNorm1d(fc_size)
        self.fc2 = nn.Linear(fc_size // 2, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        a, b = x.chunk(2, dim=1)
        x = torch.max(a, b)
        
        logits = self.fc2(x)
        return logits
