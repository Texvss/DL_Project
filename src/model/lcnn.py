import torch
from torch import nn

class LCNNModel4(nn.Module):

    def __init__(
            self,
            input_channels: int,
            channels_list: list[int],
            kernel_sizes: list[int],
            steps: list[int],
            kernel_pool: int,
            step_pool: int,
            dropout: float,
            FLayer_size: int,
            n_classes: int
            ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        inp_channels = input_channels
        for i in range(len(channels_list)):
            output_channels = channels_list[i]
            kernel_size = kernel_sizes[i]
            step = steps[i]
            pad = kernel_size // 2
            
            self.convs.append(nn.Conv2d(inp_channels, output_channels, kernel_size=kernel_size, stride=step, padding=pad))
            self.pools.append(nn.MaxPool2d(kernel_size=kernel_pool, stride=step_pool))
            self.dropouts.append(nn.Dropout2d(p=dropout))
            
            inp_channels = output_channels // 2

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.Layer1 = nn.Linear(inp_channels, FLayer_size)
        self.Layer2 = nn.Linear(FLayer_size // 2, n_classes)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, pool, dropout in zip(self.convs, self.pools, self.dropouts):
            x = conv(x)

            a, b = x.chunk(chunks=2, dim=1)
            x = torch.max(a, b)
            x = pool(x)
            x = dropout(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.Layer1(x)
        a, b = x.chunk(chunks=2, dim=1)
        x = torch.max(a, b)
        logits = self.Layer2(x)
        return logits