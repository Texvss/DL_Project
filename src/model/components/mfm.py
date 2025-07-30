# import torch
# from torch import nn

# class MFM(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()
#         self.out_channels = out_channels
#         self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)

#     def forward(self, x):
#         x = self.conv(x)
#         out = torch.split(x, self.out_channels, dim=1)
#         return torch.max(out[0], out[1])