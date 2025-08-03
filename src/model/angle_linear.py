import torch
import torch.nn as nn
import torch.nn.functional as F

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, s=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.m = m
        self.s = s

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cos_t = x_norm @ w_norm.t()
        cos_t = cos_t.clamp(-1 + 1e-7, 1 - 1e-7)

        if labels is not None:
            theta = torch.acos(cos_t)
            phi = torch.cos(self.m * theta)
            one_hot = torch.zeros_like(cos_t)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            logits = self.s * (one_hot * phi + (1.0 - one_hot) * cos_t)
        else:
            logits = self.s * cos_t
        return logits