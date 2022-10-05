import torch
from torch import nn
import torch.nn.functional as F
import math

class SineEncoding(nn.Module):
    def __init__(self, num_pos_feats=32, temperature=10000, normalize=False):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x, mask):
        not_mask = ~mask
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        eps = 1e-6
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            out = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return out