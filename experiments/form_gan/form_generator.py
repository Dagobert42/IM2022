from torch import nn
from experiments.form_gan.utils import *

class FormGenerator(nn.Module):
    def __init__(self, noise_dim, transform_dim, cube_side_len):
        super(FormGenerator, self).__init__()
        self.cube_side_len = cube_side_len
        self.mlp = MLP(noise_dim, 1024, (self.initial_size ** 2) * self.dim, 1)

        self.pos_encoding1 = SineEncoding(transform_dim // 2, normalize=True)
        self.transform1 = nn.Transformer(transform_dim)

        self.pos_encoding2 = SineEncoding(transform_dim // 2, normalize=True)
        self.transform2 = nn.Transformer(transform_dim)

        # self.reverse_backbone
        self.unflattening = nn.Sequential(nn.Conv3d(transform_dim, 1, 1))
        
    def forward(self, noise):
        out = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)

        out = self.linear(out.permute(0, 2, 1).view(
            -1,
            self.cube_side_len,
            self.cube_side_len,
            self.cube_side_len))

        return 