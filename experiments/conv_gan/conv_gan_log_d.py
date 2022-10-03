import torch
from torch import nn
from utils import init_weights

def deconv_layer(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_dim,
            out_dim,
            kernel_size=4,
            stride=2,
            bias=False,
            padding=(1,1,1)),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True))

class ConvGenerator(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super(ConvGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim

        self.deconv1 = deconv_layer(self.noise_dim, self.feature_dim*4)
        self.deconv2 = deconv_layer(self.feature_dim*4, self.feature_dim*2)
        self.deconv3 = deconv_layer(self.feature_dim*2, self.feature_dim)
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(
                self.feature_dim,
                1,
                kernel_size=4,
                stride=2,
                bias=False,
                padding=(1,1,1)),
            nn.Sigmoid())
        
        self.apply(init_weights)

    def forward(self, x):
        out = x.view(-1, self.noise_dim, 1, 1, 1)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = torch.squeeze(out)
        return out

def conv_layer(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(
            in_dim,
            out_dim,
            kernel_size=4,
            stride=2,
            bias=False,
            padding=(1,1,1)),

        nn.BatchNorm3d(out_dim),
        nn.Dropout3d(0.5, inplace=True),
        nn.LeakyReLU(0.2, inplace=True))

class ConvDiscriminator(nn.Module):
    def __init__(self, feature_dim, output_shape):
        super(ConvDiscriminator, self).__init__()
        self.feature_dim = feature_dim
        self.output_shape = output_shape

        self.conv1 = conv_layer(1, self.feature_dim)
        self.conv2 = conv_layer(self.feature_dim, self.feature_dim*2)
        self.conv3 = conv_layer(self.feature_dim*2, self.feature_dim*4)

        self.conv4 = nn.Sequential(
            nn.Conv3d(
                self.feature_dim*4,
                1,
                kernel_size=4,
                stride=2,
                bias=False,
                padding=(1,1,1)),
            nn.Sigmoid()
        )

        self.apply(init_weights)

    def forward(self, x):
        out = x.view(-1, 1, self.output_shape, self.output_shape, self.output_shape)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = torch.squeeze(out)
        return out
