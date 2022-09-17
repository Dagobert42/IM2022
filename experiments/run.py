import torch
from models.conv_gan import *
from train import *

# define model parameters
output_shape = 24
noise_dim = 200
feature_dim = 32

Generator=ConvGenerator(noise_dim, feature_dim)
Discriminator=ConvDiscriminator(feature_dim, output_shape)

# D_PATH = '/models/states/ConvGAN16-09-2022-19-36-49_D.pth'
# G_PATH = '/models/states/ConvGAN16-09-2022-19-36-49_G.pth'
# Discriminator.load_state_dict(torch.load(D_PATH, map_location=torch.device("cpu")))
# Generator.load_state_dict(torch.load(G_PATH, map_location=torch.device("cpu")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fit(
    Discriminator,
    Generator,
    name="ConvGAN",
    data_path="segmentation_3d_data.pkl",
    generator_lr=0.00001,
    discriminator_lr=0.0025,
    n_epochs=100,
    batch_size=32,
    noise_dim=noise_dim,
    target_accuracy=0.8,
    beta=(0.5, 0.999),
    live_mode=False,
    device=device)