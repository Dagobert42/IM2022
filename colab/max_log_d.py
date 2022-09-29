import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from numpy.random import rand
from typing import List
from utils import *

# ===== Standard max log D Training=====

# constants for soft and noisy label generation
FLIP_CHANCE = 1.05
NOISE_LEVEL = 0.4
OFFSET = 0.8

def max_log_d_epoch(
    dataloader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optim: optim.Adam,
    d_optim: optim.Adam,
    criterion: nn.BCELoss,
    noise_dim: int,
    epoch_adv_loss: float,
    epoch_d_loss: float,
    real_accs: List,
    fake_accs: List):

    for real_data in tqdm(dataloader):
        batch_size = real_data.size(0)
        real_data = real_data.to(device)

        # prepare labels
        flips = np.floor(rand(batch_size) * FLIP_CHANCE) * OFFSET
        real_labels = rand(batch_size).astype(np.float32) * NOISE_LEVEL + OFFSET
        fake_labels = rand(batch_size).astype(np.float32) * NOISE_LEVEL
        real_labels -= flips
        fake_labels += flips
        real_labels = torch.from_numpy(real_data).to(device)
        fake_labels = torch.from_numpy(fake_data).to(device)

        # train generator
        noise = latent_vector(batch_size, noise_dim)
        fake_data = generator(noise)
        predict_fake = discriminator(fake_data)

        g_adv_loss = criterion(predict_fake, real_labels)
        epoch_adv_loss += g_adv_loss

        g_optim.zero_grad()
        g_adv_loss.backward()
        g_optim.step()
        
        # train discriminator
        real_data = real_data.to(device)
        predict_real = discriminator(real_data)

        noise = latent_vector(batch_size, noise_dim)
        fake_data = generator(noise)
        predict_fake = discriminator(fake_data)

        real_loss = criterion(predict_real, real_labels)
        fake_loss = criterion(predict_fake, fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        epoch_d_loss += d_loss

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
        
        # track accuracies
        d_real_acc = torch.ge(predict_real.squeeze(), 0.5).float()
        real_accs.append(d_real_acc.mean().item())
        d_fake_acc = torch.lt(predict_fake.squeeze(), 0.5).float()
        fake_accs.append(d_fake_acc.mean().item())
