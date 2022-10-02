import torch
from torch import optim
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from typing import List
from utils import *

# ===== Wasserstein Gradient Penalty Training =====

# constants for Wasserstein gradient penalty
LAMBDA = 10.0
def gradient_penalty(real_data, predict_real):
    gradients, *_ = autograd.grad(outputs=predict_real,
        inputs=real_data,
        grad_outputs=torch.ones_like(predict_real),
        create_graph=True
        )
    gradients = gradients.reshape(real_data.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

TRAIN_G_EVERY = 5
def wgan_gp_epoch(
    dataloader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optim: optim.Adam,
    d_optim: optim.Adam,
    noise_dim: int,
    epoch_adv_losses: float,
    epoch_d_losses: float,
    real_accs: List,
    fake_accs: List
    ):
    for i, real_data in enumerate(tqdm(dataloader)):
        batch_size = real_data.size(0)

        real_data.requires_grad(True)
        # train discriminator
        predict_real = discriminator(real_data)
        d_real_loss = F.relu(1 - predict_real).mean()

        noise = latent_vector(batch_size, noise_dim)
        fake_data = generator(noise)
        predict_fake = discriminator(fake_data.detach())
        d_fake_loss = F.relu(1 + predict_fake).mean()
        
        gp = gradient_penalty(real_data, predict_real)
        d_loss = d_real_loss + d_fake_loss + gp
        epoch_d_losses.append(d_real_loss.item() + d_fake_loss.item())
        
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
        
        g_optim.zero_grad()
        if i % TRAIN_G_EVERY == 0:
            # train generator
            noise = latent_vector(batch_size, noise_dim)
            fake_data = generator(noise)

            predict_fake = discriminator(fake_data)
            g_adv_loss = -predict_fake.mean()
            epoch_adv_losses.append(-predict_fake.mean().item())

            g_adv_loss.backward()
            g_optim.step()

        # track accuracies
        d_real_acc = torch.ge(predict_real.squeeze(), 0.5).float()
        real_accs.append(d_real_acc.mean().item())
        d_fake_acc = torch.lt(predict_fake.squeeze(), 0.5).float()
        fake_accs.append(d_fake_acc.mean().item())
