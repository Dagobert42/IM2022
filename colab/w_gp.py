import torch
from torch import optim
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
from utils import *

# ===== Wasserstein Gradient Penalty Training =====

# constants for Wasserstein gradient penalty
LAMBDA = 10.0
def gradient_penalty(discriminator, real_data, fake_data):
    # random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_data.size(0), 1, 1, 1), dtype=torch.float).to(device)

    # random interpolation between real and fake samples
    interpolates = alpha * real_data + ((1 - alpha) * fake_data).to(device)
    interpolates = Variable(interpolates, requires_grad=True)

    score_interpolates = discriminator(interpolates)

    # gradient over interpolates
    grad_out = torch.ones(score_interpolates.size()).to(device)
    gradients = autograd.grad(
        outputs=score_interpolates,
        inputs=interpolates,
        grad_outputs=grad_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

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

        # train discriminator
        real_data = Variable(real_data)
        predict_real = discriminator(real_data)

        noise = latent_vector(batch_size, noise_dim)
        fake_data = generator(noise)
        predict_fake = discriminator(fake_data)
        
        gp = gradient_penalty(discriminator, real_data.data, fake_data.data)
        d_loss = torch.mean(predict_fake) - torch.mean(predict_real) + gp
        # wasserstein_distance = predict_real - predict_fake
        epoch_d_losses.append(predict_fake.mean().item() - predict_real.mean().item())
        
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
        

        g_optim.zero_grad()
        if i % TRAIN_G_EVERY == 0:
            # train generator
            noise = latent_vector(batch_size, noise_dim)
            fake_data = generator(noise)

            predict_fake = discriminator(fake_data)
            g_adv_loss = -torch.mean(predict_fake)
            epoch_adv_losses.append(predict_fake.mean().item())

            g_adv_loss.backward()
            g_optim.step()

        # track accuracies
        d_real_acc = torch.ge(predict_real.squeeze(), 0.5).float()
        real_accs.append(d_real_acc.mean().item())
        d_fake_acc = torch.lt(predict_fake.squeeze(), 0.5).float()
        fake_accs.append(d_fake_acc.mean().item())
