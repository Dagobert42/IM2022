import torch
from torch import Tensor
from torch import optim
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.random import rand
from typing import List
from utils import *

# ===== Wasserstein Divergence Training =====

# constants for Wasserstein divergence gradient penalty
K = 2
P = 6
def gradient_penalty(
    real_data: Variable,
    score_real: Variable,
    fake_data: Variable,
    score_fake: Variable
    ):
    real_grad_outputs = torch.full((real_data.size(0),), 1, dtype=torch.float32, requires_grad=False, device=device)
    fake_grad_outputs = torch.full((fake_data.size(0),), 1, dtype=torch.float32, requires_grad=False, device=device)

    real_gradient = torch.autograd.grad(
        outputs=score_real,
        inputs=real_data,
        grad_outputs=real_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    fake_gradient = torch.autograd.grad(
        outputs=score_fake,
        inputs=fake_data,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    real_gradient_norm = real_gradient.view(real_gradient.size(0), -1).pow(2).sum(1) ** (P / 2)
    fake_gradient_norm = fake_gradient.view(fake_gradient.size(0), -1).pow(2).sum(1) ** (P / 2)

    return torch.mean(real_gradient_norm + fake_gradient_norm) * K / 2

# constant for generator update
TRAIN_G_EVERY = 5
def wgan_div_epoch(
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
        
        # train discriminator (critic)
        d_optim.zero_grad()

        noise = latent_vector(batch_size, noise_dim)
        real_data = Variable(real_data[0].type(Tensor), requires_grad=True)
        real_data = real_data.to(device)
        fake_data = generator(noise)

        score_real = discriminator(real_data)
        score_fake = discriminator(fake_data)

        grad_penalty = gradient_penalty(
            real_data,
            score_real,
            fake_data,
            score_fake)

        d_loss = torch.mean(score_real) - torch.mean(score_fake) + grad_penalty
        epoch_d_losses.append(score_real.mean().item() - score_fake.mean().item())
        
        d_loss.backward()
        d_optim.step()

        # train generator (a bit less often than critic)
        if (i + 1) % TRAIN_G_EVERY == 0:
            noise = latent_vector(batch_size, noise_dim).to(device)
            fakes = generator(noise)
            score_fake = discriminator(fakes)
            g_loss = torch.mean(score_fake)
            epoch_adv_losses += score_fake.mean().item()

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
        
        # there are no real_data accuracies here, the discriminator just scores
        # the input on a scale of validity:
        #   fakeness - <---> + realness
        # but we can still chart whether the tendencies are correct
        d_real_acc = torch.ge(score_fake.squeeze(), 0.0).float()
        real_accs.append(d_real_acc.mean().item())
        d_fake_acc = torch.lt(score_real.squeeze(), 0.0).float()
        fake_accs.append(d_fake_acc.mean().item())
