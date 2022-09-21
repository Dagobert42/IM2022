import torch
from torch import optim
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from numpy.random import rand
from typing import List
from utils import *

# ===== Deep Convolutional GAN =====

# constants for soft and noisy label generation
FLIP_CHANCE = 1.05
NOISE_LEVEL = 0.4
OFFSET = 0.8

def dcgan_epoch(
    dataloader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optim: optim.Adam,
    d_optim: optim.Adam,
    criterion: nn.BCELoss,
    noise_dim: int,
    device: torch.device,
    epoch_adv_loss: float,
    epoch_d_loss: float,
    real_accs: List,
    fake_accs: List
    ):
    for samples in tqdm(dataloader):
        batch_dim = samples.size(0)

        # prepare labels
        flips = np.floor(rand(batch_dim) * FLIP_CHANCE) * OFFSET
        real = rand(batch_dim).astype(np.float32) * NOISE_LEVEL + OFFSET
        fake = rand(batch_dim).astype(np.float32) * NOISE_LEVEL
        real -= flips
        fake += flips
        real_labels = torch.from_numpy(real).to(device)
        fake_labels = torch.from_numpy(fake).to(device)

        # train generator
        noise = Tensor(batch_dim, noise_dim).normal_(0.0, 1.0)
        fakes = generator(noise)
        predict_fakes = discriminator(fakes)

        g_adv_loss = criterion(predict_fakes, real_labels)
        epoch_adv_loss += g_adv_loss

        g_optim.zero_grad()
        g_adv_loss.backward()
        g_optim.step()
        
        # train discriminator
        samples = samples.to(device)
        predict_samples = discriminator(samples)

        noise = Tensor(batch_dim, noise_dim).normal_(0.0, 1.0)
        fakes = generator(noise)
        predict_fakes = discriminator(fakes)

        real_loss = criterion(predict_samples, real_labels)
        fake_loss = criterion(predict_fakes, fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        epoch_d_loss += d_loss

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
        
        # track accuracies
        d_real_acc = torch.ge(predict_samples.squeeze(), 0.5).float()
        real_accs.append(torch.mean(d_real_acc))
        d_fake_acc = torch.lt(predict_fakes.squeeze(), 0.5).float()
        fake_accs.append(torch.mean(d_fake_acc))


# ===== Wasserstein Divergence GAN =====

# constants for WGAN div's gradient penalty
K = 2
P = 6

def gradient_penalty(
    originals: Variable,
    critique_samples: Variable,
    fakes: Variable,
    critique_fakes: Variable
    ):
    real_grad_out = torch.full(
        (originals.size(0),), 1, dtype=torch.float32, requires_grad=False, device=device)
    print('original', critique_samples.size())
    print('grad', real_grad_out.size())
    real_grad = autograd.grad(
        critique_samples,
        originals,
        real_grad_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (P / 2)

    fake_grad_out = torch.full(
        (fakes.size(0),), 1, dtype=torch.float32, requires_grad=False, device=device)
    fake_grad = autograd.grad(
        critique_fakes,
        fakes,
        fake_grad_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (P / 2)

    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * K / 2

    return -torch.mean(critique_samples) + torch.mean(critique_fakes) + div_gp

# constant for generator update
TRAIN_G_EVERY = 5

def wgan_div_epoch(
    dataloader: DataLoader,
    generator: nn.Module,
    critic: nn.Module,
    g_optim: optim.Adam,
    d_optim: optim.Adam,
    noise_dim: int,
    epoch_adv_loss: float,
    epoch_d_loss: float,
    real_accs: List,
    fake_accs: List
    ):
    for i, samples in enumerate(tqdm(dataloader)):
        batch_dim = samples.size(0)
        
        # train discriminator (critic)
        noise = Variable(Tensor(batch_dim, noise_dim).normal_(0.0, 1.0))
        real_samples = Variable(samples.type(Tensor), requires_grad=True)
        fakes = generator(noise)

        critique_samples = critic(real_samples)
        critique_fakes = critic(fakes)

        d_loss = gradient_penalty(
            real_samples,
            critique_samples,
            fakes,
            critique_fakes
            )
        epoch_d_loss += d_loss
        
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        # train generator (a bit less often than critic)
        if (i + 1) % TRAIN_G_EVERY == 0:
            noise = Variable(Tensor(batch_dim, noise_dim).normal_(0.0, 1.0))
            fakes = generator(noise)
            critique_fakes = critic(fakes)
            g_loss = -torch.mean(critique_fakes)
            epoch_adv_loss += g_loss

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
        
        # there are no real accuracies here, the critic just scores
        # the input on a scale of:
        #   fakeness (negative values) <---> realness (positive values)
        # but we can still chart whether the tendencies are correct
        d_real_acc = torch.ge(critique_fakes.squeeze(), 0.0).float()
        real_accs.append(torch.mean(d_real_acc))
        d_fake_acc = torch.lt(critique_samples.squeeze(), 0.0).float()
        fake_accs.append(torch.mean(d_fake_acc))