import torch
from torch import optim
from torch import nn
import datetime
import time
from tqdm import tqdm
import numpy as np
from numpy.random import rand
import pickle
from utils import *
from minetest.adapter import *
from torch.utils.data import DataLoader
import os
from random import choices

def fit(
    Generator: nn.Module,
    Discriminator: nn.Module,
    name: str,
    data_path: str,
    learning_rate: float,
    n_epochs: float,
    batch_size: int,
    noise_dim: int,
    live_mode: bool,
    device: torch.device
    ):
    dataset = ArtificialStructuresDataset(data_path, device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    model_uid = name + datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_save_path = '/models/states/' + model_uid
    print("Using device: ", device)

    if live_mode:
        mta = MinetestAdapter()
        mta.connect()
        print("Printing live to Minetest.")

    Generator.to(device)
    Discriminator.to(device)

    g_optim = optim.Adam(Generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optim = optim.Adam(Discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion = torch.nn.BCELoss()

    Discriminator.train()
    Generator.train()

    # define constants for label generation
    FLIP_CHANCE = 1.05
    NOISE_LEVEL = 0.4
    OFFSET = 0.8

    # track performance measures for analysis
    d_losses = []
    adv_losses = []
    epoch_real_accs = []
    epoch_fake_accs = []
    epoch_mean_accs = []

    for epoch in range(n_epochs):
        start = time.time()

        epoch_adv_loss = 0.0
        epoch_d_loss = 0.0
        real_accs = []
        fake_accs = []
        for samples in tqdm(dataloader):
            batch_dim = samples.size(0)

            # make use of soft and noisy labels
            flips = np.floor(rand(batch_dim) * FLIP_CHANCE) * OFFSET
            real = rand(batch_dim).astype(np.float32) * NOISE_LEVEL + OFFSET
            fake = rand(batch_dim).astype(np.float32) * NOISE_LEVEL
            real -= flips
            fake += flips
            real_labels = torch.from_numpy(real).to(device)
            fake_labels = torch.from_numpy(fake).to(device)

            samples = samples.to(device)
            predict_samples = Discriminator(samples)

            ###### Train Generator ######
            noise = torch.Tensor(batch_dim, noise_dim).normal_(0.0, 1.0).to(device)
            fakes = Generator(noise)
            predict_fakes = Discriminator(fakes)

            g_adv_loss = criterion(predict_fakes, real_labels)
            epoch_adv_loss += g_adv_loss

            g_optim.zero_grad()
            g_adv_loss.backward()
            g_optim.step()
            
            ###### Train Discriminator ######
            noise = torch.Tensor(batch_dim, noise_dim).normal_(0.0, 1.0).to(device)
            fakes = Generator(noise)
            predict_fakes = Discriminator(fakes)

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

        # chart losses
        epoch_d_loss /= len(dataset)
        d_losses.append(epoch_d_loss)
        epoch_adv_loss /= len(dataset)
        adv_losses.append(epoch_adv_loss)

        # chart accuracies
        epoch_real_acc = sum(real_accs) / len(real_accs)
        epoch_real_accs.append(epoch_real_acc)
        epoch_fake_acc = sum(fake_accs) / len(fake_accs)
        epoch_fake_accs.append(epoch_fake_acc)
        epoch_mean_acc = (epoch_real_acc + epoch_fake_acc) / 2
        epoch_mean_accs.append(epoch_mean_acc)

        end = time.time()
        epoch_time = end - start

        print('Epoch-{} , D Loss : {:.4}, G Adv Loss : {:.4}, Mean Acc : {:.4}'.format(
            epoch + 1,
            epoch_d_loss,
            epoch_adv_loss,
            epoch_mean_acc))
        print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

    dir = os.getcwd()
    torch.save(Generator.state_dict(), dir + model_save_path + '_G.pth')
    torch.save(Discriminator.state_dict(), dir + model_save_path + '_D.pth')
    log = {
        "dl": d_losses,
        "al": adv_losses,
        "ra": epoch_real_accs,
        "fa": epoch_fake_accs,
        "ma": epoch_mean_accs,
        }
    with open(dir + '/models/logs/'+ model_uid + '_log.pkl', 'wb') as f:
        pickle.dump(log, f, protocol=pickle.DEFAULT_PROTOCOL)
    print('Models and training statistics saved...')
