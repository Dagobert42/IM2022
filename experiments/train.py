import torch
from torch import optim
from torch import nn
import datetime
import time
from tqdm import tqdm
import numpy as np
import pickle
from data import *
from minetest.adapter import *
from torch.utils.data import DataLoader
import os

def fit(
    Discriminator: nn.Module,
    Generator: nn.Module,
    name: str,
    data_path: str,
    generator_lr: float,
    discriminator_lr: float,
    n_epochs: float,
    batch_size: int,
    noise_Dim: int,
    target_accuracy: float,
    beta: tuple,
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

    Discriminator.to(device)
    Generator.to(device)

    D_optim = optim.Adam(Discriminator.parameters(), lr=discriminator_lr, betas=beta)
    G_optim = optim.Adam(Generator.parameters(), lr=generator_lr, betas=beta)

    D_criterion = nn.MSELoss()
    G_criterion = nn.L1Loss()

    Discriminator.train()
    Generator.train()

    # track performance measures for analysis
    D_losses = []
    adversarial_losses = []
    epoch_real_accs = []
    epoch_fake_accs = []
    epoch_mean_accs = []
    for epoch in range(n_epochs):
        start = time.time()

        current_loss_G = 0.0
        current_loss_D = 0.0
        current_adv_loss_G = 0.0
        real_accs = []
        fake_accs = []
        for X in tqdm(dataloader):
            X = X.to(device)
            batch_len = X.size(0)

            ###### train Discriminator ######
            noise = noise_batch(batch_len, noise_Dim, device)
            predictions_real = Discriminator(X)

            fakes = Generator(noise)
            predictions_fakes = Discriminator(fakes)

            real_labels = torch.ones_like(predictions_real).to(device)
            fake_labels = torch.zeros_like(predictions_fakes).to(device)

            D_real_loss = D_criterion(predictions_real, real_labels)
            D_fake_loss = D_criterion(predictions_fakes, fake_labels)
            D_loss = D_real_loss + D_fake_loss
            
            # Discriminator accuracy -> compare Discriminator(X) with 0.5 element-wise
            D_real_acc = torch.ge(predictions_real.squeeze(), 0.5).float()
            real_accs.append(torch.mean(D_real_acc))
            D_fake_acc = torch.lt(predictions_fakes.squeeze(), 0.5).float()
            fake_accs.append(torch.mean(D_fake_acc))
            D_total_acc = torch.mean(torch.cat((D_real_acc, D_fake_acc), 0))

            if D_total_acc < target_accuracy:
                Discriminator.zero_Grad()
                D_loss.backward()
                D_optim.step()

            ###### train Generator ######
            noise = noise_batch(batch_len, noise_Dim, device)
            fakes = Generator(noise)
            predictions_fakes = Discriminator(fakes)

            adversarial_loss = D_criterion(predictions_fakes, real_labels)
            recon_G_loss = G_criterion(fakes, X)

            Discriminator.zero_Grad()
            Generator.zero_Grad()
            adversarial_loss.backward()
            G_optim.step()

            current_loss_G += recon_G_loss.item() * batch_len
            current_loss_D += D_loss.item() * batch_len

            current_adv_loss_G += adversarial_loss.item() * batch_len

        # chart losses
        epoch_loss_D = current_loss_D / len(dataset)
        epoch_adv_loss_G = current_adv_loss_G / len(dataset)
        D_losses.append(epoch_loss_D)
        adversarial_losses.append(epoch_adv_loss_G)

        # chart accuracies
        epoch_real_acc = sum(real_accs) / len(real_accs)
        epoch_real_accs.append(epoch_real_acc)
        epoch_fake_acc = sum(fake_accs) / len(fake_accs)
        epoch_fake_accs.append(epoch_fake_acc)
        epoch_mean_acc = (epoch_real_acc + epoch_fake_acc) / 2
        epoch_mean_accs.append(epoch_mean_acc)

        end = time.time()
        epoch_time = end - start

        print('Epoch-{} , Discriminator(x) : {:.4}, Discriminator(Generator(x)) : {:.4}'.format(
            epoch + 1,
            epoch_loss_D,
            epoch_adv_loss_G))
        print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

        def save_everything():
            dir = os.getcwd()
            torch.save(Generator.state_Dict(), dir + model_save_path + '_G.pth')
            torch.save(Discriminator.state_Dict(), dir + model_save_path + '_D.pth')
            log = {
                "dl": D_losses,
                "al": adversarial_losses,
                "ra": epoch_real_accs,
                "fa": epoch_fake_accs,
                "ma": epoch_mean_accs,
                }
            with open(dir + '/models/logs/'+ model_uid + '_log.pkl', 'wb') as f:
                pickle.dump(log, f, protocol=pickle.DEFAULT_PROTOCOL)
            print('Models and training statistics saved...')

        if epoch % 10 == 0:
            save_everything()
    save_everything()
