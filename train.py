import os
import torch
from torch import optim
from torch import nn
import datetime
import time
import pickle
from utils import *
from minetest.adapter import *
from epochs import *

def start_training(
    generator: nn.Module,
    discriminator: nn.Module,
    model_type: ModelType,
    data_path: str,
    learning_rate: float,
    n_epochs: float,
    batch_size: int,
    noise_dim: int,
    live_mode: bool
    ):
    dataset = ArtificialStructuresDataset(data_path, device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_save_path = '/models/states/' + model_uid
    print("Using device: ", device)

    if live_mode:
        mta = MinetestAdapter()
        mta.connect()
        print("Printing live to Minetest.")

    generator.to(device)
    discriminator.to(device)

    g_optim = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    discriminator.train()
    generator.train()

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

        if model_type == ModelType.DCGAN:
            dcgan_epoch(
                dataloader,
                generator,
                discriminator,
                g_optim,
                d_optim,
                criterion,
                noise_dim,
                device,
                epoch_adv_loss,
                epoch_d_loss,
                real_accs,
                fake_accs
                )

        if model_type == ModelType.WGANdiv:
            wgan_div_epoch(
                dataloader,
                generator,
                discriminator,
                g_optim,
                d_optim,
                noise_dim,
                epoch_adv_loss,
                epoch_d_loss,
                real_accs,
                fake_accs
                )
            
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
    torch.save(generator.state_dict(), dir + model_save_path + '_G.pth')
    torch.save(discriminator.state_dict(), dir + model_save_path + '_D.pth')
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
