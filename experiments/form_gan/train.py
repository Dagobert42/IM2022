import os
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
import datetime
import time
import pickle5 as pickle
from utils import *
from max_log_d import *
from w_gp import *
from google.colab import files

def start_training(
    generator: nn.Module,
    discriminator: nn.Module,
    data_path: str,
    learning_rate: float,
    n_epochs: float,
    t: str,
    batch_size: int,
    noise_dim: int
    ):
    dir = os.getcwd()
    dataset = ArtificialStructuresDataset(data_path, device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
        )
    model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_save_path = dir + "/" + model_uid
    print("Using device: ", device)

    generator.to(device)
    discriminator.to(device)

    g_optim = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    if t == "log_d":
        d_scheduler = MultiStepLR(d_optim, milestones=[10, 25, 50, 100], gamma=0.5)
    
    discriminator.train()
    generator.train()

    criterion = nn.BCELoss()

    # track performance measures for analysis
    d_losses = []
    adv_losses = []
    epoch_real_accs = []
    epoch_fake_accs = []
    epoch_mean_accs = []

    def save_states():
        torch.save(generator.state_dict(), model_save_path + '_G.pth')
        torch.save(discriminator.state_dict(), model_save_path + '_D.pth')
        log = {
            "dl": d_losses,
            "al": adv_losses,
            "ra": epoch_real_accs,
            "fa": epoch_fake_accs,
            "ma": epoch_mean_accs,
            }
        with open(model_save_path + '_log.pkl', 'wb') as f:
            pickle.dump(log, f, protocol=pickle.DEFAULT_PROTOCOL)
        print('Models and training statistics saved...')

    for epoch in range(n_epochs):
        start = time.time()

        epoch_adv_losses = []
        epoch_d_losses = []
        real_accs = []
        fake_accs = []

        if t == "log_d":
            max_log_d_epoch(
                dataloader,
                generator,
                discriminator,
                g_optim,
                d_optim,
                criterion,
                noise_dim,
                epoch_adv_losses,
                epoch_d_losses,
                real_accs,
                fake_accs
                )

        if t == "w_gp":
            wgan_gp_epoch(
                dataloader,
                generator,
                discriminator,
                g_optim,
                d_optim,
                noise_dim,
                epoch_adv_losses,
                epoch_d_losses,
                real_accs,
                fake_accs
                )
            
        # chart losses
        epoch_d_loss = sum(epoch_d_losses) / len(dataset)
        d_losses.append(epoch_d_loss)
        epoch_adv_loss = sum(epoch_adv_losses) / len(dataset)
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

        print(
            'Epoch-{} , D Loss : {:.4}, G Adv Loss : {:.4}, Real Acc : {:.4}, Fake Acc : {:.4}, Mean Acc : {:.4}'.format(
            epoch + 1,
            epoch_d_loss,
            epoch_adv_loss,
            epoch_real_acc,
            epoch_fake_acc,
            epoch_mean_acc)
            )
        print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

        if epoch % 25 == 0:
            # number of outputs to generate
            NUM_SAMPLES = 4
            x = latent_vector(NUM_SAMPLES, noise_dim)
            samples = generator(x).cpu().data[:NUM_SAMPLES].squeeze().numpy()
            samples = np.fix(samples * 32.0)
            save_samples(samples, dir, epoch)
            save_states()
        
        if t == "log_d":
            d_scheduler.step()

    save_states()
    files.download(model_save_path + '_G.pth')
    files.download(model_save_path + '_D.pth')
    files.download(model_save_path + '_log.pkl')
