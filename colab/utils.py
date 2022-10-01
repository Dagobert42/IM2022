import torch
from torch.autograd import Variable
from torch.utils import data
import pickle5 as pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def latent_vector(batch_size, noise_dim):
    x = torch.FloatTensor(batch_size, noise_dim).normal_(0.0, 1.0).to(device)
    return Variable(x)

class ArtificialStructuresDataset(data.Dataset):
    def __init__(self, data_path, device):
        with open(data_path,'rb') as f:
            # send data to GPU if possible
            self.data = pickle.load(f)

    def __getitem__(self, index):
        return self.data[index].to(device)

    def __len__(self):
        return len(self.data)

def save_samples(samples, path, epoch):
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.05, hspace=0.05)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    for i, sample in enumerate(samples):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        # y is the height dim in Minetest
        ax.scatter(x, z, y, zdir='z', c=sample[x,y,z], cmap='jet', marker="h", alpha=0.8, linewidth=0.)
        ax.axis('off')
    plt.savefig(path + '/sample_@epoch{}.png'.format(str(epoch)))
    plt.close()
